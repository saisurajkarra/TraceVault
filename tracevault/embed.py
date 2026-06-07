"""Semantic embeddings: real, local, offline (sentence-transformers -> LanceDB).

Each artifact's searchable ``text`` is embedded with all-MiniLM-L6-v2 and upserted
into an embedded LanceDB table keyed by the real artifact id, carrying enough
metadata (kind/source/repo/actor/title/object_uri) to render and trace a hit back to
its source. If the model cannot load, this raises — it NEVER fabricates vectors.
"""

from __future__ import annotations

import logging
from typing import Any

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa

from .config import Settings
from .lakehouse import Lakehouse

logger = logging.getLogger(__name__)

ARTIFACTS_VECTOR_TABLE = "artifacts"


class EmbedError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def _vector_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
            ("kind", pa.string()),
            ("source", pa.string()),
            ("repo", pa.string()),
            ("actor", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
            ("object_uri", pa.string()),
            ("created_at", pa.string()),
        ]
    )


class Embedder:
    """Loads the model once and embeds text; fails loudly if the model is unavailable."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = self._load_model(settings.embedding_model)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        logger.info("Loaded embedding model %r (dim=%d)", settings.embedding_model, self.dim)

    @staticmethod
    def _load_model(name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbedError(
                "sentence-transformers is not installed; embeddings are required and "
                "there is no fallback. Install dependencies with `uv sync`."
            ) from exc
        try:
            return SentenceTransformer(name, device="cpu")
        except Exception as exc:
            raise EmbedError(
                f"Could not load embedding model {name!r}. The first run downloads it; "
                f"check your network or a pre-populated model cache. Underlying error: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()

    # --- LanceDB ---

    def open_table(self) -> Any:
        db = lancedb.connect(str(self.settings.lancedb_path))
        try:
            return db.open_table(ARTIFACTS_VECTOR_TABLE)
        except Exception:
            # Doesn't exist yet — create it. exist_ok guards against a concurrent create.
            return db.create_table(
                ARTIFACTS_VECTOR_TABLE, schema=_vector_schema(self.dim), exist_ok=True
            )

    def embed_artifacts(self, lake: Lakehouse) -> int:
        """Embed every artifact not yet in LanceDB. Returns the number newly embedded."""
        df = lake.scan_artifacts()
        if df.height == 0:
            logger.info("No artifacts to embed.")
            return 0

        table = self.open_table()
        try:
            existing_ids = set(table.to_arrow().column("id").to_pylist())
        except Exception:
            existing_ids = set()

        todo = df.filter(~pl.col("id").is_in(list(existing_ids))) if existing_ids else df
        if todo.height == 0:
            logger.info("All %d artifacts already embedded.", df.height)
            return 0

        texts = [
            (t or title or "")
            for t, title in zip(todo["text"].to_list(), todo["title"].to_list(), strict=False)
        ]
        logger.info("Embedding %d artifacts...", len(texts))
        vectors = self.encode(texts)

        created = todo["created_at"].to_list()
        records = []
        for i, row in enumerate(todo.iter_rows(named=True)):
            records.append(
                {
                    "id": row["id"],
                    "vector": vectors[i].tolist(),
                    "kind": row["kind"],
                    "source": row["source"],
                    "repo": row["repo"] or "",
                    "actor": row["actor"] or "",
                    "title": (row["title"] or "")[:300],
                    "text": (row["text"] or "")[:2000],
                    "object_uri": row["object_uri"] or "",
                    "created_at": created[i].isoformat() if created[i] is not None else "",
                }
            )
        table.add(records)
        total = len(records) + len(existing_ids)
        logger.info("Embedded %d artifacts into LanceDB (total now %d).", len(records), total)
        return len(records)
