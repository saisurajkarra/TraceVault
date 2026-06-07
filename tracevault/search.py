"""Semantic search over the real ingested artifacts (LanceDB ANN).

Embeds the query with the same model used at ingest time, runs an approximate
nearest-neighbour search in LanceDB, and returns real artifacts ranked by cosine
similarity. Every hit carries the fields needed to trace it back to its source
(its artifact id, kind, source, and object_uri where one exists).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from .config import Settings
from .embed import Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    id: str
    score: float
    kind: str
    source: str
    repo: str
    actor: str
    title: str
    snippet: str
    object_uri: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Searcher:
    """Holds a loaded embedder + LanceDB table; answers queries with real ranked hits."""

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self.table = embedder.open_table()

    def search(
        self, query: str, *, k: int = 10, kind: str | None = None, repo: str | None = None
    ) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        if self.table.count_rows() == 0:
            return []
        vector = self.embedder.encode_one(query)
        builder = self.table.search(vector, vector_column_name="vector").metric("cosine").limit(k)
        clauses = []
        if kind:
            clauses.append(f"kind = '{_escape(kind)}'")
        if repo:
            clauses.append(f"repo = '{_escape(repo)}'")
        if clauses:
            builder = builder.where(" AND ".join(clauses))
        rows = builder.to_list()
        hits: list[SearchHit] = []
        for row in rows:
            distance = float(row.get("_distance", 1.0))
            hits.append(
                SearchHit(
                    id=row["id"],
                    score=round(max(0.0, 1.0 - distance), 4),  # cosine similarity
                    kind=row.get("kind", ""),
                    source=row.get("source", ""),
                    repo=row.get("repo", ""),
                    actor=row.get("actor", ""),
                    title=row.get("title", ""),
                    snippet=(row.get("text", "") or "")[:400],
                    object_uri=row.get("object_uri", ""),
                )
            )
        return hits


def _escape(value: str) -> str:
    return value.replace("'", "''")
