"""The lakehouse: two Apache Iceberg tables whose data lives physically in MinIO.

``artifacts`` holds every searchable item (commits, files, AI sessions, AI messages);
``edges`` holds the relationships between them. The catalog is a local SQLite pointer
file, but the table data/metadata/manifests are written to ``s3://<bucket>/warehouse``
over the S3 protocol — a real lakehouse, not local files pretending to be one.

Everything downstream (embeddings, graph, search) derives from these two tables.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import MonthTransform
from pyiceberg.types import (
    DoubleType,
    NestedField,
    StringType,
    TimestamptzType,
)

from .config import Settings

logger = logging.getLogger(__name__)

NAMESPACE = "tracevault"
ARTIFACTS_TABLE = f"{NAMESPACE}.artifacts"
EDGES_TABLE = f"{NAMESPACE}.edges"

# Allowed enum values, validated on construction so bad data fails loudly.
ARTIFACT_KINDS = {"file", "commit", "ai_session", "ai_message"}
ARTIFACT_SOURCES = {"git", "claude_code"}
EDGE_RELATIONS = {
    "authored",
    "modified",
    "co_edited",
    "ai_touched_file",
    "message_of_session",
}

# --- Iceberg schemas (explicit field ids) ---

ARTIFACTS_SCHEMA = Schema(
    NestedField(1, "id", StringType(), required=True),
    NestedField(2, "kind", StringType(), required=True),
    NestedField(3, "source", StringType(), required=True),
    NestedField(4, "repo", StringType(), required=False),
    NestedField(5, "actor", StringType(), required=False),
    NestedField(6, "created_at", TimestamptzType(), required=True),
    NestedField(7, "title", StringType(), required=False),
    NestedField(8, "text", StringType(), required=False),
    NestedField(9, "content_hash", StringType(), required=False),
    NestedField(10, "object_uri", StringType(), required=False),
    NestedField(11, "extra", StringType(), required=False),
)
ARTIFACTS_PARTITION = PartitionSpec(
    PartitionField(source_id=6, field_id=1000, transform=MonthTransform(), name="created_at_month")
)

EDGES_SCHEMA = Schema(
    NestedField(1, "src_id", StringType(), required=True),
    NestedField(2, "dst_id", StringType(), required=True),
    NestedField(3, "relation", StringType(), required=True),
    NestedField(4, "weight", DoubleType(), required=False),
    NestedField(5, "created_at", TimestamptzType(), required=False),
)


# --- row models ---


@dataclass
class Artifact:
    id: str
    kind: str
    source: str
    created_at: datetime
    repo: str | None = None
    actor: str | None = None
    title: str | None = None
    text: str | None = None
    content_hash: str | None = None
    object_uri: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unknown artifact kind {self.kind!r}; allowed: {ARTIFACT_KINDS}")
        if self.source not in ARTIFACT_SOURCES:
            raise ValueError(f"Unknown source {self.source!r}; allowed: {ARTIFACT_SOURCES}")

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = _to_utc(self.created_at)
        row["extra"] = json.dumps(self.extra, ensure_ascii=False, default=str)
        return row


@dataclass
class Edge:
    src_id: str
    dst_id: str
    relation: str
    weight: float = 1.0
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.relation not in EDGE_RELATIONS:
            raise ValueError(f"Unknown relation {self.relation!r}; allowed: {EDGE_RELATIONS}")

    def to_row(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation": self.relation,
            "weight": float(self.weight),
            "created_at": _to_utc(self.created_at) if self.created_at else None,
        }


def _q(value: str) -> str:
    """Escape single quotes for an Iceberg string row-filter literal."""
    return value.replace("'", "''")


def _to_utc(dt: datetime) -> datetime:
    """Normalize to a tz-aware UTC datetime (Iceberg timestamptz)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class Lakehouse:
    """Owns the Iceberg catalog and the two tables; provides append + scan helpers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        # One process can run the API and a background auto-ingest thread at once; this
        # re-entrant lock serializes catalog access so reads and appends never collide.
        self._lock = threading.RLock()
        self._catalog = SqlCatalog(NAMESPACE, **settings.iceberg_catalog_properties())
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        if (NAMESPACE,) not in self._catalog.list_namespaces():
            logger.info("Creating Iceberg namespace %r", NAMESPACE)
            self._catalog.create_namespace_if_not_exists(NAMESPACE)
        self.artifacts: Table = self._catalog.create_table_if_not_exists(
            ARTIFACTS_TABLE, schema=ARTIFACTS_SCHEMA, partition_spec=ARTIFACTS_PARTITION
        )
        self.edges: Table = self._catalog.create_table_if_not_exists(
            EDGES_TABLE, schema=EDGES_SCHEMA
        )
        logger.info("Iceberg tables ready (warehouse=%s)", self.settings.warehouse)

    # --- writes ---

    def append_artifacts(self, artifacts: list[Artifact]) -> int:
        if not artifacts:
            return 0
        arrow_schema = self.artifacts.schema().as_arrow()
        table = pa.Table.from_pylist([a.to_row() for a in artifacts], schema=arrow_schema)
        with self._lock:
            self.artifacts.append(table)
        logger.info("Appended %d artifacts to Iceberg", len(artifacts))
        return len(artifacts)

    def append_edges(self, edges: list[Edge]) -> int:
        if not edges:
            return 0
        arrow_schema = self.edges.schema().as_arrow()
        table = pa.Table.from_pylist([e.to_row() for e in edges], schema=arrow_schema)
        with self._lock:
            self.edges.append(table)
        logger.info("Appended %d edges to Iceberg", len(edges))
        return len(edges)

    # --- gold / medallion: create-or-replace a derived table from a pyarrow table ---

    def overwrite_table(self, name: str, arrow_table: pa.Table) -> Table:
        """Idempotently (re)compute a derived gold table. Full overwrite each run."""
        ident = f"{NAMESPACE}.{name}"
        with self._lock:
            tbl = self._catalog.create_table_if_not_exists(ident, schema=arrow_table.schema)
            tbl.overwrite(arrow_table)
        return tbl

    def gold_table(self, name: str) -> pl.DataFrame:
        """Read a derived gold table as polars (empty frame if it doesn't exist yet)."""
        ident = f"{NAMESPACE}.{name}"
        with self._lock:
            try:
                tbl = self._catalog.load_table(ident)
            except Exception:
                return pl.DataFrame()
            return _to_polars(tbl.scan().to_arrow())

    def table_snapshots(self, name: str) -> list[dict[str, Any]]:
        """Iceberg snapshot history (time-travel) for a table in the namespace."""
        ident = f"{NAMESPACE}.{name}"
        with self._lock:
            try:
                tbl = self._catalog.load_table(ident)
            except Exception:
                return []
            snapshots = list(tbl.metadata.snapshots)
        out: list[dict[str, Any]] = []
        for snap in snapshots:
            summary = snap.summary
            props: dict[str, Any] = {}
            operation = None
            if summary is not None:
                operation = str(getattr(summary.operation, "value", summary.operation))
                props = dict(getattr(summary, "additional_properties", {}) or {})
            out.append(
                {
                    "snapshot_id": snap.snapshot_id,
                    "timestamp_ms": snap.timestamp_ms,
                    "operation": operation,
                    "added_records": props.get("added-records"),
                    "total_records": props.get("total-records"),
                }
            )
        return out

    # --- reads (return polars). pyiceberg parses string row filters; values here are
    #     our own controlled enums/hash ids, single-quote-escaped defensively. ---

    def scan_artifacts(self, *, kind: str | None = None) -> pl.DataFrame:
        with self._lock:
            scan = self.artifacts.scan(row_filter=f"kind = '{_q(kind)}'") if kind else self.artifacts.scan()
            return _to_polars(scan.to_arrow())

    def scan_edges(self, *, relation: str | None = None) -> pl.DataFrame:
        with self._lock:
            scan = (
                self.edges.scan(row_filter=f"relation = '{_q(relation)}'")
                if relation
                else self.edges.scan()
            )
            return _to_polars(scan.to_arrow())

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            rows = self.artifacts.scan(row_filter=f"id = '{_q(artifact_id)}'").to_arrow().to_pylist()
        return rows[0] if rows else None

    def get_artifacts(self, artifact_ids: list[str]) -> list[dict[str, Any]]:
        if not artifact_ids:
            return []
        joined = ", ".join(f"'{_q(i)}'" for i in artifact_ids)
        with self._lock:
            return self.artifacts.scan(row_filter=f"id IN ({joined})").to_arrow().to_pylist()

    def existing_artifact_ids(self) -> set[str]:
        """Ids already in the lakehouse, used to keep ingestion idempotent."""
        with self._lock:
            arrow = self.artifacts.scan(selected_fields=("id",)).to_arrow()
        return set(arrow.column("id").to_pylist())

    def counts(self, *, repo: str | None = None) -> dict[str, int]:
        with self._lock:
            arts = _to_polars(self.artifacts.scan(selected_fields=("kind", "repo")).to_arrow())
            edges_total = self.edges.scan(selected_fields=("relation",)).to_arrow().num_rows
        if repo and arts.height:
            arts = arts.filter(pl.col("repo") == repo)
        out: dict[str, int] = {"artifacts_total": arts.height}
        if arts.height:
            for row in arts.group_by("kind").len().iter_rows(named=True):
                out[f"artifacts_{row['kind']}"] = row["len"]
        if repo is None:
            out["edges_total"] = edges_total
        return out

    def repos(self) -> list[dict[str, Any]]:
        """Distinct ingested git projects with their commit/file/session counts."""
        with self._lock:
            df = _to_polars(self.artifacts.scan(selected_fields=("repo", "kind")).to_arrow())
        if df.height == 0:
            return []
        grouped: dict[str, dict[str, int]] = {}
        for row in df.filter(pl.col("repo").is_not_null()).group_by(["repo", "kind"]).len().iter_rows(
            named=True
        ):
            grouped.setdefault(row["repo"], {})[row["kind"]] = row["len"]
        out = []
        for repo, kinds in grouped.items():
            commits, files = kinds.get("commit", 0), kinds.get("file", 0)
            if commits or files:  # a real git project (not a stray AI-only repo label)
                out.append(
                    {
                        "repo": repo,
                        "commits": commits,
                        "files": files,
                        "ai_sessions": kinds.get("ai_session", 0),
                    }
                )
        return sorted(out, key=lambda d: -(d["commits"] + d["files"]))


def _to_polars(arrow: pa.Table) -> pl.DataFrame:
    if arrow.num_rows == 0:
        # Preserve column names even when empty so downstream selects don't KeyError.
        return pl.DataFrame(schema={name: pl.Utf8 for name in arrow.schema.names})
    df = pl.from_arrow(arrow)
    assert isinstance(df, pl.DataFrame)
    return df
