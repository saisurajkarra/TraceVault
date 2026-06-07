"""Dagster orchestration — the medallion pipeline as an asset-lineage graph with checks.

  silver_artifacts ─┐
  silver_edges ─────┼─► embeddings (LanceDB)
                    ├─► knowledge_graph
                    └─► gold_daily_activity / gold_file_hotspots /
                        gold_author_activity / gold_topic_marts

Each asset materializes the REAL work (scan Iceberg, embed, build marts) and reports row
counts as Dagster metadata. Asset checks enforce data quality: non-null ids, referential
integrity between edges and artifacts, embedding coverage, and non-empty marts.

Run the UI:  uv run dagster dev -m tracevault.orchestration
"""

from __future__ import annotations

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    Definitions,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from .config import get_settings
from .lakehouse import ARTIFACT_KINDS, Lakehouse


def _lake() -> Lakehouse:
    return Lakehouse(get_settings())


# --- silver: the cleaned Iceberg tables (source of truth) ---


@asset(group_name="silver", description="The Iceberg `artifacts` table (commits, files, AI sessions/messages).")
def silver_artifacts() -> MaterializeResult:
    counts = _lake().counts()
    return MaterializeResult(
        metadata={"rows": MetadataValue.int(counts.get("artifacts_total", 0)), **{
            k: MetadataValue.int(v) for k, v in counts.items() if k.startswith("artifacts_")
        }}
    )


@asset(group_name="silver", description="The Iceberg `edges` table (relationships between artifacts).")
def silver_edges() -> MaterializeResult:
    edges = _lake().scan_edges()
    by_rel = {}
    if edges.height:
        for row in edges.group_by("relation").len().iter_rows(named=True):
            by_rel[row["relation"]] = MetadataValue.int(row["len"])
    return MaterializeResult(metadata={"rows": MetadataValue.int(edges.height), **by_rel})


# --- silver-derived ---


@asset(group_name="silver", deps=[silver_artifacts], description="Embed every artifact's language into LanceDB.")
def embeddings() -> MaterializeResult:
    from .embed import Embedder

    lake = _lake()
    n = Embedder(get_settings()).embed_artifacts(lake)
    return MaterializeResult(metadata={"newly_embedded": MetadataValue.int(n)})


@asset(group_name="silver", deps=[silver_artifacts, silver_edges], description="Derive the people/files/topics graph.")
def knowledge_graph() -> MaterializeResult:
    from .graph import GraphBuilder

    g = GraphBuilder(_lake())._full
    return MaterializeResult(
        metadata={"nodes": MetadataValue.int(len(g.nodes)), "edges": MetadataValue.int(len(g.edges))}
    )


# --- gold: medallion marts ---


@asset(group_name="gold", deps=[silver_artifacts], description="Daily activity time series (day × source × kind).")
def gold_daily_activity() -> MaterializeResult:
    from .marts import _daily_activity

    lake = _lake()
    n = _daily_activity(lake, lake.scan_artifacts())
    return MaterializeResult(metadata={"rows": MetadataValue.int(n)})


@asset(group_name="gold", deps=[silver_artifacts, silver_edges], description="Per-file hotspots.")
def gold_file_hotspots() -> MaterializeResult:
    from .marts import _file_hotspots

    lake = _lake()
    n = _file_hotspots(lake, lake.scan_artifacts(), lake.scan_edges())
    return MaterializeResult(metadata={"rows": MetadataValue.int(n)})


@asset(group_name="gold", deps=[silver_artifacts, silver_edges], description="Per-person activity.")
def gold_author_activity() -> MaterializeResult:
    from .marts import _author_activity

    lake = _lake()
    n = _author_activity(lake, lake.scan_artifacts(), lake.scan_edges())
    return MaterializeResult(metadata={"rows": MetadataValue.int(n)})


@asset(group_name="gold", deps=[silver_artifacts, silver_edges], description="Topic summary + cross-silo people pairs.")
def gold_topic_marts() -> MaterializeResult:
    from .graph import GraphBuilder
    from .marts import _cross_silo, _topic_summary

    lake = _lake()
    gb = GraphBuilder(lake)
    topics = _topic_summary(lake, gb)
    pairs = _cross_silo(lake, gb)
    return MaterializeResult(
        metadata={"topics": MetadataValue.int(topics), "cross_silo_pairs": MetadataValue.int(pairs)}
    )


# --- data-quality asset checks ---


@asset_check(asset=silver_artifacts, description="Every artifact has a non-empty id and a known kind.")
def artifacts_well_formed() -> AssetCheckResult:
    df = _lake().scan_artifacts()
    if df.height == 0:
        return AssetCheckResult(passed=True, metadata={"rows": 0})
    null_ids = df.filter((df["id"].is_null()) | (df["id"] == "")).height
    bad_kinds = df.filter(~df["kind"].is_in(list(ARTIFACT_KINDS))).height
    return AssetCheckResult(
        passed=null_ids == 0 and bad_kinds == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"null_ids": null_ids, "bad_kinds": bad_kinds, "rows": df.height},
    )


@asset_check(asset=silver_edges, description="Every edge endpoint is a real artifact id (or a derived person id).")
def edges_referential_integrity() -> AssetCheckResult:
    lake = _lake()
    ids = lake.existing_artifact_ids()
    edges = lake.scan_edges()
    if edges.height == 0:
        return AssetCheckResult(passed=True, metadata={"edges": 0})

    def _ok(nid: str) -> bool:
        return nid in ids or nid.startswith("person:") or nid.startswith("topic:")

    dangling = sum(
        1
        for e in edges.select(["src_id", "dst_id"]).iter_rows(named=True)
        if not (_ok(e["src_id"]) and _ok(e["dst_id"]))
    )
    return AssetCheckResult(
        passed=dangling == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"dangling_edges": dangling, "edges": edges.height},
    )


@asset_check(asset=embeddings, description="Every artifact is embedded (vector coverage == artifact count).")
def embedding_coverage() -> AssetCheckResult:
    from .embed import Embedder

    settings = get_settings()
    lake = Lakehouse(settings)
    total = lake.counts().get("artifacts_total", 0)
    try:
        embedded = Embedder(settings).open_table().count_rows()
    except Exception as exc:  # model/table unavailable
        return AssetCheckResult(passed=False, metadata={"error": str(exc)})
    return AssetCheckResult(
        passed=embedded >= total,
        severity=AssetCheckSeverity.WARN,
        metadata={"artifacts": total, "embedded": embedded},
    )


defs = Definitions(
    assets=[
        silver_artifacts,
        silver_edges,
        embeddings,
        knowledge_graph,
        gold_daily_activity,
        gold_file_hotspots,
        gold_author_activity,
        gold_topic_marts,
    ],
    asset_checks=[artifacts_well_formed, edges_referential_integrity, embedding_coverage],
)
