"""Gold medallion marts — analytical Iceberg tables derived from the silver lakehouse.

Medallion mapping:
  bronze   raw bytes in MinIO (blobs/) + raw event records
  silver   the cleaned, typed Iceberg `artifacts` and `edges` tables (the source of truth)
  gold     these aggregate marts, recomputed idempotently (full overwrite) and queried via DuckDB

Five marts:
  gold_author_activity   per person: commits, files touched, AI sessions, first/last activity
  gold_file_hotspots     per file: times modified, co-edit degree, AI touches, modality
  gold_topic_summary     per topic: #artifacts / #people / #files / #sessions
  gold_daily_activity     day × source × kind counts (time series)
  gold_cross_silo        people pairs connected via shared topics ("has anyone done this?")

Everything is real and derived only from the lakehouse — no synthetic data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import polars as pl

from .graph import GraphBuilder
from .ids import person_id
from .lakehouse import Lakehouse

logger = logging.getLogger(__name__)

GOLD_TABLES = [
    "gold_author_activity",
    "gold_file_hotspots",
    "gold_topic_summary",
    "gold_daily_activity",
    "gold_cross_silo",
]


def build_marts(lake: Lakehouse) -> dict[str, int]:
    """Recompute every gold mart from silver. Returns row counts per mart."""
    arts = lake.scan_artifacts()
    edges = lake.scan_edges()
    counts: dict[str, int] = {}
    counts["gold_daily_activity"] = _daily_activity(lake, arts)
    counts["gold_file_hotspots"] = _file_hotspots(lake, arts, edges)
    counts["gold_author_activity"] = _author_activity(lake, arts, edges)
    gb = GraphBuilder(lake)
    counts["gold_topic_summary"] = _topic_summary(lake, gb)
    counts["gold_cross_silo"] = _cross_silo(lake, gb)
    logger.info("Built gold marts: %s", counts)
    return counts


def _write(lake: Lakehouse, name: str, df: pl.DataFrame) -> int:
    lake.overwrite_table(name, df.to_arrow())
    return df.height


def _daily_activity(lake: Lakehouse, arts: pl.DataFrame) -> int:
    schema = {"day": pl.Utf8, "source": pl.Utf8, "kind": pl.Utf8, "count": pl.Int64}
    if arts.height == 0:
        return _write(lake, "gold_daily_activity", pl.DataFrame(schema=schema))
    df = (
        arts.select(
            pl.col("created_at").dt.date().cast(pl.Utf8).alias("day"),
            pl.col("source"),
            pl.col("kind"),
        )
        .group_by(["day", "source", "kind"])
        .len()
        .rename({"len": "count"})
        .sort("day")
    )
    return _write(lake, "gold_daily_activity", df)


def _file_hotspots(lake: Lakehouse, arts: pl.DataFrame, edges: pl.DataFrame) -> int:
    schema = {
        "file_id": pl.Utf8, "repo": pl.Utf8, "path": pl.Utf8, "modality": pl.Utf8,
        "times_modified": pl.Int64, "ai_touches": pl.Int64, "coedit_degree": pl.Int64,
    }
    files = arts.filter(pl.col("kind") == "file")
    if files.height == 0:
        return _write(lake, "gold_file_hotspots", pl.DataFrame(schema=schema))

    base = files.select(
        pl.col("id").alias("file_id"),
        pl.col("repo"),
        pl.col("title").alias("path"),
        pl.col("extra").str.json_path_match("$.modality").alias("modality"),
    )

    def _count(rel: str, col: str, out: str) -> pl.DataFrame:
        sub = edges.filter(pl.col("relation") == rel)
        if sub.height == 0:
            return pl.DataFrame(schema={"file_id": pl.Utf8, out: pl.Int64})
        return sub.group_by(col).len().rename({col: "file_id", "len": out})

    modified = _count("modified", "dst_id", "times_modified")
    ai = _count("ai_touched_file", "dst_id", "ai_touches")
    # co_edited is file<->file; degree = appearances as either endpoint.
    coe = edges.filter(pl.col("relation") == "co_edited")
    if coe.height:
        deg = (
            pl.concat([coe.select(pl.col("src_id").alias("file_id")), coe.select(pl.col("dst_id").alias("file_id"))])
            .group_by("file_id")
            .len()
            .rename({"len": "coedit_degree"})
        )
    else:
        deg = pl.DataFrame(schema={"file_id": pl.Utf8, "coedit_degree": pl.Int64})

    out = (
        base.join(modified, on="file_id", how="left")
        .join(ai, on="file_id", how="left")
        .join(deg, on="file_id", how="left")
        .with_columns(
            pl.col("times_modified").fill_null(0),
            pl.col("ai_touches").fill_null(0),
            pl.col("coedit_degree").fill_null(0),
            pl.col("modality").fill_null("unknown"),
        )
        .sort(["ai_touches", "times_modified"], descending=True)
    )
    return _write(lake, "gold_file_hotspots", out)


def _author_activity(lake: Lakehouse, arts: pl.DataFrame, edges: pl.DataFrame) -> int:
    schema = {
        "person_id": pl.Utf8, "name": pl.Utf8, "commits": pl.Int64, "ai_sessions": pl.Int64,
        "files_touched": pl.Int64, "first_activity": pl.Utf8, "last_activity": pl.Utf8,
    }
    if arts.height == 0:
        return _write(lake, "gold_author_activity", pl.DataFrame(schema=schema))

    id_kind: dict[str, str] = {}
    id_day: dict[str, str] = {}
    name_by_pid: dict[str, str] = {}
    for row in arts.select(["id", "kind", "actor", "created_at"]).iter_rows(named=True):
        id_kind[row["id"]] = row["kind"]
        id_day[row["id"]] = row["created_at"].date().isoformat() if row["created_at"] else ""
        if row["kind"] in ("commit", "ai_session") and row["actor"]:
            name_by_pid.setdefault(person_id(row["actor"]), row["actor"].split("<", 1)[0].strip())

    commits: dict[str, int] = defaultdict(int)
    sessions: dict[str, int] = defaultdict(int)
    days: dict[str, list[str]] = defaultdict(list)
    commit_person: dict[str, str] = {}
    person_files: dict[str, set[str]] = defaultdict(set)

    authored = edges.filter(pl.col("relation") == "authored")
    for e in authored.select(["src_id", "dst_id"]).iter_rows(named=True):
        pid, dst = e["src_id"], e["dst_id"]
        k = id_kind.get(dst)
        if k == "commit":
            commits[pid] += 1
            commit_person[dst] = pid
        elif k == "ai_session":
            sessions[pid] += 1
        if id_day.get(dst):
            days[pid].append(id_day[dst])

    for e in edges.filter(pl.col("relation") == "modified").select(["src_id", "dst_id"]).iter_rows(named=True):
        pid = commit_person.get(e["src_id"])
        if pid:
            person_files[pid].add(e["dst_id"])

    rows = []
    for pid in set(commits) | set(sessions):
        d = sorted(days.get(pid, []))
        rows.append(
            {
                "person_id": pid,
                "name": name_by_pid.get(pid, pid),
                "commits": commits.get(pid, 0),
                "ai_sessions": sessions.get(pid, 0),
                "files_touched": len(person_files.get(pid, set())),
                "first_activity": d[0] if d else "",
                "last_activity": d[-1] if d else "",
            }
        )
    df = pl.DataFrame(rows, schema=schema).sort("commits", descending=True) if rows else pl.DataFrame(schema=schema)
    return _write(lake, "gold_author_activity", df)


def _topic_summary(lake: Lakehouse, gb: GraphBuilder) -> int:
    schema = {
        "topic": pl.Utf8, "n_artifacts": pl.Int64, "n_people": pl.Int64,
        "n_files": pl.Int64, "n_sessions": pl.Int64,
    }
    nodes_by_id = {n["data"]["id"]: n["data"] for n in gb._full.nodes}
    per: dict[str, dict[str, int]] = defaultdict(lambda: {"people": 0, "files": 0, "sessions": 0})
    for e in gb._full.edges:
        if e["data"]["relation"] != "about":
            continue
        topic = nodes_by_id.get(e["data"]["source"], {}).get("label")
        tgt_kind = nodes_by_id.get(e["data"]["target"], {}).get("kind")
        if topic is None:
            continue
        if tgt_kind == "person":
            per[topic]["people"] += 1
        elif tgt_kind == "file":
            per[topic]["files"] += 1
        elif tgt_kind == "ai_session":
            per[topic]["sessions"] += 1
    rows = []
    for n in gb._full.nodes:
        if n["data"]["kind"] != "topic":
            continue
        t = n["data"]["label"]
        rows.append(
            {
                "topic": t,
                "n_artifacts": n["data"]["artifact_count"],
                "n_people": per[t]["people"],
                "n_files": per[t]["files"],
                "n_sessions": per[t]["sessions"],
            }
        )
    df = pl.DataFrame(rows, schema=schema).sort("n_artifacts", descending=True) if rows else pl.DataFrame(schema=schema)
    return _write(lake, "gold_topic_summary", df)


def _cross_silo(lake: Lakehouse, gb: GraphBuilder) -> int:
    schema = {
        "person_a": pl.Utf8, "person_a_name": pl.Utf8, "person_b": pl.Utf8,
        "person_b_name": pl.Utf8, "shared_topics": pl.Int64, "topics": pl.Utf8,
    }
    nodes_by_id = {n["data"]["id"]: n["data"] for n in gb._full.nodes}
    people_per_topic: dict[str, set[str]] = defaultdict(set)
    for e in gb._full.edges:
        if e["data"]["relation"] != "about":
            continue
        topic = nodes_by_id.get(e["data"]["source"], {}).get("label")
        tgt = nodes_by_id.get(e["data"]["target"], {})
        if topic and tgt.get("kind") == "person":
            people_per_topic[topic].add(e["data"]["target"])

    pair_topics: dict[tuple[str, str], list[str]] = defaultdict(list)
    for topic, people in people_per_topic.items():
        for a, b in combinations(sorted(people), 2):
            pair_topics[(a, b)].append(topic)

    rows = []
    for (a, b), topics in pair_topics.items():
        rows.append(
            {
                "person_a": a,
                "person_a_name": gb.person_label.get(a, a),
                "person_b": b,
                "person_b_name": gb.person_label.get(b, b),
                "shared_topics": len(topics),
                "topics": ", ".join(sorted(topics)[:8]),
            }
        )
    df = (
        pl.DataFrame(rows, schema=schema).sort("shared_topics", descending=True)
        if rows
        else pl.DataFrame(schema=schema)
    )
    return _write(lake, "gold_cross_silo", df)
