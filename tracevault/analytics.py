"""DuckDB SQL analytics over the lakehouse (silver + gold), for the DE dashboard.

The silver Iceberg tables (`artifacts`, `edges`) and the gold marts are registered into an
in-process DuckDB connection (as Arrow), then queried with SQL. The dashboard `summary()`
runs a fixed set of queries for charts; `run_sql()` exposes a read-only SQL console so you
can query the lakehouse live. The registered tables are snapshots (Arrow copies), so SQL can
never mutate the real lakehouse.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb

from .lakehouse import Lakehouse
from .marts import GOLD_TABLES

logger = logging.getLogger(__name__)


def _connect(lake: Lakehouse) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("artifacts", lake.scan_artifacts().to_arrow())
    con.register("edges", lake.scan_edges().to_arrow())
    for name in GOLD_TABLES:
        df = lake.gold_table(name)
        if df.width:  # only register marts that exist (have columns)
            con.register(name, df.to_arrow())
    return con


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    try:
        cur = con.execute(sql)
    except duckdb.Error as exc:
        logger.warning("analytics query failed (%s): %s", sql.split("FROM")[-1][:40], exc)
        return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def summary(lake: Lakehouse) -> dict[str, Any]:
    """Run the dashboard's fixed analytics queries over the gold marts. Returns chart data."""
    con = _connect(lake)
    try:
        edge_rows = _rows(con, "SELECT count(*) AS n FROM edges")
        return {
            "totals": _rows(con, "SELECT kind, count(*) AS n FROM artifacts GROUP BY kind ORDER BY n DESC"),
            "edges_total": edge_rows[0]["n"] if edge_rows else 0,
            "top_authors": _rows(
                con,
                "SELECT name, commits, ai_sessions, files_touched FROM gold_author_activity "
                "ORDER BY (commits + files_touched + ai_sessions) DESC LIMIT 10",
            ),
            "top_topics": _rows(
                con,
                "SELECT topic, n_artifacts, n_people, n_sessions FROM gold_topic_summary "
                "ORDER BY n_artifacts DESC LIMIT 12",
            ),
            "daily": _rows(
                con,
                "SELECT day, sum(count) AS n FROM gold_daily_activity GROUP BY day ORDER BY day",
            ),
            "cross_silo": _rows(
                con,
                "SELECT person_a_name, person_b_name, shared_topics, topics FROM gold_cross_silo "
                "ORDER BY shared_topics DESC LIMIT 12",
            ),
            "hotspots": _rows(
                con,
                "SELECT path, modality, times_modified, ai_touches FROM gold_file_hotspots "
                "ORDER BY ai_touches DESC, times_modified DESC LIMIT 12",
            ),
            "modality_mix": _rows(
                con,
                "SELECT modality, count(*) AS n FROM gold_file_hotspots GROUP BY modality ORDER BY n DESC",
            ),
        }
    finally:
        con.close()


def run_sql(lake: Lakehouse, query: str, *, limit: int = 500) -> dict[str, Any]:
    """Run a read-only SQL query over the registered lakehouse tables."""
    q = (query or "").strip().rstrip(";")
    low = q.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    con = _connect(lake)
    try:
        cur = con.execute(f"SELECT * FROM ({q}) LIMIT {int(limit)}")
        cols = [d[0] for d in cur.description]
        rows = [list(map(_jsonable, r)) for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows)}
    except duckdb.Error as exc:
        raise ValueError(f"SQL error: {exc}") from exc
    finally:
        con.close()


def run_trino(settings: Any, query: str, *, limit: int = 500) -> dict[str, Any]:
    """Run a read-only query through Trino (distributed SQL over the shared Iceberg catalog)."""
    import trino  # lazy

    q = (query or "").strip().rstrip(";")
    low = q.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    try:
        conn = trino.dbapi.connect(
            host=settings.trino_host,
            port=settings.trino_port,
            user="tracevault",
            catalog=settings.trino_catalog,
            schema="tracevault",
        )
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM ({q}) LIMIT {int(limit)}")
        cols = [d[0] for d in cur.description]
        rows = [list(map(_jsonable, r)) for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows), "engine": "trino"}
    except Exception as exc:
        raise ValueError(f"Trino error: {exc}") from exc


def tables(lake: Lakehouse) -> list[str]:
    return ["artifacts", "edges", *[t for t in GOLD_TABLES if lake.gold_table(t).width]]


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
