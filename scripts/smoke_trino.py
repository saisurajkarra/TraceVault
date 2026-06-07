"""Smoke test the multi-engine lakehouse: pyiceberg (Postgres catalog) writes a table to
MinIO; Trino reads the SAME table through the SAME Postgres JDBC catalog. Proves the shared
catalog before the full re-ingest. Run: uv run python scripts/smoke_trino.py
"""

from __future__ import annotations

import pyarrow as pa
import trino
from pyiceberg.catalog.sql import SqlCatalog

from tracevault.config import Settings


def main() -> None:
    s = Settings(catalog_backend="postgres")
    print(f"catalog uri = {s.catalog_uri}")
    cat = SqlCatalog("tracevault", **s.iceberg_catalog_properties())
    cat.create_namespace_if_not_exists("smoke_ns")
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    if ("smoke_ns", "smoke_tbl") in cat.list_tables("smoke_ns"):
        cat.drop_table("smoke_ns.smoke_tbl")
    tbl = cat.create_table("smoke_ns.smoke_tbl", schema=schema)
    tbl.append(pa.table({"id": [1, 2, 3], "name": ["alpha", "beta", "gamma"]}, schema=schema))
    print(f"pyiceberg wrote smoke_ns.smoke_tbl -> {tbl.scan().to_arrow().num_rows} rows on MinIO")

    conn = trino.dbapi.connect(host="localhost", port=8085, user="smoke", catalog="iceberg")
    cur = conn.cursor()
    cur.execute("SHOW SCHEMAS FROM iceberg")
    print("TRINO schemas:", [r[0] for r in cur.fetchall()])
    cur.execute("SELECT count(*) FROM iceberg.smoke_ns.smoke_tbl")
    print("TRINO row count:", cur.fetchone()[0])
    cur.execute("SELECT id, name FROM iceberg.smoke_ns.smoke_tbl ORDER BY id")
    rows = cur.fetchall()
    print("TRINO data:", rows)
    assert rows == [[1, "alpha"], [2, "beta"], [3, "gamma"]], rows

    cat.drop_table("smoke_ns.smoke_tbl")
    print("\nSMOKE PASSED: Trino read pyiceberg's Iceberg table via the shared Postgres catalog.")


if __name__ == "__main__":
    main()
