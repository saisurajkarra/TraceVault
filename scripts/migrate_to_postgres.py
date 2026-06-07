"""Migrate the Iceberg catalog from SQLite to Postgres WITHOUT moving data.

The table data/metadata already live in MinIO. This re-registers each table's current
metadata pointer into the Postgres JDBC catalog, so both the app (postgres backend) and
Trino read the exact same Iceberg tables. Fast — no re-ingest, no re-embed.

Run: uv run python scripts/migrate_to_postgres.py
"""

from __future__ import annotations

from pyiceberg.catalog.sql import SqlCatalog

from tracevault.config import Settings
from tracevault.lakehouse import NAMESPACE


def main() -> None:
    src = SqlCatalog("tracevault", **Settings(catalog_backend="sql").iceberg_catalog_properties())
    dst = SqlCatalog("tracevault", **Settings(catalog_backend="postgres").iceberg_catalog_properties())
    dst.create_namespace_if_not_exists(NAMESPACE)

    for ident in src.list_tables(NAMESPACE):
        name = f"{NAMESPACE}.{ident[1]}"
        meta = src.load_table(name).metadata_location
        try:
            dst.register_table(name, meta)
        except Exception:
            dst.drop_table(name)
            dst.register_table(name, meta)
        rows = dst.load_table(name).scan().to_arrow().num_rows
        print(f"registered {name:42s} -> {rows} rows  ({meta.split('/metadata/')[-1]})")

    print("\nMigration complete. Set TRACEVAULT_CATALOG_BACKEND=postgres to use it.")


if __name__ == "__main__":
    main()
