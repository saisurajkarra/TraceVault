"""tracevault — a real organizational knowledge base on Apache Iceberg + MinIO.

Ingests real Git repositories and real Claude Code session logs, stores raw bytes
in MinIO and traces/metadata in Apache Iceberg tables, builds a semantic search
index (LanceDB) and a people<->files<->topics knowledge graph derived from the
Iceberg tables, and serves a browser UI over it. No synthetic data anywhere.
"""

__version__ = "0.1.0"
