# Contributing to tracevault

tracevault is meant to be maintainable and reliable, not a one-off. PRs welcome.

## Principles (non-negotiable)

- **No synthetic data, ever.** The system only ingests real artifacts the user points it
  at. No mock/sample/fixture data generators, no seeded datasets.
- **Fail loud, never fake.** If MinIO, a model, or a credential is missing, raise a clear,
  actionable error. Never substitute placeholder/canned data or silently degrade.
- **Everything traces to a real source.** Anything shown in the UI must link back to a real
  commit, file, or AI session by path/URI.
- **The lakehouse is real.** Iceberg tables are physically written to MinIO over S3.

## Dev setup

```bash
docker compose up -d          # real MinIO (tests need it)
uv sync --extra dev
uv run tracevault ingest --repo /path/to/a/real/repo
uv run tracevault serve
```

## Before you push

```bash
uv run ruff check tracevault     # lint (and `ruff format` if you add it)
uv run mypy tracevault           # types: the codebase is strict-typed and clean
uv run pytest -q                 # real-input tests against the live MinIO container
```

CI runs all three on every push/PR against a real MinIO service (see
`.github/workflows/ci.yml`). Keep it green.

## Architecture map

| Concern | Module |
|---|---|
| Settings (env-driven, fail-fast) | `config.py` |
| MinIO blob store (content-addressed) | `storage.py` |
| Iceberg tables + scans | `lakehouse.py` |
| Stable ids | `ids.py` |
| Git ingest | `ingest_git.py` |
| Claude Code log ingest | `ingest_ai.py` |
| Multimodal file → language ("speaking") | `extract.py` |
| Ingest any folder (no git) | `ingest_folder.py` |
| Embeddings → LanceDB | `embed.py` |
| Semantic search / RAG | `search.py`, `ask.py` |
| Derived knowledge graph | `graph.py` |
| Gold medallion marts | `marts.py` |
| Dagster orchestration + data-quality checks | `orchestration.py` |
| DuckDB + Trino SQL analytics | `analytics.py` |
| Real-time streaming (Kafka) | `streaming.py` |
| API + UIs (+ Prometheus `/metrics`) | `api.py`, `web/index.html`, `web/dashboard.html` |
| CLI | `cli.py` |

## Performance

The hot paths are already native (Polars/LanceDB = Rust, PyArrow = C++, PyTorch =
C++/CUDA); Python is orchestration glue. Profile before optimizing. If a pure-Python loop
is proven hot, compile that module with PyO3+maturin or mypyc rather than rewriting — keep
CPython as the orchestrator.

## Scale-out roadmap (multi-engine / observability)

The single-operator platform is complete and verified. The natural production scale-out:

- **Distributed SQL (Trino)** + **versioned REST catalog (Nessie/Polaris)** or a **Postgres
  JDBC catalog.** Trino's Iceberg connector needs a catalog it understands (REST or JDBC) —
  not the local SQLite `SqlCatalog`. The clean path is to make the catalog backend
  config-pluggable (`sql` → `postgres`/`rest`) and re-register the tables there; then Trino and
  pyiceberg read the *same* Iceberg tables on MinIO. This is a deliberate migration (the catalog
  pointer store changes), so it should be done and **verified** end-to-end, not shipped as
  unverified compose config — the project's rule is that everything claimed works is real.
- **Observability (Prometheus + Grafana).** Add a `/metrics` endpoint and scrape it; Grafana
  dashboards for ingest throughput, request latency, and lakehouse growth. Additive (does not
  touch the lakehouse).
- **Permissions.** Authentication + per-user authorization + access-scoped search/graph/marts
  for a real multi-tenant org deployment.

## Style

Type hints everywhere; dataclasses/pydantic for models; inject config (don't scatter
constants); log in libraries, display only in CLI/API. ruff- and mypy-clean is required.
