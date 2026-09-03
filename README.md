## Star History

<a href="https://www.star-history.com/?repos=saisurajkarra%2FTraceVault&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=saisurajkarra/TraceVault&type=date&theme=dark&legend=top-left&sealed_token=Fjgh56V44yPau6FcXHw__y3Ckd8mN0mQG7ovzSN_4hqZsrWQ6YANzTcGECeu3DPJZA14sZk3AuT66lt1kfX141zODUoYc5wTYG8c7uMr_kvIcIElfVn-5-rYHQ" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=saisurajkarra/TraceVault&type=date&legend=top-left&sealed_token=Fjgh56V44yPau6FcXHw__y3Ckd8mN0mQG7ovzSN_4hqZsrWQ6YANzTcGECeu3DPJZA14sZk3AuT66lt1kfX141zODUoYc5wTYG8c7uMr_kvIcIElfVn-5-rYHQ" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=saisurajkarra/TraceVault&type=date&legend=top-left&sealed_token=Fjgh56V44yPau6FcXHw__y3Ckd8mN0mQG7ovzSN_4hqZsrWQ6YANzTcGECeu3DPJZA14sZk3AuT66lt1kfX141zODUoYc5wTYG8c7uMr_kvIcIElfVn-5-rYHQ" />
 </picture>
</a>

# tracevault — a real data platform on Apache Iceberg + MinIO

tracevault is a **modern data-engineering platform** that turns an organization's *actual*
work — a Git repo **or any folder** (most work is never committed), plus real Claude Code
session logs — into a governed lakehouse you can search, analyze, orchestrate, and stream.

It is built on the real modern data stack, end to end, on data you own:

| Capability | Technology |
|---|---|
| Object storage (raw bytes + warehouse) | **MinIO** (S3) |
| Open table format / lakehouse (ACID, time-travel, partitioning) | **Apache Iceberg** (pyiceberg) |
| Iceberg catalog — *shared by the app **and** the query engine* | **Postgres** JDBC catalog |
| Distributed SQL engine over the lakehouse | **Trino** |
| Embedded / in-process SQL | **DuckDB** |
| Orchestration (asset lineage + data-quality checks) | **Dagster** |
| Real-time streaming ingestion (CDC) | **Redpanda** (Kafka) |
| Multimodal ingestion — *make every file "speak"* | sentence-transformers + **BLIP** + doc extractors |
| Semantic search + RAG | **LanceDB** + Claude |
| Observability | **Prometheus + Grafana** |
| Knowledge graph + UI | derived graph + cytoscape, FastAPI |

> **The multi-engine core:** pyiceberg (the app) and **Trino** read the *exact same* Iceberg
> tables on MinIO through *one shared Postgres catalog*. The same SQL runs on **DuckDB
> (embedded)** or **Trino (distributed)** from a single toggle — proven live, not a diagram.

**Everything is real and traceable** — no synthetic data, no mocks, no "demo mode." If a
backend or credential is missing it fails loudly. Verified live on **13,000+ real artifacts**
across multiple projects (a 1,200-commit Git repo and an 8,000-file non-git folder with its
real AI sessions).

**One page shows it all** — `/platform` is a live mission-control: every component below with a
real-time up/down probe and a measured proof metric, the medallion pipeline, and the full
capability ledger. Nothing asserted; everything probed.



---

## Architecture — a medallion lakehouse

```
 SOURCES                         INGESTION                         STORAGE
 ───────                         ─────────                         ───────
 Git repo  ─────────►  ingest_git.py  ──┐
 Any folder ────────►  ingest_folder.py ┼─ extract.py (every file → language:        ┌─ MinIO (S3) ─────────┐
 Claude Code logs ──►  ingest_ai.py  ───┘   images→BLIP caption, PDF/DOCX→text,  ───► │ BRONZE  blobs/<sha>  │
 Live file changes ─►  streaming.py (Redpanda/Kafka, real-time CDC)                   │ + warehouse/ (Iceberg)│
                                                                                      └──────────┬───────────┘
                                                                                                 ▼
   ORCHESTRATION (Dagster)                                              ┌──── Apache Iceberg ────┐
   silver_artifacts ─┐                                                  │ SILVER artifacts/edges │  ACID · snapshots
   silver_edges  ────┼─► embeddings (LanceDB)                          │ GOLD  5 medallion marts│  · time-travel
                     ├─► knowledge_graph                                └───────────┬────────────┘  · month-partitioned
                     └─► gold_* marts          + data-quality asset checks          ▼
                                                                       SERVE
   gold: author_activity · file_hotspots · topic_summary ·   ┌─ FastAPI ─────────────────────────────────┐
         daily_activity · cross_silo                          │ /search /ask(RAG) /graph /analytics /sql  │
                                                              │ /snapshots /artifact                      │
   DuckDB SQL ─────────────────────────────────────────────► │ UIs: knowledge graph + DE dashboard       │
                                                              └───────────────────────────────────────────┘
```

**Bronze** = raw bytes in MinIO. **Silver** = the typed, deduped Iceberg `artifacts`/`edges`
tables (the source of truth). **Gold** = five analytical marts, recomputed idempotently and
queried with DuckDB.

### Module map

| Concern | Module |
|---|---|
| Env-driven config (fail-fast) | `config.py` |
| MinIO blob store (content-addressed) | `storage.py` |
| Iceberg tables, scans, gold overwrite, snapshots | `lakehouse.py` |
| Multimodal "make every file speak" | `extract.py` |
| Ingest: git / any folder / Claude logs | `ingest_git.py`, `ingest_folder.py`, `ingest_ai.py` |
| Embeddings → LanceDB | `embed.py` |
| Semantic search / RAG | `search.py`, `ask.py` |
| Derived knowledge graph | `graph.py` |
| Gold medallion marts | `marts.py` |
| Dagster orchestration + checks | `orchestration.py` |
| DuckDB analytics + SQL console | `analytics.py` |
| Real-time streaming (Kafka) | `streaming.py` |
| API + UIs | `api.py`, `web/index.html`, `web/dashboard.html` |
| CLI | `cli.py` |

---

## Quickstart

Prerequisites: **[uv](https://docs.astral.sh/uv/)**. **Docker is optional** — the desktop app
runs a real MinIO itself (see below); Docker is only needed for the browser-driven path or the
"lab" stack.

### The desktop app (zero-Docker, double-click)

The product as a real app: a **native window** (not a browser tab), **no Docker required**, with
**first-run onboarding** and optional **auto-start at login**.

```bash
uv sync --extra desktop              # adds pywebview (uses the Edge WebView2 already on Win11)
uv run tracevault desktop            # opens a native window; first run sets you up
```

On first launch it brings up the whole stack itself and walks you through it:

- **Zero-Docker storage** — it starts a **real local MinIO server** (the official `minio` binary,
  downloaded once and **SHA-256-verified**) as a managed child process. This is genuine S3 object
  storage — the Iceberg warehouse is physically on it, never local files pretending. (MinIO is
  AGPL-3.0; it is fetched/run on your machine, never bundled into this MIT repo.)
- **Onboarding** — an empty install lands on a welcome page: point it at a folder or a git repo
  (a native folder picker in the window), watch the first ingest climb, then land on your graph.
- **Always-on** — the background auto-ingest daemon keeps every source current while it runs.

Make it feel installed (Windows):

```bash
.\install.cmd                        # Desktop + Start-menu shortcuts (double-click "tracevault")
uv run tracevault autostart enable   # also open it automatically when you log in (one Startup shortcut)
```

`autostart` is per-user and trivially reversible (`autostart disable`, or the toggle on the
welcome page, or just delete the shortcut). Everything stays on your laptop.

### Run it in the browser (Docker MinIO)

tracevault is **local-first**: a private, always-on knowledge service on your own machine. The
default stack is light — just **one MinIO container** plus the app; the embedded engines (DuckDB,
LanceDB, a SQLite Iceberg catalog) run in-process. Nothing leaves your laptop.

```bash
docker compose up -d                 # the only thing it needs: MinIO (lakehouse object store)
uv sync                              # Python 3.12 + local models (pulled once)

# tell it what to keep ingested (a git repo or ANY folder); add as many as you want
uv run tracevault add-source --repo   /path/to/a/repo
uv run tracevault add-source --folder /path/to/a/folder

# run the service: it serves the UI AND keeps your knowledge current in the background
uv run tracevault app                # http://localhost:8000  (opens automatically)
```

`tracevault app` re-ingests your sources and your Claude Code logs on a schedule, incrementally
and idempotently — **drop a new file and it shows up in search within seconds, no commands.**
Manual one-shots (`ingest`, `marts`, `serve`) are still there if you want them.

### The "lab" — the full data-engineering stack (optional)

The heavy distributed stack (Postgres catalog + Trino + Redpanda + Prometheus/Grafana) is
**opt-in** so it never weighs down the laptop. Bring it up to explore the multi-engine lakehouse:

```bash
docker compose --profile lab --profile observability up -d   # Postgres, Trino, Grafana, Prometheus
uv run python scripts/migrate_to_postgres.py                  # share the catalog with Trino (no re-ingest)
TRACEVAULT_CATALOG_BACKEND=postgres uv run tracevault serve   # app + Trino on one catalog
uv run dagster dev -m tracevault.orchestration               # http://localhost:3000  pipeline lineage
```

### The surfaces

| URL | What it shows |
|---|---|
| `http://localhost:8000/platform` | **Platform mission-control** — every DE component with a *live* up/down probe + a real proof metric, the medallion pipeline, and the full capability ledger |
| `http://localhost:8000` | **Knowledge graph** — pick a project, explore people ↔ files ↔ topics; every file speaks |
| `http://localhost:8000/dashboard` | **DE dashboard** — medallion stats, charts, cross-silo, hotspots, **DuckDB *and* Trino** SQL, Iceberg time-travel |
| `http://localhost:3000` | **Dagster** — the medallion asset-lineage DAG + data-quality checks |
| `http://localhost:8085` | **Trino** — distributed SQL over the lakehouse (also driven from the dashboard) |
| `http://localhost:3001` | **Grafana** — live platform metrics (Prometheus) |
| `http://localhost:9001` | **MinIO** console — the real warehouse + blobs |

In the lab, the dashboard's SQL console has an engine toggle: run the *same* query on DuckDB
(embedded) or Trino (distributed) — both hit the identical Iceberg tables on MinIO.


### Real-time streaming demo

```bash
# terminal 1 — consume file events into the lakehouse, live
uv run tracevault stream-consume
# terminal 2 — watch a folder and stream new/changed files
uv run tracevault stream-emit --folder /path/to/a/live/folder --repo live
# now create or edit a file in that folder → it lands in Iceberg and is searchable in seconds
```

---

## Investor demo script (5 minutes)

1. **"It's real."** `docker compose up -d` → open MinIO console (`:9001`): the bucket holds the
   Iceberg `warehouse/` (parquet + manifests) and content-addressed `blobs/`. This is a real
   lakehouse on real object storage — not files pretending to be one.
2. **"It ingests anything, not just git."** Show `tracevault ingest --folder …` — the gap:
   in a real org most work never lands in a commit. 8,150 files from a non-git folder + 26 real
   AI sessions, **946 file↔AI cross-links**.
3. **"Every file speaks."** Dashboard → modality doughnut: code, text, **images (captioned by a
   local vision model)**, PDF, DOCX, notebooks — all translated to language and searchable.
4. **"It's orchestrated and governed."** Dagster (`:3000`): the medallion silver→gold asset
   lineage, with **data-quality checks** (referential integrity, embedding coverage).
5. **"It's queryable."** Dashboard → DuckDB SQL console: `SELECT * FROM gold_cross_silo ORDER BY
   shared_topics DESC` → *who across silos has worked on related things* — the "has anyone done
   this?" insight, as a gold table. Show Iceberg **time-travel** snapshots.
6. **"It's real-time."** `stream-emit` on a folder, drop a file in → watch it appear in the
   lakehouse and search within seconds (Redpanda CDC).
7. **"And it's explorable."** Knowledge graph (`:8000`): click a person → their files and AI
   sessions; topic clusters bridging people across projects.

---

## Quality & operations

- **No synthetic data, fail-loud, everything traceable.** Real inputs only.
- **ruff + mypy clean** across all modules; **real-input tests** (no mocks of MinIO/Iceberg/embedder)
  run against a live MinIO container. CI (`.github/workflows/ci.yml`) runs lint + types + tests on
  every push.
- **Idempotent** ingestion (stable content ids); **partitioned** by month; **ACID** appends with
  full snapshot history (time-travel).

```bash
uv run pytest -q ; uv run ruff check tracevault ; uv run mypy tracevault
```

## Performance / scale

Heavy layers are already native under CPython: **Polars (Rust)**, **LanceDB (Rust)**, **PyArrow/
Iceberg (C++)**, **PyTorch (C++/CUDA)** for embeddings + captioning. Folder ingestion runs reads,
extraction, and uploads in a **thread pool** with **batched** image captioning; embeddings are
batched; GPU is used automatically when present. The Python is thin orchestration — a Rust/Julia
rewrite would optimize the already-instant part. If profiling flags a pure-Python hot loop, the
maintainable fix is a small PyO3/maturin or mypyc extension for that loop.

## What this is / isn't

- **Is:** a real, single-operator data platform over data you own; every result traces to a real
  artifact; a real Iceberg lakehouse on real object storage; ingests git, non-git folders, and
  live file streams.
- **Isn't:** multi-tenant. No permissions-aware access control yet — an org deployment would add
  authentication, per-user authorization, and access-scoped search/graph/marts. It runs on one
  machine (the full stack — MinIO, Postgres, Trino, Redpanda, Dagster, Grafana — co-located);
  production would distribute these and add a versioned catalog (Nessie/Polaris) for data
  branching. Everything here is real and verified, not aspirational.

## License

AGPL-3.0 — see [LICENSE](LICENSE). Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

