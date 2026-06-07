"""FastAPI service: ingest, search, graph, ask, artifact resolution, and the UI.

Loads the real backends once at startup (MinIO bucket, Iceberg tables, the embedding
model). It fails loudly if MinIO is unreachable or the model cannot load — there is no
degraded mode. Every endpoint returns data derived from real ingested artifacts.
"""

from __future__ import annotations

import logging
import mimetypes
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel

from . import analytics
from .ask import AskError, ask
from .config import Settings, get_settings
from .embed import Embedder
from .graph import build_graph
from .ingest_ai import ingest_ai
from .ingest_git import ingest_git
from .lakehouse import Lakehouse
from .search import Searcher
from .sources import add_source, load_sources
from .storage import BlobStore, StorageError

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Prometheus metrics (scraped at /metrics).
ARTIFACTS = Gauge("tracevault_artifacts", "Artifact / edge counts in the lakehouse", ["kind"])
SEARCH_REQUESTS = Counter("tracevault_search_requests_total", "Semantic search requests")
SQL_QUERIES = Counter("tracevault_sql_queries_total", "SQL queries run", ["engine"])
INGESTS = Counter("tracevault_ingests_total", "Ingest runs")


# First-run onboarding: state of the very first ingest, shared with a background worker.
_ONBOARD_LOCK = threading.Lock()
_ONBOARD: dict[str, Any] = {"running": False, "done": False, "error": None, "summary": None}


def _needs_onboarding(state: Any) -> bool:
    """True on first run: nothing configured and nothing ingested yet."""
    if load_sources(state.settings):
        return False
    try:
        return int(state.lake.counts().get("artifacts_total", 0)) == 0
    except Exception:
        return False


class OnboardSourceReq(BaseModel):
    path: str
    kind: str = "folder"  # "folder" (any directory) | "repo" (a git repo)


class AutostartReq(BaseModel):
    enabled: bool


class IngestRequest(BaseModel):
    repo_path: str
    ai_logs_path: str | None = None
    max_commits: int | None = None
    max_sessions: int | None = None
    enable_images: bool = True


class AskRequest(BaseModel):
    q: str
    k: int = 6
    repo: str | None = None


class SqlRequest(BaseModel):
    query: str
    limit: int = 500
    engine: str = "duckdb"  # "duckdb" (embedded) | "trino" (distributed)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    logger.info("Starting tracevault API (endpoint=%s bucket=%s)", settings.minio_endpoint, settings.bucket)
    store = BlobStore(settings)
    store.ensure_bucket()  # fails loudly if MinIO is down
    lake = Lakehouse(settings)
    embedder = Embedder(settings)  # fails loudly if the model cannot load
    app.state.settings = settings
    app.state.store = store
    app.state.lake = lake
    app.state.embedder = embedder
    app.state.searcher = Searcher(settings, embedder)
    # One shared lock serializes every ingest/embed writer (the auto-ingest daemon, first-run
    # onboarding, the /ingest endpoint) so they can never double-append to the LanceDB index.
    app.state.ingest_lock = threading.Lock()
    # Local "knowledge service": keep the lakehouse current in the background.
    service = None
    if settings.auto_ingest:
        from .auto_ingest import AutoIngestService

        service = AutoIngestService(app.state, settings.auto_ingest_interval)
        service.start()
        logger.info("Auto-ingest service started (every %ds).", settings.auto_ingest_interval)
    app.state.auto_ingest = service
    yield
    if service is not None:
        service.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="tracevault", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    def index() -> Response:
        # First run (no sources, nothing ingested) -> send the user to onboarding instead
        # of an empty graph. We never show placeholder/fake data while the lakehouse is empty.
        if _needs_onboarding(app.state):
            return RedirectResponse(url="/welcome")
        idx = WEB_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(status_code=500, detail=f"UI not found at {idx}")
        return FileResponse(idx)

    @app.get("/welcome")
    def welcome_page() -> FileResponse:
        page = WEB_DIR / "welcome.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail=f"Welcome page not found at {page}")
        return FileResponse(page)

    @app.get("/api/onboarding")
    def onboarding_info() -> dict:
        state = app.state
        return {
            "needs_onboarding": _needs_onboarding(state),
            "sources": [s.as_dict() for s in load_sources(state.settings)],
            "counts": state.lake.counts(),
        }

    @app.post("/onboard/add-source")
    def onboard_add_source(req: OnboardSourceReq) -> dict:
        kind = req.kind if req.kind in ("folder", "repo") else "folder"
        path = Path(req.path.strip()).expanduser()
        if not path.exists() or not path.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a folder on this machine: {req.path!r}")
        if kind == "repo" and not (path / ".git").exists():
            raise HTTPException(
                status_code=400,
                detail=f"{req.path!r} is not a git repo (no .git). Add it as a folder instead.",
            )
        add_source(app.state.settings, str(path), kind)
        return {"sources": [s.as_dict() for s in load_sources(app.state.settings)]}

    @app.post("/onboard/ingest")
    def onboard_ingest() -> dict:
        from .auto_ingest import sync_once

        state = app.state
        with _ONBOARD_LOCK:
            if _ONBOARD["running"]:
                return {"running": True, "done": False}
            _ONBOARD.update(running=True, done=False, error=None, summary=None)

        def _work() -> None:
            from .marts import build_marts

            try:
                # Serialize with the always-on daemon so they never double-embed into LanceDB.
                with state.ingest_lock:
                    out = sync_once(state.lake, state.store, state.embedder, state.settings)
                    if out.get("embedded", 0) > 0:
                        try:
                            build_marts(state.lake)
                        except Exception as exc:  # marts are derived; don't fail the whole ingest
                            logger.warning("onboarding mart build failed: %s", exc)
                        state.searcher = Searcher(state.settings, state.embedder)
                # Every configured source failed and nothing was ingested -> fail LOUDLY rather
                # than drop the user onto an empty knowledge base with a green "Done".
                attempted = out.get("sources_attempted", 0)
                if attempted and out.get("sources_failed", 0) >= attempted and out.get("embedded", 0) == 0:
                    with _ONBOARD_LOCK:
                        _ONBOARD["error"] = "Ingest failed for every source:\n" + "\n".join(
                            out.get("errors", [])
                        )
                else:
                    with _ONBOARD_LOCK:
                        _ONBOARD["summary"] = out
            except Exception as exc:
                logger.exception("onboarding ingest failed")
                with _ONBOARD_LOCK:
                    _ONBOARD["error"] = str(exc)
            finally:
                with _ONBOARD_LOCK:
                    _ONBOARD.update(running=False, done=True)

        threading.Thread(target=_work, name="tracevault-onboard-ingest", daemon=True).start()
        return {"running": True, "done": False}

    @app.get("/onboard/status")
    def onboard_status() -> dict:
        with _ONBOARD_LOCK:
            st = dict(_ONBOARD)
        st["counts"] = app.state.lake.counts()
        st["sources"] = [s.as_dict() for s in load_sources(app.state.settings)]
        return st

    @app.get("/autostart")
    def autostart_get() -> dict:
        from . import autostart

        try:
            return {"supported": True, "enabled": autostart.is_enabled(), "path": str(autostart.shortcut_path())}
        except Exception:
            return {"supported": False, "enabled": False, "path": None}

    @app.post("/autostart")
    def autostart_set(req: AutostartReq) -> dict:
        from . import autostart

        try:
            if req.enabled:
                autostart.enable()
            else:
                autostart.disable()
            return {"supported": True, "enabled": autostart.is_enabled()}
        except autostart.AutostartError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/dashboard")
    def dashboard() -> FileResponse:
        page = WEB_DIR / "dashboard.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail=f"Dashboard not found at {page}")
        return FileResponse(page)

    @app.get("/platform")
    def platform_page() -> FileResponse:
        page = WEB_DIR / "platform.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail=f"Platform page not found at {page}")
        return FileResponse(page)

    @app.get("/platform-status")
    def platform_status() -> dict:
        from . import platform as platform_mod

        return platform_mod.platform_status(app.state.settings, app.state.lake, app.state.embedder)

    @app.get("/analytics")
    def analytics_summary() -> dict:
        return analytics.summary(app.state.lake)

    @app.post("/sql")
    def sql(req: SqlRequest) -> dict:
        SQL_QUERIES.labels(engine=req.engine).inc()
        try:
            if req.engine == "trino":
                return analytics.run_trino(app.state.settings, req.query, limit=req.limit)
            return analytics.run_sql(app.state.lake, req.query, limit=req.limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/snapshots")
    def snapshots(table: str = Query("artifacts")) -> dict:
        return {"table": table, "snapshots": app.state.lake.table_snapshots(table)}

    @app.get("/metrics")
    def metrics() -> Response:
        # Refresh lakehouse gauges on scrape (cheap: scans only the kind column).
        for k, v in app.state.lake.counts().items():
            ARTIFACTS.labels(kind=k).set(v)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict:
        s: Settings = app.state.settings
        return {"status": "ok", "model": s.embedding_model, "ask_enabled": bool(s.anthropic_api_key)}

    @app.get("/stats")
    def stats(repo: str | None = Query(None)) -> dict:
        return app.state.lake.counts(repo=repo)

    @app.get("/repos")
    def repos() -> dict:
        return {"repos": app.state.lake.repos()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        lake: Lakehouse = app.state.lake
        store: BlobStore = app.state.store
        embedder: Embedder = app.state.embedder
        try:
            g = ingest_git(
                req.repo_path, lake, store, max_commits=req.max_commits, enable_images=req.enable_images
            )
            a = ingest_ai(
                req.ai_logs_path,
                lake,
                store,
                git_file_index=g.file_id_by_abspath,
                max_sessions=req.max_sessions,
            )
        except (StorageError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Serialize the embed with the always-on daemon so they can't double-append to LanceDB.
        with app.state.ingest_lock:
            embedded = embedder.embed_artifacts(lake)
        INGESTS.inc()
        # Reopen the vector table so search sees the freshly embedded rows.
        app.state.searcher = Searcher(app.state.settings, embedder)
        return {"git": g.as_dict(), "ai": a.as_dict(), "embedded": embedded, "counts": lake.counts()}

    @app.get("/search")
    def search(
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=50),
        kind: str | None = Query(None),
        repo: str | None = Query(None),
    ) -> dict:
        SEARCH_REQUESTS.inc()
        hits = app.state.searcher.search(q, k=k, kind=kind, repo=repo)
        return {"query": q, "count": len(hits), "results": [h.as_dict() for h in hits]}

    @app.get("/graph")
    def graph(
        repo: str | None = Query(None),
        focus: str | None = Query(None),
        kind: str | None = Query(None),
        max_nodes: int = Query(220, ge=10, le=2000),
    ) -> dict:
        return build_graph(app.state.lake, repo=repo, focus=focus, kind=kind, max_nodes=max_nodes)

    @app.post("/ask")
    def ask_endpoint(req: AskRequest) -> dict:
        try:
            result = ask(req.q, app.state.searcher, app.state.settings, k=req.k, repo=req.repo)
        except AskError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result.as_dict()

    @app.get("/artifact/{artifact_id}")
    def artifact(artifact_id: str) -> JSONResponse:
        row = app.state.lake.get_artifact(artifact_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"No artifact with id {artifact_id!r}")
        created = row.get("created_at")
        out = {
            "id": row["id"],
            "kind": row["kind"],
            "source": row["source"],
            "repo": row.get("repo"),
            "actor": row.get("actor"),
            "created_at": created.isoformat() if created is not None else None,
            "title": row.get("title"),
            "text": row.get("text"),
            "content_hash": row.get("content_hash"),
            "object_uri": row.get("object_uri"),
            "extra": row.get("extra"),
            "has_blob": bool(row.get("object_uri")),
            "raw_url": f"/artifact/{artifact_id}/raw" if row.get("object_uri") else None,
        }
        return JSONResponse(out)

    @app.get("/artifact/{artifact_id}/raw")
    def artifact_raw(artifact_id: str) -> Response:
        row = app.state.lake.get_artifact(artifact_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"No artifact with id {artifact_id!r}")
        uri = row.get("object_uri")
        if not uri:
            raise HTTPException(status_code=404, detail="This artifact has no raw blob (it is an event).")
        try:
            data = app.state.store.get_blob(uri)
        except StorageError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        title = row.get("title") or artifact_id
        content_type = mimetypes.guess_type(title)[0] or "application/octet-stream"
        # Render text-like content inline in the browser.
        if content_type.startswith("text/") or content_type in (
            "application/json",
            "application/x-ndjson",
        ):
            return Response(content=data, media_type=f"{content_type}; charset=utf-8")
        return Response(content=data, media_type=content_type)

    return app


app = create_app()
