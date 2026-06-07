"""Always-on background ingestion: keep the lakehouse current from all configured sources.

This is what makes tracevault a local *service* rather than a one-shot tool: a daemon thread
re-ingests every configured source (a git repo or any folder) plus your Claude Code logs on a
schedule, incrementally and idempotently (unchanged files are skipped). When something new
lands it re-embeds, rebuilds the gold marts, and refreshes the searcher --- so your knowledge
stays current with no manual commands. Single process, shared (thread-safe) lakehouse.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .config import Settings
from .embed import Embedder
from .ingest_ai import ingest_ai
from .ingest_folder import ingest_folder
from .ingest_git import ingest_git
from .lakehouse import Lakehouse
from .sources import load_sources
from .storage import BlobStore

logger = logging.getLogger(__name__)


def sync_once(
    lake: Lakehouse,
    store: BlobStore,
    embedder: Embedder,
    settings: Settings,
    *,
    enable_images: bool = True,
) -> dict[str, Any]:
    """Ingest every configured source + Claude logs once. Returns a small summary.

    Per-source failures are collected (``errors`` / ``sources_failed``) instead of raising, so
    one bad source never stops the rest — but a caller (e.g. first-run onboarding) can still
    detect a *total* failure and surface it loudly rather than report a false success.
    """
    file_index: dict[str, str] = {}
    out: dict[str, Any] = {
        "repos": 0, "folders": 0, "ai_sessions": 0, "embedded": 0,
        "sources_attempted": 0, "sources_failed": 0, "errors": [],
    }
    for src in load_sources(settings):
        out["sources_attempted"] += 1
        try:
            if src.kind == "repo":
                g = ingest_git(src.path, lake, store, enable_images=enable_images)
                file_index.update(g.file_id_by_abspath)
                out["repos"] += 1
            else:
                f = ingest_folder(src.path, lake, store, enable_images=enable_images)
                file_index.update(f.file_id_by_abspath)
                out["folders"] += 1
        except Exception as exc:  # one bad source shouldn't stop the rest
            out["sources_failed"] += 1
            out["errors"].append(f"{src.path}: {exc}")
            logger.warning("auto-ingest source %s failed: %s", src.path, exc)
    try:
        a = ingest_ai(settings.claude_logs_path, lake, store, git_file_index=file_index)
        out["ai_sessions"] = a.sessions
    except Exception as exc:
        logger.warning("auto-ingest of Claude logs failed: %s", exc)
    out["embedded"] = embedder.embed_artifacts(lake)
    return out


class AutoIngestService:
    """Runs sync_once on a schedule in a daemon thread, sharing the app's lakehouse."""

    def __init__(self, app_state: Any, interval: int) -> None:
        self.app_state = app_state
        self.interval = max(15, interval)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tracevault-auto-ingest", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from .marts import build_marts
        from .search import Searcher

        # One shared lock serializes EVERY embedder/writer (this daemon + first-run onboarding +
        # the /ingest endpoint) so two threads can never read the same not-yet-embedded set and
        # both append it to LanceDB (which would duplicate vectors). Created here if absent.
        lock = getattr(self.app_state, "ingest_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.app_state.ingest_lock = lock

        while not self._stop.is_set():
            try:
                settings = self.app_state.settings
                with lock:
                    out = sync_once(
                        self.app_state.lake, self.app_state.store, self.app_state.embedder, settings
                    )
                    if out["embedded"] > 0:
                        # Something new landed: refresh marts + the search index.
                        try:
                            build_marts(self.app_state.lake)
                        except Exception as exc:
                            logger.warning("auto mart rebuild failed: %s", exc)
                        self.app_state.searcher = Searcher(settings, self.app_state.embedder)
                        logger.info("auto-ingest: %s (knowledge refreshed)", out)
            except Exception as exc:  # never let the daemon die
                logger.warning("auto-ingest cycle failed: %s", exc)
            self._stop.wait(self.interval)
