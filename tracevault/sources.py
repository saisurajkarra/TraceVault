"""The list of real sources the local knowledge service keeps ingested.

A source is a real Git repo (\"repo\") or any folder (\"folder\") on disk. The list is a small
JSON file under the data dir so it survives restarts and is easy to inspect/edit. The local
service re-ingests every source on a schedule, incrementally and idempotently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Source:
    path: str
    kind: str  # "repo" (git) | "folder" (any directory)

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind}


def load_sources(settings: Settings) -> list[Source]:
    f = settings.sources_file
    if not f.exists():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Could not read sources file %s: %s", f, exc)
        return []
    return [Source(path=s["path"], kind=s.get("kind", "folder")) for s in raw if s.get("path")]


def add_source(settings: Settings, path: str, kind: str) -> list[Source]:
    """Add a source (deduped by resolved path). Returns the full list."""
    settings.ensure_dirs()
    resolved = str(Path(path).resolve())
    sources = load_sources(settings)
    if any(s.path == resolved for s in sources):
        return sources
    sources.append(Source(path=resolved, kind=kind))
    settings.sources_file.write_text(
        json.dumps([s.as_dict() for s in sources], indent=2), encoding="utf-8"
    )
    logger.info("Added %s source: %s", kind, resolved)
    return sources


def remove_source(settings: Settings, path: str) -> list[Source]:
    resolved = str(Path(path).resolve())
    sources = [s for s in load_sources(settings) if s.path != resolved]
    settings.sources_file.write_text(
        json.dumps([s.as_dict() for s in sources], indent=2), encoding="utf-8"
    )
    return sources
