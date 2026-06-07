"""Deterministic, content-derived ids shared across ingestion and the graph.

Ids are stable functions of real source identity so re-ingesting the same artifact
yields the same id (idempotency) and so the graph can reconstruct person ids from
the ``actor`` field without a separate lookup table.
"""

from __future__ import annotations

import hashlib
import re


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:40]


def artifact_id(source: str, kind: str, natural_key: str) -> str:
    """Stable id for an artifact from its source, kind, and a natural key."""
    return f"{kind}:{_h(source, kind, natural_key)}"


_EMAIL_RE = re.compile(r"<([^>]+)>")


def normalize_actor(actor: str) -> str:
    """Canonical key for a person: lowercased email if present, else lowercased name."""
    actor = actor.strip()
    m = _EMAIL_RE.search(actor)
    if m:
        return m.group(1).strip().lower()
    return actor.lower()


def person_id(actor: str) -> str:
    """Stable id for a person derived from their actor string ('Name <email>')."""
    return f"person:{_h('person', normalize_actor(actor))}"
