"""Ingest REAL Claude Code session logs (JSONL transcripts) from disk.

The parser is derived from the real file structure (inspected, not assumed): each
session is a ``.jsonl`` file whose records carry a ``type`` (``user``/``assistant``/
``ai-title``/``attachment``/...), a ``message`` payload, ``cwd``, ``gitBranch``,
``sessionId``, ``timestamp``, and ``uuid``. Assistant ``message.content`` is a list of
blocks (``thinking``/``text``/``tool_use``); user content is a string or block list.

Emits ``ai_session`` + ``ai_message`` artifacts and edges:
  - ``message_of_session``  ai_message -> ai_session
  - ``authored``            person (local user) -> ai_session
  - ``ai_touched_file``     ai_session -> file  (when a tool_use file_path resolves to
                            a real ingested git file)

If no session files are found, the caller logs it and continues — that is NOT a
fallback; the git data remains fully real. Nothing here is fabricated.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .ids import artifact_id, person_id
from .ingest_git import norm_path
from .lakehouse import Artifact, Edge, Lakehouse
from .storage import BlobStore

logger = logging.getLogger(__name__)

MESSAGE_TEXT_CHARS = 4000
SESSION_TEXT_CHARS = 2000
# tool_use input keys that name a file the session touched.
FILE_PATH_KEYS = ("file_path", "notebook_path", "path")
# User "messages" that are system/tool noise, not human prompts — excluded as messages.
_NOISE_PREFIXES = (
    "<task-notification>",
    "<system-reminder>",
    "<command-",
    "<local-command-",
    "Caveat:",
    "[Request interrupted",
)


def default_logs_root() -> Path:
    """The usual on-disk location of Claude Code project transcripts."""
    return Path.home() / ".claude" / "projects"


def locate_session_files(ai_logs_path: str | Path | None) -> list[Path]:
    """Find real session .jsonl transcripts. Skips nested subagent/workflow transcripts."""
    root = Path(ai_logs_path) if ai_logs_path else default_logs_root()
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    if not root.is_dir():
        return []
    found: set[Path] = set()
    # Direct session files, and one level down (so pointing at the projects root works).
    for pattern in ("*.jsonl", "*/*.jsonl"):
        for p in root.glob(pattern):
            parts = {seg.lower() for seg in p.parts}
            if "subagents" in parts or "workflows" in parts:
                continue
            found.add(p)
    return sorted(found)


@dataclass
class AiIngestStats:
    files_scanned: int = 0
    sessions: int = 0
    messages: int = 0
    edges: int = 0
    touched_file_edges: int = 0
    skipped_files: int = 0
    session_titles: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "sessions": self.sessions,
            "messages": self.messages,
            "edges": self.edges,
            "touched_file_edges": self.touched_file_edges,
            "skipped_files": self.skipped_files,
        }


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _block_text(content: object) -> str:
    """Extract human-readable text from a message ``content`` (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _is_noise(text: str) -> bool:
    t = text.lstrip()
    return any(t.startswith(p) for p in _NOISE_PREFIXES)


def _tool_file_paths(content: object) -> list[str]:
    """Absolute/relative file paths referenced by tool_use blocks in a message."""
    out: list[str] = []
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        inp = block.get("input")
        if not isinstance(inp, dict):
            continue
        for key in FILE_PATH_KEYS:
            val = inp.get(key)
            if isinstance(val, str) and val:
                out.append(val)
    return out


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ingest_one_session(
    path: Path,
    lake: Lakehouse,
    store: BlobStore,
    git_file_index: dict[str, str],
    existing: set[str],
    local_actor: str,
    stats: AiIngestStats,
) -> tuple[list[Artifact], list[Edge]]:
    """Parse a single transcript file into artifacts + edges. Pure of side effects on lake."""
    records = list(_iter_records(path))
    if not records:
        return [], []

    session_id = None
    cwd = None
    branch = None
    title = None
    timestamps: list[datetime] = []
    message_records: list[dict] = []
    touched: Counter[str] = Counter()  # file_id -> reference count

    for rec in records:
        if not isinstance(rec, dict):
            continue
        rtype = rec.get("type")
        session_id = session_id or rec.get("sessionId")
        cwd = cwd or rec.get("cwd")
        branch = branch or rec.get("gitBranch")
        if rtype == "ai-title" and not title:
            title = rec.get("aiTitle")
        ts = _parse_ts(rec.get("timestamp"))
        if ts:
            timestamps.append(ts)
        if rtype in ("user", "assistant"):
            message_records.append(rec)

    if not session_id:
        session_id = path.stem
    if not timestamps:
        # A real transcript with no parseable timestamps is unusable for a time-partitioned
        # lakehouse; skip it loudly rather than inventing a time.
        logger.warning("No timestamps in %s; skipping session", path.name)
        stats.skipped_files += 1
        return [], []

    created_at = min(timestamps)
    last_at = max(timestamps)
    session_artifact_id = artifact_id("claude_code", "ai_session", session_id)

    # Build ai_message artifacts + message_of_session edges + collect file references.
    artifacts: list[Artifact] = []
    edges: list[Edge] = []
    first_prompt = ""
    msg_count = 0

    for rec in message_records:
        msg = rec.get("message") or {}
        role = msg.get("role") or rec.get("type")
        content = msg.get("content")
        ts = _parse_ts(rec.get("timestamp")) or created_at
        uuid = rec.get("uuid") or f"{session_id}:{msg_count}"

        # Collect file references from assistant tool_use blocks (and any role, defensively).
        for raw in _tool_file_paths(content):
            resolved = raw if os.path.isabs(raw) else os.path.join(cwd or "", raw)
            fid = git_file_index.get(norm_path(resolved))
            if fid:
                touched[fid] += 1

        text = _block_text(content).strip()
        if rec.get("type") == "user":
            # Skip tool-result records and system/tool noise; keep genuine human prompts.
            if rec.get("toolUseResult") is not None or not text or _is_noise(text):
                continue
            if not first_prompt:
                first_prompt = text[:SESSION_TEXT_CHARS]
        elif not text:
            continue  # assistant message with only thinking/tool_use, no visible text

        mid = artifact_id("claude_code", "ai_message", f"{session_id}:{uuid}")
        if mid in existing:
            continue
        artifacts.append(
            Artifact(
                id=mid,
                kind="ai_message",
                source="claude_code",
                created_at=ts,
                repo=None,
                actor=("assistant" if role == "assistant" else local_actor),
                title=text[:120],
                text=text[:MESSAGE_TEXT_CHARS],
                content_hash=None,
                object_uri=None,
                extra={"session_id": session_id, "role": role, "uuid": uuid},
            )
        )
        edges.append(
            Edge(src_id=mid, dst_id=session_artifact_id, relation="message_of_session", created_at=ts)
        )
        msg_count += 1

    # ai_session artifact (store raw transcript bytes in MinIO for full traceability).
    if session_artifact_id not in existing:
        raw_bytes = path.read_bytes()
        object_uri = store.put_blob(raw_bytes, filename=path.name, content_type="application/x-ndjson")
        session_title = title or (first_prompt[:120] if first_prompt else f"session {session_id[:8]}")
        session_text_parts = [session_title]
        if first_prompt:
            session_text_parts.append(first_prompt)
        artifacts.append(
            Artifact(
                id=session_artifact_id,
                kind="ai_session",
                source="claude_code",
                created_at=created_at,
                repo=Path(cwd).name if cwd else None,
                actor=local_actor,
                title=session_title[:200],
                text="\n\n".join(session_text_parts)[:MESSAGE_TEXT_CHARS],
                content_hash=None,
                object_uri=object_uri,
                extra={
                    "session_id": session_id,
                    "cwd": cwd,
                    "branch": branch,
                    "messages": msg_count,
                    "ended_at": last_at.isoformat(),
                    "source_file": str(path),
                },
            )
        )
        # person (local user) authored this session.
        edges.append(
            Edge(
                src_id=person_id(local_actor),
                dst_id=session_artifact_id,
                relation="authored",
                created_at=created_at,
            )
        )
        stats.sessions += 1
        stats.session_titles.append(session_title)

    # ai_touched_file edges: session -> real git file (weight = #references).
    for fid, count in touched.items():
        edges.append(
            Edge(
                src_id=session_artifact_id,
                dst_id=fid,
                relation="ai_touched_file",
                weight=float(count),
                created_at=created_at,
            )
        )
        stats.touched_file_edges += 1

    stats.messages += msg_count
    return artifacts, edges


def ingest_ai(
    ai_logs_path: str | Path | None,
    lake: Lakehouse,
    store: BlobStore,
    *,
    git_file_index: dict[str, str] | None = None,
    max_sessions: int | None = None,
) -> AiIngestStats:
    """Ingest real Claude Code transcripts. Returns stats; never fabricates content."""
    git_file_index = git_file_index or {}
    files = locate_session_files(ai_logs_path)
    stats = AiIngestStats()
    if not files:
        logger.info(
            "No Claude Code session logs found at %s — skipping AI ingest (git data is still real).",
            ai_logs_path or default_logs_root(),
        )
        return stats
    if max_sessions:
        files = files[:max_sessions]

    local_actor = f"{getpass.getuser()} (local)"
    existing = lake.existing_artifact_ids()
    all_artifacts: list[Artifact] = []
    all_edges: list[Edge] = []

    for path in files:
        stats.files_scanned += 1
        try:
            arts, edges = _ingest_one_session(
                path, lake, store, git_file_index, existing, local_actor, stats
            )
        except Exception as exc:  # one corrupt transcript shouldn't abort the run
            logger.warning("Failed to parse %s: %s", path.name, exc)
            stats.skipped_files += 1
            continue
        all_artifacts.extend(arts)
        all_edges.extend(edges)

    lake.append_artifacts(all_artifacts)
    lake.append_edges(all_edges)
    stats.edges = len(all_edges)
    logger.info(
        "AI ingest done: %d sessions, %d messages, %d edges (%d file links) from %d files",
        stats.sessions, stats.messages, stats.edges, stats.touched_file_edges, stats.files_scanned,
    )
    return stats
