"""Ingest a REAL Git repository: commits, the current file tree, and their edges.

Emits ``commit`` and ``file`` artifacts into Iceberg, uploads each current file's
raw bytes to MinIO (content-addressed), and emits edges:
  - ``authored``   person -> commit
  - ``modified``   commit -> file
  - ``co_edited``  file <-> file (files changed together in one commit)

Real repo only. Raises loudly if the path is not a valid Git repository or has no
commits. No synthetic data, no fallback.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from git import Commit, InvalidGitRepositoryError, NoSuchPathError, Repo

from .extract import extract
from .ids import artifact_id, person_id
from .lakehouse import Artifact, Edge, Lakehouse
from .storage import BlobStore, sha256_hex

logger = logging.getLogger(__name__)

# Engineering bounds (documented in README); not data fabrication.
MAX_BLOB_BYTES = 10 * 1024 * 1024  # don't upload individual files larger than 10 MB
MAX_COEDIT_FILES = 25  # skip co_edited fan-out for sprawling commits (O(n^2))


class IngestError(RuntimeError):
    """Raised when the ingest input is not a real, usable Git repository."""


def norm_path(p: str) -> str:
    """Case/sep-normalized absolute path, for matching AI file references to git files."""
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


@dataclass
class GitIngestStats:
    repo: str
    commits: int = 0
    files: int = 0
    edges: int = 0
    blobs_uploaded: int = 0
    skipped_existing: int = 0
    file_paths: list[str] = field(default_factory=list)
    # normalized absolute path -> file artifact id, for ai_touched_file linking.
    file_id_by_abspath: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "commits": self.commits,
            "files": self.files,
            "edges": self.edges,
            "blobs_uploaded": self.blobs_uploaded,
            "skipped_existing": self.skipped_existing,
        }


def _open_repo(repo_path: str) -> Repo:
    try:
        repo = Repo(repo_path, search_parent_directories=False)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise IngestError(
            f"{repo_path!r} is not a valid Git repository. "
            "Point --repo at a real repo (a directory containing .git)."
        ) from exc
    if repo.bare:
        raise IngestError(f"{repo_path!r} is a bare repository; need a working tree.")
    try:
        _ = repo.head.commit
    except Exception as exc:  # unborn HEAD / empty repo
        raise IngestError(f"{repo_path!r} has no commits to ingest: {exc}") from exc
    return repo


def ingest_git(
    repo_path: str,
    lake: Lakehouse,
    store: BlobStore,
    *,
    max_commits: int | None = None,
    enable_images: bool = True,
) -> GitIngestStats:
    """Ingest a real Git repo into the lakehouse + MinIO. Idempotent on artifact id."""
    repo = _open_repo(repo_path)
    repo_name = Path(repo.working_tree_dir or repo_path).resolve().name
    stats = GitIngestStats(repo=repo_name)
    existing = lake.existing_artifact_ids()
    logger.info("Ingesting git repo %r (existing artifacts: %d)", repo_name, len(existing))

    head_commit = repo.head.commit
    head_dt = head_commit.authored_datetime

    # --- 1. current file tree: path -> file artifact id (so edges can resolve) ---
    tree_files: dict[str, Any] = {}
    for raw_item in head_commit.tree.traverse():
        item: Any = raw_item
        if item.type == "blob":
            tree_files[str(item.path)] = item
    file_id_by_path = {
        path: artifact_id("git", "file", f"{repo_name}:{path}") for path in tree_files
    }
    repo_root = str(repo.working_tree_dir or repo_path)
    for path, fid in file_id_by_path.items():
        stats.file_id_by_abspath[norm_path(os.path.join(repo_root, path))] = fid

    # --- 2. iterate commits: commit artifacts, edges, and per-file last-touch time ---
    artifacts: list[Artifact] = []
    edges: list[Edge] = []
    seen_authored: set[tuple[str, str]] = set()
    seen_modified: set[tuple[str, str]] = set()
    coedit_counts: Counter[tuple[str, str]] = Counter()
    coedit_time: dict[tuple[str, str], datetime] = {}
    last_touch: dict[str, datetime] = {}
    # The commit subjects of commits that touched each file — what people "said" about it.
    file_commit_titles: dict[str, list[str]] = {}
    coedit_skipped = 0

    commit_iter = repo.iter_commits(max_count=max_commits) if max_commits else repo.iter_commits()
    for commit in commit_iter:
        commit = _as_commit(commit)
        cid = artifact_id("git", "commit", f"{repo_name}:{commit.hexsha}")
        actor = f"{commit.author.name} <{commit.author.email}>"
        pid = person_id(actor)
        cdt = commit.authored_datetime
        message = (commit.message or "").strip() if isinstance(commit.message, str) else str(commit.message)
        title = message.splitlines()[0][:200] if message else commit.hexsha[:12]

        changed = [str(p) for p in commit.stats.files.keys()]
        for path in changed:
            ts = last_touch.get(path)
            if ts is None or cdt > ts:
                last_touch[path] = cdt

        if cid not in existing:
            artifacts.append(
                Artifact(
                    id=cid,
                    kind="commit",
                    source="git",
                    created_at=cdt,
                    repo=repo_name,
                    actor=actor,
                    title=title,
                    text=message or title,
                    content_hash=commit.hexsha,
                    object_uri=None,
                    extra={
                        "hexsha": commit.hexsha,
                        "branch": _safe_branch(repo),
                        "files_changed": len(changed),
                        "insertions": commit.stats.total.get("insertions"),
                        "deletions": commit.stats.total.get("deletions"),
                    },
                )
            )
            stats.commits += 1

        # authored: person -> commit
        if (pid, cid) not in seen_authored:
            seen_authored.add((pid, cid))
            edges.append(Edge(src_id=pid, dst_id=cid, relation="authored", created_at=cdt))

        # modified: commit -> file (only files that still exist in the current tree)
        present = [p for p in changed if p in file_id_by_path]
        for path in present:
            fid = file_id_by_path[path]
            if (cid, fid) not in seen_modified:
                seen_modified.add((cid, fid))
                edges.append(Edge(src_id=cid, dst_id=fid, relation="modified", created_at=cdt))
            file_commit_titles.setdefault(fid, []).append(title)

        # co_edited: file <-> file changed together (bounded fan-out)
        if len(present) <= MAX_COEDIT_FILES:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = sorted((file_id_by_path[present[i]], file_id_by_path[present[j]]))
                    coedit_counts[(a, b)] += 1
                    if (a, b) not in coedit_time or cdt > coedit_time[(a, b)]:
                        coedit_time[(a, b)] = cdt
        else:
            coedit_skipped += 1

    for (a, b), count in coedit_counts.items():
        edges.append(
            Edge(src_id=a, dst_id=b, relation="co_edited", weight=float(count), created_at=coedit_time[(a, b)])
        )

    # --- 3. file artifacts (current tree); upload real bytes to MinIO ---
    for path, item in tree_files.items():
        fid = file_id_by_path[path]
        if fid in existing:
            stats.skipped_existing += 1
            stats.file_paths.append(path)
            continue
        data = item.data_stream.read()
        content_hash = sha256_hex(data)
        # Translate the file's bytes into language so it can "speak" (multimodal).
        ex = extract(path, data, enable_images=enable_images)
        object_uri: str | None = None
        if len(data) <= MAX_BLOB_BYTES:
            object_uri = store.put_blob(data, filename=path)
            stats.blobs_uploaded += 1
        else:
            logger.warning("Skipping blob upload for oversized file (%d bytes): %s", len(data), path)
        # Compose the searchable text: path + what the file says + what people said about it.
        said = file_commit_titles.get(fid, [])
        text_parts = [path, ex.text.strip()]
        if said:
            unique_said = list(dict.fromkeys(said))[:10]
            text_parts.append("Commits that touched this file:\n- " + "\n- ".join(unique_said))
        artifacts.append(
            Artifact(
                id=fid,
                kind="file",
                source="git",
                created_at=last_touch.get(path, head_dt),
                repo=repo_name,
                actor=None,
                title=path,
                text="\n\n".join(p for p in text_parts if p),
                content_hash=content_hash,
                object_uri=object_uri,
                extra={"path": path, "size": len(data), "modality": ex.modality, "binary": ex.is_binary},
            )
        )
        stats.files += 1
        stats.file_paths.append(path)

    # --- 4. persist ---
    lake.append_artifacts(artifacts)
    lake.append_edges(edges)
    stats.edges = len(edges)
    if coedit_skipped:
        logger.info("Skipped co_edited fan-out for %d large commits (> %d files)", coedit_skipped, MAX_COEDIT_FILES)
    logger.info(
        "Git ingest done: %d commits, %d files, %d edges, %d blobs (repo=%s)",
        stats.commits, stats.files, stats.edges, stats.blobs_uploaded, repo_name,
    )
    return stats


def _as_commit(obj: object) -> Commit:
    assert isinstance(obj, Commit)
    return obj


def _safe_branch(repo: Repo) -> str | None:
    try:
        return repo.active_branch.name
    except Exception:  # detached HEAD
        return None
