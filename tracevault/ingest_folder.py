"""Ingest ANY real folder (not just a Git repo): every file becomes a 'speaking' artifact.

The gap this closes: in a real team, most work is NOT committed to git — local code, files
passed between people, scratch dirs. ingest_folder walks a directory and turns every real
file into a multimodal artifact (images captioned, docs extracted, code/text kept), with no
commits/authors. Files connect to people and topics via the AI sessions that touched them.

Built for scale: file reads, extraction, and blob uploads run in a thread pool (I/O- and
native-bound work releases the GIL), and image captioning is batched on the model. Sensible
defaults skip dependency/build/binary noise. Real files only; nothing is fabricated.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .extract import Extracted, caption_images, extract
from .ids import artifact_id
from .ingest_git import norm_path
from .lakehouse import Artifact, Lakehouse
from .storage import BlobStore, sha256_hex

logger = logging.getLogger(__name__)

MAX_BLOB_BYTES = 10 * 1024 * 1024
MAX_INGEST_BYTES = 6 * 1024 * 1024  # don't ingest files larger than this (skip data dumps)

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build", ".next", ".nuxt",
    "target", "vendor", ".idea", ".vscode", "coverage", ".cache", "out", ".turbo",
    "bin", "obj", ".gradle", ".terraform", "site-packages", ".tox",
}
IGNORE_EXTS = {
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".jar", ".war",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".class", ".pyc", ".pyd",
    ".lock", ".map", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".mp3", ".wav", ".flac", ".ogg",
    ".iso", ".dmg", ".db", ".sqlite", ".parquet",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


class FolderIngestError(RuntimeError):
    """Raised when the folder path is not a usable directory."""


@dataclass
class FolderIngestStats:
    folder: str
    files: int = 0
    blobs_uploaded: int = 0
    skipped_existing: int = 0
    skipped_ignored: int = 0
    file_id_by_abspath: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "folder": self.folder,
            "files": self.files,
            "blobs_uploaded": self.blobs_uploaded,
            "skipped_existing": self.skipped_existing,
            "skipped_ignored": self.skipped_ignored,
        }


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for name in filenames:
            yield Path(dirpath) / name


def ingest_folder(
    folder_path: str,
    lake: Lakehouse,
    store: BlobStore,
    *,
    enable_images: bool = True,
    max_files: int | None = None,
    workers: int | None = None,
    caption_batch: int = 8,
) -> FolderIngestStats:
    """Ingest every real file under a folder as a multimodal 'speaking' artifact (parallel)."""
    root = Path(folder_path)
    if not root.is_dir():
        raise FolderIngestError(f"{folder_path!r} is not a directory. Use --repo for a Git repo.")
    repo_name = root.resolve().name
    stats = FolderIngestStats(folder=repo_name)
    existing = lake.existing_artifact_ids()
    logger.info("Ingesting folder %r (existing artifacts: %d)", repo_name, len(existing))

    # 1. Collect the worklist (cheap sequential walk + filtering).
    worklist: list[tuple[Path, str, str]] = []  # (fpath, rel, fid)
    for fpath in _iter_files(root):
        if fpath.suffix.lower() in IGNORE_EXTS:
            stats.skipped_ignored += 1
            continue
        try:
            size = fpath.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_INGEST_BYTES:
            stats.skipped_ignored += 1
            continue
        rel = fpath.relative_to(root).as_posix()
        fid = artifact_id("git", "file", f"{repo_name}:{rel}")
        stats.file_id_by_abspath[norm_path(str(fpath))] = fid
        if fid in existing:
            stats.skipped_existing += 1
            continue
        worklist.append((fpath, rel, fid))
        if max_files and len(worklist) >= max_files:
            break

    if not worklist:
        logger.info("Folder ingest: no new files to ingest in %r.", repo_name)
        return stats

    n_workers = workers or min(16, (os.cpu_count() or 4) + 4)

    # 2. Read bytes in parallel (I/O-bound; GIL released during reads).
    def _read(item: tuple[Path, str, str]) -> tuple[str, str, bytes] | None:
        fpath, rel, fid = item
        try:
            return rel, fid, fpath.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", fpath, exc)
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        read_items = [r for r in pool.map(_read, worklist) if r is not None]

    data_by_fid = {fid: data for (rel, fid, data) in read_items}
    rel_by_fid = {fid: rel for (rel, fid, data) in read_items}
    fpath_by_fid = {fid: fpath for (fpath, rel, fid) in worklist}

    # 3. Translate to language: batch-caption images; extract everything else in parallel.
    extracted: dict[str, Extracted] = {}
    images = [(rel, fid, data) for (rel, fid, data) in read_items if Path(rel).suffix.lower() in IMAGE_EXTS]
    others = [(rel, fid, data) for (rel, fid, data) in read_items if Path(rel).suffix.lower() not in IMAGE_EXTS]

    if images:
        if enable_images:
            caps = caption_images([d for (_, _, d) in images], [r for (r, _, _) in images], batch_size=caption_batch)
        else:
            caps = ["image"] * len(images)
        for (rel, fid, _data), cap in zip(images, caps, strict=False):
            text = f"[image] {cap}" if enable_images else f"[image file {Path(rel).suffix.lstrip('.')}]"
            extracted[fid] = Extracted(text=text, modality="image", is_binary=True)

    def _extract_other(item: tuple[str, str, bytes]) -> tuple[str, Extracted]:
        rel, fid, data = item
        return fid, extract(rel, data, enable_images=False)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for fid, ex in pool.map(_extract_other, others):
            extracted[fid] = ex

    # 4. Upload blobs in parallel.
    def _upload(fid: str) -> tuple[str, str | None]:
        data = data_by_fid[fid]
        if len(data) <= MAX_BLOB_BYTES:
            return fid, store.put_blob(data, filename=rel_by_fid[fid])
        return fid, None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        uri_by_fid = dict(pool.map(_upload, list(data_by_fid)))
    stats.blobs_uploaded = sum(1 for u in uri_by_fid.values() if u)

    # 5. Build + persist artifacts.
    artifacts: list[Artifact] = []
    for fid, data in data_by_fid.items():
        ex = extracted.get(fid) or Extracted(text="", modality="binary", is_binary=True)
        rel = rel_by_fid[fid]
        mtime = datetime.fromtimestamp(fpath_by_fid[fid].stat().st_mtime, tz=UTC)
        artifacts.append(
            Artifact(
                id=fid,
                kind="file",
                source="git",
                created_at=mtime,
                repo=repo_name,
                actor=None,
                title=rel,
                text=f"{rel}\n\n{ex.text.strip()}",
                content_hash=sha256_hex(data),
                object_uri=uri_by_fid.get(fid),
                extra={"path": rel, "size": len(data), "modality": ex.modality, "binary": ex.is_binary},
            )
        )
    stats.files = len(artifacts)
    lake.append_artifacts(artifacts)
    logger.info(
        "Folder ingest done: %d files, %d blobs, %d skipped (folder=%s)",
        stats.files, stats.blobs_uploaded, stats.skipped_ignored, repo_name,
    )
    return stats
