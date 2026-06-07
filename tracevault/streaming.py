"""Real-time streaming ingestion: watch a folder -> Kafka (Redpanda) -> append to Iceberg.

Change-data-capture style: a watcher emits a file event for every created/modified file; a
consumer reads the topic, translates the file into language (multimodal), appends it to the
silver Iceberg `artifacts` table, and embeds it — so a brand-new file is searchable seconds
after it lands. Real files only; nothing fabricated. Fails loudly if the broker is down.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from .config import Settings
from .extract import extract
from .ids import artifact_id
from .ingest_folder import IGNORE_DIRS, IGNORE_EXTS, MAX_BLOB_BYTES, MAX_INGEST_BYTES
from .lakehouse import Artifact, Lakehouse
from .storage import BlobStore, sha256_hex

logger = logging.getLogger(__name__)


class StreamError(RuntimeError):
    """Raised when the Kafka/Redpanda broker is unreachable or input is invalid."""


def _broker_error(settings: Settings, exc: Exception) -> StreamError:
    return StreamError(
        f"Cannot reach the streaming broker at {settings.kafka_bootstrap!r}. "
        "Is Redpanda running? Try `docker compose up -d`. "
        f"Underlying error: {exc}"
    )


def _walk(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() not in IGNORE_EXTS:
                yield p


def watch_and_emit(folder: str, repo: str, settings: Settings, *, interval: float = 2.0) -> None:
    """Poll a folder and emit a Kafka event for every new/changed file (producer)."""
    root = Path(folder)
    if not root.is_dir():
        raise StreamError(f"{folder!r} is not a directory.")
    try:
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except NoBrokersAvailable as exc:
        raise _broker_error(settings, exc) from exc

    seen: dict[str, float] = {}
    for p in _walk(root):  # prime with current state so only changes-after-start stream
        try:
            seen[str(p)] = p.stat().st_mtime
        except OSError:
            pass
    logger.info(
        "Watching %s -> topic %r (repo=%s). Create or edit files to stream them.",
        root, settings.kafka_topic, repo,
    )
    while True:
        time.sleep(interval)
        for p in _walk(root):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            sp = str(p)
            if seen.get(sp) != m:
                seen[sp] = m
                event = {
                    "repo": repo,
                    "abspath": sp,
                    "rel": p.relative_to(root).as_posix(),
                    "ts": datetime.now(UTC).isoformat(),
                }
                producer.send(settings.kafka_topic, event)
                logger.info("emitted: %s", event["rel"])
        producer.flush()


def consume_and_ingest(settings: Settings, *, enable_images: bool = True) -> None:
    """Consume file events and append each into the lakehouse incrementally (consumer)."""
    from .embed import Embedder  # heavy import, only needed here

    store = BlobStore(settings)
    store.ensure_bucket()
    lake = Lakehouse(settings)
    embedder = Embedder(settings)
    seen = lake.existing_artifact_ids()
    try:
        consumer = KafkaConsumer(
            settings.kafka_topic,
            bootstrap_servers=settings.kafka_bootstrap,
            group_id="tracevault-stream-ingest",
            auto_offset_reset="latest",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
    except NoBrokersAvailable as exc:
        raise _broker_error(settings, exc) from exc

    logger.info("Streaming consumer ready on %r. Waiting for events…", settings.kafka_topic)
    for msg in consumer:
        event = msg.value
        try:
            artifact = _event_to_artifact(event, store, enable_images=enable_images)
        except Exception as exc:  # one bad file shouldn't kill the stream
            logger.warning("skip %s: %s", event.get("rel"), exc)
            continue
        if artifact is None or artifact.id in seen:
            continue
        lake.append_artifacts([artifact])
        seen.add(artifact.id)
        embedder.embed_artifacts(lake)
        logger.info("ingested + embedded (live): %s [%s]", artifact.title, event.get("repo"))


def _event_to_artifact(event: dict[str, Any], store: BlobStore, *, enable_images: bool) -> Artifact | None:
    repo = event["repo"]
    rel = event.get("rel") or Path(event["abspath"]).name
    p = Path(event["abspath"])
    if not p.is_file():
        return None
    size = p.stat().st_size
    if size == 0 or size > MAX_INGEST_BYTES:
        return None
    data = p.read_bytes()
    ex = extract(rel, data, enable_images=enable_images)
    object_uri = store.put_blob(data, filename=rel) if len(data) <= MAX_BLOB_BYTES else None
    return Artifact(
        id=artifact_id("git", "file", f"{repo}:{rel}"),
        kind="file",
        source="git",
        created_at=datetime.now(UTC),  # event time — this is a live append
        repo=repo,
        actor=None,
        title=rel,
        text=f"{rel}\n\n{ex.text.strip()}",
        content_hash=sha256_hex(data),
        object_uri=object_uri,
        extra={"path": rel, "size": size, "modality": ex.modality, "binary": ex.is_binary, "streamed": True},
    )
