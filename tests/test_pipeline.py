"""End-to-end tests against REAL inputs — no mocks of MinIO, Iceberg, or the embedder.

Fixture repo: this project's own Git history (real commits, real authors).
Backing store: the REAL MinIO container from docker-compose, isolated to a throwaway
bucket per run. If MinIO is not reachable the tests skip with an actionable message
(that is a test prerequisite, not a product fallback).

Asserts the spec's guarantees: artifacts land in Iceberg + blobs in MinIO, commit
counts match the real `git log`, semantic search returns the right real artifact, and
the graph contains the real commit authors as person nodes.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import boto3
import pytest
from botocore.config import Config as BotoConfig

from tracevault.config import Settings
from tracevault.embed import Embedder
from tracevault.graph import GraphBuilder
from tracevault.ingest_git import ingest_git
from tracevault.lakehouse import Lakehouse
from tracevault.search import Searcher
from tracevault.storage import BlobStore, StorageError

REPO = Path(__file__).resolve().parent.parent  # this project's repo root


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Real ingest + embed of this repo into an isolated MinIO bucket. Cleans up after."""
    bucket = f"tracevault-test-{uuid.uuid4().hex[:8]}"
    data_dir = tmp_path_factory.mktemp("tvdata")
    settings = Settings(bucket=bucket, data_dir=data_dir)

    store = BlobStore(settings)
    try:
        store.ensure_bucket()
    except StorageError as exc:
        pytest.skip(f"Real MinIO not reachable ({exc}). Start it with `docker compose up -d`.")

    lake = Lakehouse(settings)
    stats = ingest_git(str(REPO), lake, store)
    Embedder(settings).embed_artifacts(lake)
    searcher = Searcher(settings, Embedder(settings))

    yield {"settings": settings, "store": store, "lake": lake, "stats": stats, "searcher": searcher}

    # Teardown: empty and delete the throwaway bucket.
    s3 = boto3.client(
        "s3", config=BotoConfig(s3={"addressing_style": "path"}), **settings.boto3_client_kwargs()
    )
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
    s3.delete_bucket(Bucket=bucket)


def test_commit_count_matches_real_git_log(pipeline):
    expected = int(_git("rev-list", "--count", "HEAD"))
    counts = pipeline["lake"].counts()
    assert pipeline["stats"].commits == expected
    assert counts["artifacts_commit"] == expected


def test_files_and_commits_land_in_iceberg(pipeline):
    counts = pipeline["lake"].counts()
    tracked = len(_git("ls-files").splitlines())
    # Every current tracked file becomes a file artifact (binary + text alike).
    assert counts["artifacts_file"] == tracked
    assert counts["artifacts_total"] == counts["artifacts_file"] + counts["artifacts_commit"]
    assert counts["edges_total"] > 0


def test_blobs_physically_in_minio(pipeline):
    lake, store, settings = pipeline["lake"], pipeline["store"], pipeline["settings"]
    files = lake.scan_artifacts(kind="file")
    uris = [u for u in files["object_uri"].to_list() if u]
    assert uris, "expected at least one file blob"
    assert store.blob_exists(uris[0])

    # The Iceberg warehouse itself is physically in MinIO (parquet + metadata).
    s3 = boto3.client(
        "s3", config=BotoConfig(s3={"addressing_style": "path"}), **settings.boto3_client_kwargs()
    )
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=settings.bucket, Prefix="warehouse/").get("Contents", [])]
    assert any(k.endswith(".parquet") for k in keys)
    assert any(".metadata.json" in k for k in keys)


def test_semantic_search_finds_real_artifact(pipeline):
    # A concept that genuinely exists in this repo's source.
    hits = pipeline["searcher"].search("content-addressed blob storage on MinIO", k=8)
    assert hits, "search returned no results"
    titles = [h.title for h in hits]
    assert any("storage.py" in (t or "") for t in titles), titles
    # Every hit traces back to a real artifact id present in Iceberg.
    assert pipeline["lake"].get_artifact(hits[0].id) is not None


def test_graph_contains_real_commit_authors(pipeline):
    authors = set(_git("log", "--format=%an").splitlines())
    gb = GraphBuilder(pipeline["lake"])
    person_labels = {n["data"]["label"] for n in gb._full.nodes if n["data"]["kind"] == "person"}
    assert person_labels & authors, f"no real author among person nodes: {person_labels} vs {authors}"
    # Person nodes reference their real underlying commit artifacts.
    person_nodes = [n for n in gb._full.nodes if n["data"]["kind"] == "person"]
    assert any(n["data"]["artifact_count"] > 0 for n in person_nodes)
