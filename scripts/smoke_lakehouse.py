"""Smoke test: prove the lakehouse is real (Iceberg data physically in MinIO).

Run: uv run python scripts/smoke_lakehouse.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig

from tracevault.config import get_settings
from tracevault.lakehouse import Artifact, Edge, Lakehouse
from tracevault.storage import BlobStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    settings = get_settings()
    print(f"endpoint={settings.minio_endpoint} bucket={settings.bucket} warehouse={settings.warehouse}")

    # 1. MinIO reachable + bucket.
    store = BlobStore(settings)
    store.ensure_bucket()
    uri = store.put_blob(b"hello tracevault smoke", filename="smoke.txt")
    assert store.get_blob(uri) == b"hello tracevault smoke"
    print(f"blob round-trip OK: {uri}")

    # 2. Iceberg tables created on MinIO; append + read back.
    lake = Lakehouse(settings)
    now = datetime.now(timezone.utc)
    lake.append_artifacts(
        [
            Artifact(
                id="smoke-commit-1",
                kind="commit",
                source="git",
                created_at=now,
                actor="smoke <smoke@example.com>",
                title="smoke commit",
                text="this is a smoke test commit message",
                content_hash="deadbeef",
            ),
            Artifact(
                id="smoke-file-1",
                kind="file",
                source="git",
                created_at=now,
                title="smoke.txt",
                text="smoke.txt :: hello tracevault",
                object_uri=uri,
            ),
        ]
    )
    lake.append_edges(
        [Edge(src_id="smoke-commit-1", dst_id="smoke-file-1", relation="modified", created_at=now)]
    )

    arts = lake.scan_artifacts()
    edges = lake.scan_edges()
    print(f"artifacts scanned: {arts.height}  edges scanned: {edges.height}")
    print(f"counts: {lake.counts()}")
    one = lake.get_artifact("smoke-commit-1")
    assert one is not None and one["title"] == "smoke commit"
    print(f"get_artifact OK: {one['id']} -> {one['title']!r}")

    # 3. Prove the Iceberg metadata/data physically landed in MinIO under warehouse/.
    s3 = boto3.client(
        "s3",
        config=BotoConfig(s3={"addressing_style": "path"}),
        **settings.boto3_client_kwargs(),
    )
    resp = s3.list_objects_v2(Bucket=settings.bucket, Prefix="warehouse/")
    keys = [o["Key"] for o in resp.get("Contents", [])]
    print(f"objects in MinIO under warehouse/: {len(keys)}")
    for k in keys[:12]:
        print(f"  {k}")
    assert any(k.endswith(".parquet") for k in keys), "no parquet data files in MinIO!"
    assert any(".metadata.json" in k for k in keys), "no Iceberg metadata in MinIO!"
    print("\nSMOKE TEST PASSED: Iceberg warehouse is physically on MinIO.")


if __name__ == "__main__":
    main()
