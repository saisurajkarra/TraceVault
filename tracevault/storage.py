"""Content-addressed raw blob storage on MinIO (S3 protocol).

Raw file bytes from ingested artifacts (e.g. the content of a file at a commit)
are stored here, addressed by the sha256 of their content so identical bytes are
stored once. The object URI returned (``s3://<bucket>/blobs/<sha256>``) is recorded
on the Iceberg artifact row so every UI item can be traced back to its real bytes.

Fails loudly if MinIO is unreachable — there is no local fallback.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from urllib.parse import quote

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, EndpointConnectionError, ParamValidationError

from .config import Settings

logger = logging.getLogger(__name__)

BLOB_PREFIX = "blobs"


class StorageError(RuntimeError):
    """Raised when MinIO is unreachable or an object operation fails."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/with/slashes`` into ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise StorageError(f"Not an s3:// URI: {uri!r}")
    without = uri[len("s3://") :]
    bucket, _, key = without.partition("/")
    if not bucket or not key:
        raise StorageError(f"Malformed s3:// URI (need bucket and key): {uri!r}")
    return bucket, key


@dataclass
class BlobStore:
    """boto3-backed object store on MinIO. One bucket; blobs live under ``blobs/``."""

    settings: Settings

    def __post_init__(self) -> None:
        self._client = boto3.client(
            "s3",
            config=BotoConfig(
                # MinIO requires path-style addressing.
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            **self.settings.boto3_client_kwargs(),
        )

    # --- lifecycle ---

    def ensure_bucket(self) -> None:
        """Create the bucket if missing. Raise a clear error if MinIO is down."""
        bucket = self.settings.bucket
        try:
            existing = {b["Name"] for b in self._client.list_buckets().get("Buckets", [])}
        except EndpointConnectionError as exc:
            raise StorageError(
                f"Cannot reach MinIO at {self.settings.minio_endpoint!r}. "
                "Is it running? Try `docker compose up -d`. "
                f"Underlying error: {exc}"
            ) from exc
        except ClientError as exc:
            raise StorageError(
                f"MinIO rejected the connection (check credentials): {exc}"
            ) from exc

        if bucket not in existing:
            logger.info("Creating MinIO bucket %r", bucket)
            self._client.create_bucket(Bucket=bucket)
        else:
            logger.info("MinIO bucket %r present", bucket)

    # --- blobs ---

    def put_blob(
        self,
        data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Store bytes content-addressed by sha256. Idempotent. Returns the s3:// URI."""
        digest = sha256_hex(data)
        key = f"{BLOB_PREFIX}/{digest}"
        bucket = self.settings.bucket
        if not self._object_exists(bucket, key):
            metadata = {}
            if filename:
                # S3 user-metadata travels in HTTP headers and must be ASCII; non-ASCII
                # file names (common in real folders) are percent-encoded to stay valid.
                metadata["original-filename"] = quote(filename, safe="/._-")
            extra: dict[str, object] = {"Metadata": metadata}
            if content_type:
                extra["ContentType"] = content_type
            try:
                self._client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
            except (ClientError, EndpointConnectionError, ParamValidationError) as exc:
                raise StorageError(f"Failed to write blob {key!r} to MinIO: {exc}") from exc
        return f"s3://{bucket}/{key}"

    def get_blob(self, object_uri: str) -> bytes:
        bucket, key = parse_s3_uri(object_uri)
        try:
            resp = self._client.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except (ClientError, EndpointConnectionError) as exc:
            raise StorageError(f"Failed to read blob {object_uri!r} from MinIO: {exc}") from exc

    def blob_exists(self, object_uri: str) -> bool:
        bucket, key = parse_s3_uri(object_uri)
        return self._object_exists(bucket, key)

    def _object_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise StorageError(f"head_object failed for {key!r}: {exc}") from exc
        except EndpointConnectionError as exc:
            raise StorageError(
                f"Cannot reach MinIO at {self.settings.minio_endpoint!r}: {exc}"
            ) from exc
