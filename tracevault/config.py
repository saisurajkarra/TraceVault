"""Env-driven settings with fail-fast validation.

Every configurable value lives here so it can be injected, not scattered as
constants. Settings are read from the environment (prefix ``TRACEVAULT_``) and an
optional ``.env`` file. The Anthropic key is read from the conventional unprefixed
``ANTHROPIC_API_KEY`` and is optional — without it, ``/ask`` degrades to returning
real retrieved context (it never fabricates an answer).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid. Always actionable."""


class Settings(BaseSettings):
    """All tracevault configuration. Injectable; never read os.environ elsewhere."""

    model_config = SettingsConfigDict(
        env_prefix="TRACEVAULT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- MinIO / S3 ---
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_region: str = "us-east-1"
    # One bucket holds BOTH raw blobs (blobs/) and the Iceberg warehouse (warehouse/).
    bucket: str = "tracevault"

    # --- Zero-Docker mode: the app manages a real local MinIO server process itself,
    #     so a non-technical user needs no Docker. It is still REAL object storage
    #     (the official minio binary over S3), never local files pretending to be one.
    #     `tracevault desktop` turns this on; `serve`/`app` leave it off (use Docker MinIO). ---
    manage_minio: bool = False
    minio_binary: str | None = None  # explicit path to the minio executable (else PATH/cache/download)
    minio_auto_download: bool = True  # fetch the official minio binary if absent (sha256-verified)
    # Pin a specific MinIO release (e.g. "RELEASE.2025-04-22T22-12-26Z") for reproducible
    # downloads; None pulls the current stable build. Either way the SHA-256 is verified.
    minio_release: str | None = None

    # --- Local state. The Iceberg DATA lives in MinIO; only the catalog pointer
    #     (SQLite file, or Postgres) and the LanceDB index live on local disk. ---
    data_dir: Path = Path("./data")

    # --- Iceberg catalog backend: "sql" (local SQLite, default) or "postgres"
    #     (a JDBC catalog shared with Trino for distributed SQL). ---
    catalog_backend: str = "sql"
    postgres_dsn: str = "postgresql+psycopg2://iceberg:iceberg@localhost:5433/iceberg"

    # --- Embeddings (real, local, offline) ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- API ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Streaming (Redpanda / Kafka) ---
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic: str = "tracevault.events"

    # --- Trino (distributed SQL over the shared Iceberg catalog) ---
    trino_host: str = "localhost"
    trino_port: int = 8085
    trino_catalog: str = "iceberg"

    # --- Local "knowledge service": always-on background ingestion ---
    auto_ingest: bool = False  # serve turns this off; `tracevault app` turns it on
    auto_ingest_interval: int = 120  # seconds between background re-syncs
    claude_logs_path: str | None = None  # default: ~/.claude/projects (ingest_ai default)

    # --- Optional grounded NL answers ---
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    # Current Claude model id (verified against the claude-api reference, 2026-06).
    anthropic_model: str = "claude-opus-4-8"

    @field_validator("minio_endpoint")
    @classmethod
    def _endpoint_must_be_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"TRACEVAULT_MINIO_ENDPOINT must start with http:// or https:// (got {v!r})"
            )
        return v.rstrip("/")

    @field_validator("minio_access_key", "minio_secret_key", "bucket")
    @classmethod
    def _must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("MinIO credentials and bucket name must be non-empty")
        return v

    # --- Derived paths / connection properties ---

    @property
    def catalog_db_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def catalog_uri(self) -> str:
        # Postgres JDBC catalog (shared with Trino) or local SQLite. The SQLite URI on
        # Windows yields sqlite:///C:/.../catalog.db
        if self.catalog_backend == "postgres":
            return self.postgres_dsn
        return f"sqlite:///{self.catalog_db_path.resolve().as_posix()}"

    @property
    def warehouse(self) -> str:
        return f"s3://{self.bucket}/warehouse"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def minio_data_dir(self) -> Path:
        """Local data directory for a managed (zero-Docker) MinIO server."""
        return self.data_dir / "minio-data"

    @property
    def minio_bin_path(self) -> Path:
        """Where an auto-downloaded MinIO binary is cached."""
        name = "minio.exe" if sys.platform == "win32" else "minio"
        return self.data_dir / "bin" / name

    @property
    def minio_host_port(self) -> tuple[str, int]:
        """(host, port) parsed from minio_endpoint — used to bind / health-check a local server."""
        parsed = urlparse(self.minio_endpoint)
        default_port = 443 if parsed.scheme == "https" else 9000
        return (parsed.hostname or "127.0.0.1", parsed.port or default_port)

    @property
    def sources_file(self) -> Path:
        """Persisted list of folders/repos the local service auto-ingests."""
        return self.data_dir / "sources.json"

    def iceberg_catalog_properties(self) -> dict[str, str]:
        """Properties for a pyiceberg SqlCatalog whose warehouse is physically on MinIO.

        Property keys verified against https://py.iceberg.apache.org/configuration/.
        ``s3.force-virtual-addressing=false`` selects path-style access (required by MinIO).
        """
        return {
            "uri": self.catalog_uri,
            "warehouse": self.warehouse,
            "s3.endpoint": self.minio_endpoint,
            "s3.access-key-id": self.minio_access_key,
            "s3.secret-access-key": self.minio_secret_key,
            "s3.region": self.minio_region,
            "s3.force-virtual-addressing": "false",
        }

    def boto3_client_kwargs(self) -> dict[str, str]:
        """Kwargs for boto3.client('s3', ...) pointed at MinIO."""
        return {
            "endpoint_url": self.minio_endpoint,
            "aws_access_key_id": self.minio_access_key,
            "aws_secret_access_key": self.minio_secret_key,
            "region_name": self.minio_region,
        }

    def ensure_dirs(self) -> None:
        """Create local state directories. Fails loudly if the path is unusable."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.lancedb_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise ConfigError(
                f"Cannot create data directory {self.data_dir!r}: {exc}"
            ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings, raising a clear ConfigError on any validation failure."""
    try:
        return Settings()
    except Exception as exc:  # pydantic ValidationError or similar
        raise ConfigError(
            "Invalid tracevault configuration. Check your environment / .env file.\n"
            f"Details: {exc}"
        ) from exc
