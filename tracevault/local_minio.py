"""Zero-Docker MinIO: run a REAL MinIO server as a managed child process.

This is the "no Docker" path for the desktop product. It does NOT replace MinIO with
local files pretending to be a lakehouse — it launches the genuine MinIO binary, so the
Iceberg warehouse and blobs still live on real S3-protocol object storage, exactly as
they do under Docker. The only thing that changes is *who* starts MinIO: instead of the
Docker daemon, tracevault starts/stops ``minio`` itself, on a local data directory.

The binary is located in this order: an explicit ``TRACEVAULT_MINIO_BINARY`` path, then
the system ``PATH``, then a local cache under the data dir. If still missing and
``minio_auto_download`` is on, the official binary is downloaded from dl.min.io and
verified against its published SHA-256 before first use.

Licensing note: MinIO is AGPL-3.0. tracevault (MIT) does NOT bundle or redistribute it —
the binary is fetched/run on the end user's own machine, like invoking any locally
installed tool. Set ``TRACEVAULT_MINIO_AUTO_DOWNLOAD=0`` and point
``TRACEVAULT_MINIO_BINARY`` at your own install to opt out of the download entirely.

Fails loudly at every step — there is no degraded/no-storage mode.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

from .config import Settings

logger = logging.getLogger(__name__)

_DL_BASE = "https://dl.min.io/server/minio/release"
_HEALTH_PATH = "/minio/health/live"  # MinIO returns 200 here once the server is live


def _assign_kill_on_close_job(proc: subprocess.Popen[bytes]) -> int | None:
    """Put a child in a Win32 Job Object that dies when our process does.

    Without this, a hard-killed parent (Task Manager "End task", a crash) would orphan
    minio.exe — it would keep port 9000 bound and lock the data dir, so the next launch
    silently adopts a stale server. With KILL_ON_JOB_CLOSE, when our last handle to the
    job closes (which happens when this process terminates), Windows terminates the child.
    Returns the job handle to keep alive, or None if it could not be set up (we then fall
    back to the explicit terminate/kill ladder in stop()).
    """
    if sys.platform != "win32":
        return None
    try:
        from ctypes import wintypes

        ulong_ptr = ctypes.c_size_t

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ulong_ptr),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job_object_extended_limit_information = 9
        kill_on_job_close = 0x2000

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = kill_on_job_close
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        if not k32.SetInformationJobObject(
            job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
        ):
            return None
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            return None
        return int(job)
    except Exception as exc:  # never let process-group hygiene block startup
        logger.debug("Could not assign MinIO to a Job Object: %s", exc)
        return None


class LocalMinioError(RuntimeError):
    """Raised when a local MinIO server cannot be located, started, or reached."""


def _platform_slug() -> str:
    """dl.min.io path component for this OS/arch, e.g. 'windows-amd64'."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if sys.platform == "win32":
        return f"windows-{arch}"
    if sys.platform == "darwin":
        return f"darwin-{arch}"
    return f"linux-{arch}"


def _binary_filename() -> str:
    return "minio.exe" if sys.platform == "win32" else "minio"


class LocalMinio:
    """Manages the lifecycle of a real MinIO server process for zero-Docker mode."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.host, self.port = settings.minio_host_port
        self.console_port = self.port + 1
        self._proc: subprocess.Popen[bytes] | None = None
        self._external = False  # MinIO was already running (e.g. Docker); we don't manage it
        self._job: int | None = None  # Win32 Job Object handle (kills the child if we die)
        self._log_handle: IO[bytes] | None = None

    # --- health ---

    def _health_url(self) -> str:
        return f"http://{self.host}:{self.port}{_HEALTH_PATH}"

    def is_running(self) -> bool:
        """True if a MinIO server is already answering on the configured S3 port."""
        try:
            with urllib.request.urlopen(self._health_url(), timeout=1.0) as resp:  # noqa: S310 (local)
                return int(resp.status) == 200
        except Exception:
            return False

    # --- binary resolution ---

    def resolve_binary(self) -> Path:
        """Locate the MinIO executable, downloading it (verified) as a last resort."""
        override = self.settings.minio_binary
        if override:
            p = Path(override)
            if not p.is_file():
                raise LocalMinioError(
                    f"TRACEVAULT_MINIO_BINARY is set to {override!r} but no such file exists."
                )
            return p

        from shutil import which

        found = which("minio") or which("minio.exe")
        if found:
            return Path(found)

        cached = self.settings.minio_bin_path
        if cached.is_file():
            return cached

        if not self.settings.minio_auto_download:
            raise LocalMinioError(
                "MinIO binary not found on PATH or in the cache, and auto-download is off "
                "(TRACEVAULT_MINIO_AUTO_DOWNLOAD=0). Install MinIO and put it on PATH, or set "
                "TRACEVAULT_MINIO_BINARY to its full path. Download: https://min.io/download"
            )
        self._download(cached)
        return cached

    def _download(self, dest: Path) -> None:
        """Download the official MinIO binary and verify its published SHA-256."""
        slug = _platform_slug()
        release = self.settings.minio_release
        if release:
            # Pinned, reproducible build. dl.min.io names the per-release artifact
            # "minio.<RELEASE...>" on EVERY platform — there is no ".exe" segment in the
            # archive name (that suffix only exists on the unversioned windows binary).
            url = f"{_DL_BASE}/{slug}/archive/minio.{release}"
        else:
            url = f"{_DL_BASE}/{slug}/{_binary_filename()}"
        sha_url = f"{url}.sha256sum"
        logger.info("Downloading MinIO (%s) from %s …", slug, url)
        try:
            with urllib.request.urlopen(sha_url, timeout=30) as r:  # noqa: S310 (trusted host)
                # Format: "<hex>  minio[.exe]" — first whitespace-separated token is the digest.
                parts = r.read().decode("utf-8", "replace").split()
        except (urllib.error.URLError, OSError) as exc:
            raise LocalMinioError(
                f"Could not fetch the MinIO checksum from {sha_url}: {exc}. "
                "Check your network, or provide TRACEVAULT_MINIO_BINARY."
            ) from exc
        expected = parts[0].strip().lower() if parts else ""  # empty/whitespace body -> clean error below
        if len(expected) != 64:
            raise LocalMinioError(f"Unexpected MinIO checksum format from {sha_url!r}: {expected!r}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
            got = digest.hexdigest().lower()
            if got != expected:
                raise LocalMinioError(
                    f"MinIO download failed SHA-256 verification (expected {expected}, got {got}). "
                    "Refusing to run an unverified binary."
                )
            if sys.platform != "win32":
                tmp_path.chmod(0o755)
            tmp_path.replace(dest)
            logger.info("MinIO downloaded and verified -> %s", dest)
        except (urllib.error.URLError, OSError) as exc:
            raise LocalMinioError(f"Failed to download MinIO from {url}: {exc}") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # --- lifecycle ---

    def start(self, ready_timeout: float = 40.0) -> None:
        """Start MinIO (or adopt an already-running one) and block until it is healthy."""
        if self.is_running():
            self._external = True
            logger.info("MinIO already running at %s:%s — using it (not managing its lifecycle).",
                        self.host, self.port)
            return

        if len(self.settings.minio_secret_key) < 8:
            raise LocalMinioError(
                "MinIO requires a root password (TRACEVAULT_MINIO_SECRET_KEY) of at least 8 "
                "characters. The default 'minioadmin' satisfies this; only the override is too short."
            )

        binary = self.resolve_binary()
        data_dir = self.settings.minio_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.data_dir / "minio.log"

        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = self.settings.minio_access_key
        env["MINIO_ROOT_PASSWORD"] = self.settings.minio_secret_key
        cmd = [
            str(binary), "server", str(data_dir),
            "--address", f"{self.host}:{self.port}",
            "--console-address", f"{self.host}:{self.console_port}",
        ]
        # Hide the console window and put the child in its own group so a stray Ctrl-C in a
        # launching console doesn't tear MinIO down out from under us.
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        logger.info("Starting MinIO: %s", " ".join(cmd))
        self._log_handle = open(log_path, "wb")  # noqa: SIM115 (closed in stop())
        try:
            self._proc = subprocess.Popen(
                cmd, env=env, stdout=self._log_handle, stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._log_handle.close()
            raise LocalMinioError(f"Could not launch MinIO ({binary}): {exc}") from exc

        # Tie the child's lifetime to ours so it can never be orphaned (Windows).
        self._job = _assign_kill_on_close_job(self._proc)
        self._wait_ready(ready_timeout, log_path)

    def _wait_ready(self, timeout: float, log_path: Path) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                self.stop()  # release the log + job handles symmetrically with the timeout path
                raise LocalMinioError(
                    f"MinIO exited immediately (code {code}). Last log lines:\n{_tail(log_path)}"
                )
            if self.is_running():
                logger.info("MinIO is healthy at %s:%s", self.host, self.port)
                return
            time.sleep(0.4)
        self.stop()
        raise LocalMinioError(
            f"MinIO did not become healthy within {timeout:.0f}s. Last log lines:\n{_tail(log_path)}"
        )

    def stop(self) -> None:
        """Terminate the managed MinIO process (no-op for an adopted external server)."""
        if self._external or self._proc is None:
            return
        proc = self._proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MinIO did not stop gracefully; killing it.")
                proc.kill()
                proc.wait(timeout=5)
        self._proc = None
        handle = getattr(self, "_log_handle", None)
        if handle is not None:
            handle.close()
            self._log_handle = None
        if self._job is not None and sys.platform == "win32":
            ctypes.WinDLL("kernel32").CloseHandle(self._job)
            self._job = None
        logger.info("MinIO stopped.")

    def __enter__(self) -> LocalMinio:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _tail(path: Path, lines: int = 15) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no log captured)"
    return "\n".join(text.splitlines()[-lines:]) or "(log empty)"
