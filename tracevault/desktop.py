"""The desktop product: tracevault in a native window, double-click simple.

``tracevault desktop`` brings up the whole local stack in one shot and shows it in a real
OS window (via pywebview, which uses the system WebView2 on Windows — no Node, no Rust, no
browser tab):

    1. start a real local MinIO server (zero-Docker) and wait until it is healthy
    2. start the FastAPI app (uvicorn) on a background thread — its lifespan ensures the
       bucket, loads Iceberg + the embedding model, and starts the auto-ingest daemon
    3. wait for /health, then open the native window pointing at the local app
    4. when the window is closed, shut the server and MinIO down cleanly

Every step fails LOUDLY with an actionable message; there is no degraded mode. pywebview is
an OPTIONAL dependency (``uv sync --extra desktop``) so the core install stays lean; the
browser UI (``tracevault app``) needs none of this.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import urllib.request
import webbrowser

from .config import Settings

logger = logging.getLogger(__name__)


class DesktopError(RuntimeError):
    """Raised when the desktop app cannot start its server, storage, or window."""


def _loopback(host: str) -> str:
    """A host the local machine can actually connect to (a bind-all address is not one)."""
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _webview2_present() -> bool:
    """True if the Edge WebView2 runtime (what pywebview renders with) is installed.

    Standard on Windows 11, but not guaranteed on a stripped image — if it is missing we
    open the system browser instead of crashing opaquely inside pywebview.
    """
    if sys.platform != "win32":
        return True  # mac/linux use the OS webview; let pywebview decide there
    import winreg

    key = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(root, key) as k:
                version, _ = winreg.QueryValueEx(k, "pv")
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


class _DesktopApi:
    """Methods exposed to the web UI as ``window.pywebview.api.*`` (desktop mode only).

    The browser cannot hand a backend a real folder path (sandboxing), so onboarding's
    "Browse…" button calls these to open a genuine native folder picker.
    """

    def __init__(self) -> None:
        self.window: object | None = None

    def pick_folder(self) -> str | None:
        import webview

        if self.window is None:
            return None
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)  # type: ignore[attr-defined]
        if not result:
            return None
        return result[0]

    # A git repo is just a folder; same native picker.
    pick_repo = pick_folder


def _start_server(settings: Settings, host: str, port: int) -> tuple[object, threading.Thread]:
    import uvicorn

    from .api import app as fastapi_app

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    # uvicorn skips installing signal handlers off the main thread, so this is safe.
    thread = threading.Thread(target=server.run, name="tracevault-uvicorn", daemon=True)
    thread.start()
    return server, thread


def _wait_health(host: str, port: int, thread: threading.Thread, timeout: float) -> None:
    """Block until the app answers /health, the server thread dies, or we time out."""
    url = f"http://{_loopback(host)}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise DesktopError(
                "The tracevault server stopped during startup. The most likely cause is MinIO "
                "being unreachable or the embedding model failing to load — see the log lines above."
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310 (local URL)
                if int(resp.status) == 200:
                    return
        except Exception:
            pass
        time.sleep(0.4)
    raise DesktopError(f"tracevault did not become ready within {timeout:.0f}s (waiting on {url}).")


def _block_until_dead(thread: threading.Thread) -> None:
    """Keep the process alive (and interruptible) while the server thread runs."""
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def _open_window(host: str, port: int) -> None:
    try:
        import webview
    except ImportError as exc:
        raise DesktopError(
            "The desktop window needs pywebview, which is an optional dependency. Install it with:\n"
            "  uv sync --extra desktop\n"
            "then run `tracevault desktop` again. (Or use `tracevault app` for the browser UI.)"
        ) from exc

    api = _DesktopApi()
    window = webview.create_window(
        "tracevault",
        f"http://{_loopback(host)}:{port}",
        js_api=api,
        width=1360,
        height=900,
        min_size=(1000, 640),
    )
    api.window = window
    logger.info("Opening tracevault desktop window.")
    webview.start()  # blocks on the main thread until the window is closed


def run_desktop(
    settings: Settings,
    *,
    host: str,
    port: int,
    manage_minio: bool = True,
    open_window: bool = True,
    health_timeout: float = 300.0,
) -> None:
    """Start MinIO + the app, open the native window, and tear everything down on close."""
    minio = None
    if manage_minio:
        from .local_minio import LocalMinio

        minio = LocalMinio(settings)

    server: object | None = None
    thread: threading.Thread | None = None
    try:
        if minio is not None:
            minio.start()  # fails loudly if a real MinIO cannot be brought up
        server, thread = _start_server(settings, host, port)
        _wait_health(host, port, thread, timeout=health_timeout)
        url = f"http://{_loopback(host)}:{port}"
        logger.info("tracevault is live at %s", url)
        if open_window and _webview2_present():
            _open_window(host, port)  # blocks on the main thread until the window closes
        elif open_window:
            # No native WebView2 runtime — degrade the *window*, not the data: use the browser.
            logger.warning("Edge WebView2 runtime not found; opening the system browser instead.")
            webbrowser.open(url)
            _block_until_dead(thread)
        else:
            # Headless desktop mode (e.g. on a server / for testing): run until interrupted.
            _block_until_dead(thread)
    finally:
        logger.info("Shutting tracevault down…")
        if server is not None:
            server.should_exit = True  # type: ignore[attr-defined]
        if thread is not None:
            thread.join(timeout=15)
        if minio is not None:
            minio.stop()
