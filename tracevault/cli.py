"""tracevault CLI (typer): ingest real sources, serve the UI, inspect, reset.

  uv run tracevault ingest --repo PATH [--ai-logs PATH]
  uv run tracevault serve
"""

from __future__ import annotations

import logging

import typer

from .config import get_settings

app = typer.Typer(add_completion=False, help="Real knowledge base on Iceberg + MinIO.")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    repo: str | None = typer.Option(None, "--repo", help="Path to a REAL Git repository to ingest."),
    folder: str | None = typer.Option(
        None, "--folder", help="Path to ANY real folder (no git needed); every file becomes a speaking artifact."
    ),
    ai_logs: str | None = typer.Option(
        None, "--ai-logs", help="Path to REAL Claude Code logs (a project dir, the projects root, "
        "or a single .jsonl). Defaults to ~/.claude/projects if omitted."
    ),
    include_ai: bool = typer.Option(
        True, "--ai/--no-ai", help="Whether to also ingest Claude Code logs."
    ),
    max_commits: int | None = typer.Option(None, help="Limit number of commits (default: all)."),
    max_sessions: int | None = typer.Option(None, help="Limit number of AI sessions (default: all)."),
    max_files: int | None = typer.Option(None, help="Limit number of files for --folder ingest."),
    images: bool = typer.Option(
        True, "--images/--no-images",
        help="Caption images with a local vision model so they 'speak' (slower; --no-images for speed).",
    ),
) -> None:
    """Ingest a real Git repo OR any folder (+ real AI logs), then embed for search."""
    _setup_logging()
    if bool(repo) == bool(folder):
        raise typer.BadParameter("Provide exactly one of --repo (a Git repo) or --folder (any directory).")
    # Imported lazily so `serve`/`reset` don't pay the heavy ML import cost.
    from .embed import Embedder
    from .ingest_ai import ingest_ai
    from .lakehouse import Lakehouse
    from .storage import BlobStore

    settings = get_settings()
    store = BlobStore(settings)
    store.ensure_bucket()
    lake = Lakehouse(settings)

    if repo:
        from .ingest_git import ingest_git

        g = ingest_git(repo, lake, store, max_commits=max_commits, enable_images=images)
        file_index = g.file_id_by_abspath
        typer.echo(f"git:  {g.as_dict()}")
    else:
        from .ingest_folder import ingest_folder

        f = ingest_folder(folder, lake, store, enable_images=images, max_files=max_files)  # type: ignore[arg-type]
        file_index = f.file_id_by_abspath
        typer.echo(f"folder: {f.as_dict()}")

    if include_ai:
        a = ingest_ai(ai_logs, lake, store, git_file_index=file_index, max_sessions=max_sessions)
        typer.echo(f"ai:   {a.as_dict()}")
    else:
        typer.echo("ai:   skipped (--no-ai)")

    typer.echo("embedding artifacts (loading model)…")
    n = Embedder(settings).embed_artifacts(lake)
    typer.echo(f"embedded {n} new artifacts")
    typer.echo(f"counts: {lake.counts()}")
    typer.secho("Ingest complete. Run `tracevault serve` and open the UI.", fg=typer.colors.GREEN)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default from settings)."),
    port: int | None = typer.Option(None, help="Bind port (default from settings)."),
) -> None:
    """Serve the API + browser UI (loads MinIO, Iceberg, and the embedding model)."""
    _setup_logging()
    import uvicorn

    from .api import app as fastapi_app

    settings = get_settings()
    h = host or settings.api_host
    p = port or settings.api_port
    typer.secho(f"Serving tracevault on http://{h}:{p}", fg=typer.colors.GREEN)
    uvicorn.run(fastapi_app, host=h, port=p)


@app.command("app")
def app_cmd(
    host: str | None = typer.Option(None, help="Bind host."),
    port: int | None = typer.Option(None, help="Bind port."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the UI in your browser."),
) -> None:
    """Run tracevault as a local knowledge service: serve the UI + keep it current automatically."""
    import os

    os.environ["TRACEVAULT_AUTO_INGEST"] = "1"  # must precede the first settings read
    _setup_logging()
    import threading
    import webbrowser

    import uvicorn

    from .api import app as fastapi_app
    from .sources import load_sources

    settings = get_settings()
    h = host or settings.api_host
    p = port or settings.api_port
    n = len(load_sources(settings))
    if n == 0:
        typer.secho(
            "No sources yet. Add some so the service has something to keep current, e.g.:\n"
            "  tracevault add-source --repo /path/to/repo\n"
            "  tracevault add-source --folder /path/to/folder",
            fg=typer.colors.YELLOW,
        )
    typer.secho(
        f"tracevault is live at http://{h}:{p}  ·  auto-ingesting {n} source(s) every "
        f"{settings.auto_ingest_interval}s",
        fg=typer.colors.GREEN,
    )
    if open_browser:
        threading.Timer(2.5, lambda: webbrowser.open(f"http://{h}:{p}")).start()
    uvicorn.run(fastapi_app, host=h, port=p)


@app.command("desktop")
def desktop_cmd(
    host: str | None = typer.Option(None, help="Bind host."),
    port: int | None = typer.Option(None, help="Bind port."),
    manage_minio: bool = typer.Option(
        True, "--manage-minio/--external-minio",
        help="Run a real local MinIO server (zero-Docker). Use --external-minio for Docker/your own.",
    ),
    window: bool = typer.Option(
        True, "--window/--no-window", help="Open a native window (else serve headless)."
    ),
) -> None:
    """Run tracevault as a native desktop app: zero-Docker, one window, always-on."""
    import os

    os.environ["TRACEVAULT_AUTO_INGEST"] = "1"  # always-on service; must precede first settings read
    if manage_minio:
        os.environ["TRACEVAULT_MANAGE_MINIO"] = "1"
    _setup_logging()
    from .desktop import DesktopError, run_desktop
    from .local_minio import LocalMinioError

    settings = get_settings()
    h = host or settings.api_host
    p = port or settings.api_port
    typer.secho("Starting tracevault desktop (this can take a moment on first run)…", fg=typer.colors.GREEN)
    try:
        run_desktop(settings, host=h, port=p, manage_minio=manage_minio, open_window=window)
    except (DesktopError, LocalMinioError) as exc:
        typer.secho(f"Could not start the desktop app:\n  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


autostart_app = typer.Typer(
    add_completion=False, help="Start tracevault automatically when you log in (Windows, per-user)."
)
app.add_typer(autostart_app, name="autostart")


@autostart_app.command("enable")
def autostart_enable() -> None:
    """Open tracevault automatically at login (creates one Startup shortcut)."""
    from .autostart import AutostartError, enable

    try:
        path = enable()
    except AutostartError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(f"Auto-start enabled. Shortcut: {path}", fg=typer.colors.GREEN)


@autostart_app.command("disable")
def autostart_disable() -> None:
    """Stop opening tracevault at login (removes the Startup shortcut)."""
    from .autostart import AutostartError, disable

    try:
        removed = disable()
    except AutostartError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Auto-start disabled." if removed else "Auto-start was not enabled.")


@autostart_app.command("status")
def autostart_status() -> None:
    """Show whether tracevault is set to open at login."""
    from .autostart import is_enabled, shortcut_path

    state = "enabled" if is_enabled() else "disabled"
    try:
        typer.echo(f"{state}  ({shortcut_path()})")
    except Exception:
        typer.echo(state)


@app.command("add-source")
def add_source_cmd(
    repo: str | None = typer.Option(None, "--repo", help="A real Git repository to keep ingested."),
    folder: str | None = typer.Option(None, "--folder", help="Any folder to keep ingested."),
) -> None:
    """Register a source that the local service keeps continuously ingested."""
    _setup_logging()
    from .sources import add_source, load_sources

    if bool(repo) == bool(folder):
        raise typer.BadParameter("Provide exactly one of --repo or --folder.")
    settings = get_settings()
    if repo:
        add_source(settings, repo, "repo")
    else:
        add_source(settings, folder, "folder")  # type: ignore[arg-type]
    typer.echo("Sources: " + ", ".join(f"{s.kind}:{s.path}" for s in load_sources(settings)))


@app.command("sources")
def sources_cmd() -> None:
    """List the sources the local service keeps ingested."""
    _setup_logging()
    from .sources import load_sources

    rows = load_sources(get_settings())
    if not rows:
        typer.echo("No sources. Add with `tracevault add-source --repo|--folder PATH`.")
    for s in rows:
        typer.echo(f"  {s.kind:7s} {s.path}")


@app.command("stream-emit")
def stream_emit(
    folder: str = typer.Option(..., "--folder", help="Folder to watch; new/changed files are streamed."),
    repo: str = typer.Option("live", "--repo", help="Project name for streamed files."),
    interval: float = typer.Option(2.0, help="Polling interval in seconds."),
) -> None:
    """Watch a folder and stream new/changed files as events (the producer)."""
    _setup_logging()
    from .streaming import watch_and_emit

    watch_and_emit(folder, repo, get_settings(), interval=interval)


@app.command("stream-consume")
def stream_consume(
    images: bool = typer.Option(True, "--images/--no-images", help="Caption streamed images."),
) -> None:
    """Consume file events and append them into the lakehouse live (the consumer)."""
    _setup_logging()
    from .streaming import consume_and_ingest

    consume_and_ingest(get_settings(), enable_images=images)


@app.command()
def marts() -> None:
    """Recompute the gold medallion marts from the silver lakehouse."""
    _setup_logging()
    from .lakehouse import Lakehouse
    from .marts import build_marts

    typer.echo(build_marts(Lakehouse(get_settings())))


@app.command()
def stats() -> None:
    """Show artifact / edge counts currently in the lakehouse."""
    _setup_logging()
    from .lakehouse import Lakehouse

    typer.echo(Lakehouse(get_settings()).counts())


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete ALL ingested data (MinIO objects + local catalog/index). Destructive."""
    _setup_logging()
    import shutil

    import boto3
    from botocore.config import Config as BotoConfig

    settings = get_settings()
    if not yes:
        typer.confirm(
            f"This deletes all objects in bucket {settings.bucket!r} and {settings.data_dir}. Continue?",
            abort=True,
        )
    s3 = boto3.client(
        "s3", config=BotoConfig(s3={"addressing_style": "path"}), **settings.boto3_client_kwargs()
    )
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.bucket):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3.delete_objects(Bucket=settings.bucket, Delete={"Objects": objs})
            deleted += len(objs)
    if settings.data_dir.exists():
        shutil.rmtree(settings.data_dir)
    typer.echo(f"Deleted {deleted} objects and removed {settings.data_dir}.")


if __name__ == "__main__":
    app()
