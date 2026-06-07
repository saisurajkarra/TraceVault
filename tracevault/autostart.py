"""Optional auto-start at login (Windows, per-user).

Drops a single shortcut in the per-user Startup folder so tracevault opens when you log in.
It is per-user (no admin) and trivially uninstallable — it is ONE visible file, which this
tool, Windows Settings > Apps > Startup, or Explorer can delete. The shortcut launches the
windowless ``pythonw.exe`` so nothing flashes a console at login, and it targets the venv's
interpreter directly (never the volatile ``uv`` shim path).

The shortcut is created via the WScript.Shell COM object, which exists on every Windows
install — no third-party dependency. Fails loudly on non-Windows or if the target is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SHORTCUT_NAME = "tracevault.lnk"


class AutostartError(RuntimeError):
    """Raised when auto-start cannot be configured (wrong OS, missing target, COM failure)."""


def _require_windows() -> None:
    if sys.platform != "win32":
        raise AutostartError("Auto-start at login is implemented for Windows only.")


def startup_dir() -> Path:
    """Per-user Startup folder, from the registry (honors OneDrive/Known-Folder redirection)."""
    _require_windows()
    import winreg

    key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
        raw, _ = winreg.QueryValueEx(k, "Startup")
    return Path(os.path.expandvars(raw))


def shortcut_path() -> Path:
    return startup_dir() / SHORTCUT_NAME


def _target() -> tuple[str, str, str]:
    """(exe, args, workdir) for the login launch — windowless and stable across uv upgrades."""
    if getattr(sys, "frozen", False):  # a future packaged .exe
        return sys.executable, "desktop", str(Path(sys.executable).resolve().parent)
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = str(pythonw if pythonw.exists() else sys.executable)
    workdir = str(Path(__file__).resolve().parent.parent)  # repo root so ./data resolves
    return exe, "-m tracevault desktop", workdir


def is_enabled() -> bool:
    try:
        return shortcut_path().exists()
    except AutostartError:
        return False


def enable() -> Path:
    """Create the Startup shortcut. Idempotent (overwrites). Returns its path."""
    _require_windows()
    exe, args, workdir = _target()
    if not Path(exe).exists():
        raise AutostartError(f"Launch target not found: {exe!r}; refusing to enable auto-start.")
    path = shortcut_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_shortcut(path, exe, args, workdir)
    if not path.exists():
        raise AutostartError(f"Failed to create the startup shortcut at {path}.")
    return path


def disable() -> bool:
    """Remove the Startup shortcut. Returns True if one was removed."""
    _require_windows()
    path = shortcut_path()
    if path.exists():
        path.unlink()
        return True
    return False


def _ps_quote(value: str) -> str:
    """Escape a value for a PowerShell single-quoted string."""
    return value.replace("'", "''")


def _write_shortcut(path: Path, exe: str, args: str, workdir: str) -> None:
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{_ps_quote(str(path))}'); "
        f"$s.TargetPath = '{_ps_quote(exe)}'; "
        f"$s.Arguments = '{_ps_quote(args)}'; "
        f"$s.WorkingDirectory = '{_ps_quote(workdir)}'; "
        "$s.WindowStyle = 7; "  # minimized, no-activate
        "$s.Description = 'tracevault — local knowledge base'; "
        "$s.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise AutostartError(f"Could not write the startup shortcut: {detail}") from exc
