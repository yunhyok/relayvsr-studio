"""Application data locations (per-user, override with RELAYVSR_HOME)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_ID


def app_home() -> Path:
    override = os.environ.get("RELAYVSR_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_ID


def default_models_dir() -> Path:
    return app_home() / "models"


def default_env_dir() -> Path:
    return app_home() / "env"


def settings_file() -> Path:
    return app_home() / "settings.json"


def logs_dir() -> Path:
    return app_home() / "logs"


SCRATCH_ENV = "RELAYVSR_SCRATCH"
SCRATCH_FOLDER = "RelayVSR-scratch"


def scratch_root(folder: str | os.PathLike | None = None) -> Path | None:
    """``<folder>/RelayVSR-scratch`` for a user-chosen scratch folder (argument or RELAYVSR_SCRATCH)."""
    folder = folder or os.environ.get(SCRATCH_ENV)
    if not folder:
        return None
    return Path(folder) / SCRATCH_FOLDER


def scratch_subdir(name: str, folder: str | os.PathLike | None = None) -> Path | None:
    """Create and return a scratch subfolder, or None when no scratch folder is set or it is unusable."""
    root = scratch_root(folder)
    if root is None:
        return None
    path = root / name
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def scratch_environment(folder: str | os.PathLike | None) -> dict[str, str]:
    """Environment variables that send temporary files and the Triton kernel cache to the scratch folder."""
    tmp = scratch_subdir("tmp", folder)
    triton = scratch_subdir("triton-cache", folder)
    if tmp is None or triton is None:
        return {}
    return {SCRATCH_ENV: str(folder), "TEMP": str(tmp), "TMP": str(tmp), "TMPDIR": str(tmp), "TRITON_CACHE_DIR": str(triton)}


def clean_scratch(folder: str | os.PathLike | None, max_age_hours: float = 24.0) -> int:
    """Delete stale leftovers (crashed encodes, previews, temp files) from the scratch folder."""
    import shutil
    import time

    root = scratch_root(folder)
    removed = 0
    if root is None or not root.is_dir():
        return removed
    cutoff = time.time() - max_age_hours * 3600
    for name in ("encode", "previews", "tmp"):
        base = root / name
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory holding bundled resources (PyInstaller ``_internal`` or the package root)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


__all__ = [
    "SCRATCH_ENV",
    "app_home",
    "bundle_dir",
    "clean_scratch",
    "default_env_dir",
    "default_models_dir",
    "is_frozen",
    "logs_dir",
    "scratch_environment",
    "scratch_root",
    "scratch_subdir",
    "settings_file",
]
