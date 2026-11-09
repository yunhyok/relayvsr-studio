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


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory holding bundled resources (PyInstaller ``_internal`` or the package root)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


__all__ = ["app_home", "bundle_dir", "default_env_dir", "default_models_dir", "is_frozen", "logs_dir", "settings_file"]
