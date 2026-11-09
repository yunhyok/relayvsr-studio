"""Managed inference environment.

The installer ships only the GUI. On first run the app creates a private Python
environment with ``uv`` (bundled), installs PyTorch CUDA 12.8 + Triton + the engine
dependencies from a pinned lock file, and installs the ``relayvsr_studio`` wheel into
it. The GUI then talks to ``python -m relayvsr_studio.engine.worker`` running inside
that environment. When the GUI itself runs from a source checkout whose interpreter
already has torch (development), the current interpreter is used directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..paths import app_home, bundle_dir, default_env_dir, is_frozen

Progress = Callable[[dict], None]

PYTHON_VERSION = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
LOCK_NAME = "engine-cu128.txt"


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


@dataclass
class EnvStatus:
    mode: str  # "dev" | "managed"
    python: str | None
    exists: bool = False
    torch: str | None = None
    cuda: str | None = None
    triton: str | None = None
    app_version: str | None = None
    gpu: str | None = None
    error: str | None = None
    details: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.exists and bool(self.torch) and self.error is None and (self.mode == "dev" or self.app_version == __version__)

    def to_dict(self) -> dict:
        return {**self.__dict__, "ready": self.ready}


_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    out["error"] = f"torch: {e}"
try:
    import triton
    out["triton"] = triton.__version__
except Exception as e:
    out["triton"] = None
try:
    import importlib.metadata as m
    out["app_version"] = m.version("relayvsr-studio")
except Exception:
    out["app_version"] = None
try:
    import relayvsr_studio
    out["app_import"] = relayvsr_studio.__version__
except Exception as e:
    out["app_import"] = None
print(json.dumps(out))
"""


class EnvironmentManager:
    def __init__(self, env_dir: Path | None = None):
        self.env_dir = Path(env_dir) if env_dir else default_env_dir()
        self.home = app_home()
        self.tools_dir = self.home / "tools"
        self.cache_dir = self.home / "cache"
        self._process: subprocess.Popen | None = None
        self._cancel = threading.Event()

    # ------------------------------------------------------------ discovery
    @staticmethod
    def dev_mode() -> bool:
        if is_frozen():
            return False
        try:
            import torch  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def python_executable(self) -> Path:
        if self.dev_mode():
            return Path(sys.executable)
        return self.env_dir / "Scripts" / "python.exe" if os.name == "nt" else self.env_dir / "bin" / "python"

    def source_root(self) -> Path | None:
        """Repository root when running from source (so the worker can import the package)."""
        if is_frozen():
            return None
        root = Path(__file__).resolve().parents[2]
        return root if (root / "relayvsr_studio" / "__init__.py").exists() else None

    def worker_command(self) -> tuple[list[str], dict[str, str]]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        root = self.source_root()
        if root is not None:
            env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return [str(self.python_executable()), "-X", "utf8", "-m", "relayvsr_studio.engine.worker"], env

    def lock_file(self) -> Path:
        candidates = [
            bundle_dir() / "requirements" / LOCK_NAME,
            Path(__file__).resolve().parents[2] / "requirements" / LOCK_NAME,
            Path(__file__).resolve().parent / LOCK_NAME,
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Engine requirements lock ({LOCK_NAME}) not found.")

    def app_wheel(self) -> Path | None:
        for folder in (bundle_dir() / "wheels", Path(__file__).resolve().parents[2] / "dist"):
            if folder.is_dir():
                wheels = sorted(folder.glob("relayvsr_studio-*.whl"))
                if wheels:
                    return wheels[-1]
        return None

    # ------------------------------------------------------------------ uv
    def uv_executable(self, download: bool = True, progress: Progress | None = None) -> Path:
        name = "uv.exe" if os.name == "nt" else "uv"
        for candidate in (bundle_dir() / "tools" / name, self.tools_dir / name):
            if candidate.is_file():
                return candidate
        found = shutil.which("uv")
        if found:
            return Path(found)
        if not download:
            raise FileNotFoundError("uv not found")
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        archive = self.tools_dir / "uv.zip"
        if progress:
            progress({"type": "log", "message": f"Downloading uv from {UV_RELEASE}"})
        urllib.request.urlretrieve(UV_RELEASE, archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith(name):
                    with zf.open(member) as src, open(self.tools_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        archive.unlink(missing_ok=True)
        return self.tools_dir / name

    # -------------------------------------------------------------- status
    def status(self, timeout: float = 120.0) -> EnvStatus:
        mode = "dev" if self.dev_mode() else "managed"
        python = self.python_executable()
        status = EnvStatus(mode=mode, python=str(python), exists=python.is_file())
        if not status.exists:
            return status
        cmd, env = self.worker_command()
        try:
            result = subprocess.run(
                [cmd[0], "-X", "utf8", "-c", _PROBE], capture_output=True, text=True, timeout=timeout, env=env, creationflags=_no_window(), encoding="utf-8", errors="replace"
            )
            line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            data = json.loads(line) if line.startswith("{") else {}
            if not data:
                status.error = (result.stderr or result.stdout or "probe produced no output").strip()[-2000:]
                return status
            status.torch = data.get("torch")
            status.cuda = data.get("cuda")
            status.triton = data.get("triton")
            status.gpu = data.get("gpu")
            status.app_version = data.get("app_version") or data.get("app_import")
            status.error = data.get("error")
            status.details = data
            if mode == "dev":
                status.app_version = __version__
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as error:
            status.error = str(error)
        return status

    # --------------------------------------------------------------- setup
    def cancel(self) -> None:
        self._cancel.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def _run(self, cmd: list[str], progress: Progress | None, env: dict[str, str] | None = None) -> None:
        if progress:
            progress({"type": "command", "message": " ".join(str(c) for c in cmd)})
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, creationflags=_no_window(), bufsize=1
        )
        assert self._process.stdout is not None
        tail: list[str] = []
        for line in self._process.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-40:]
            if progress:
                progress({"type": "log", "message": line})
            if self._cancel.is_set():
                self._process.kill()
                raise InterruptedError("Setup cancelled")
        code = self._process.wait()
        self._process = None
        if code != 0:
            raise RuntimeError(f"Command failed ({code}): {' '.join(str(c) for c in cmd)}\n" + "\n".join(tail[-15:]))

    def create(self, progress: Progress | None = None, *, reinstall_app_only: bool = False) -> EnvStatus:
        """Create or repair the managed environment. Blocking; run on a thread."""
        self._cancel.clear()
        if self.dev_mode():
            return self.status()
        uv = self.uv_executable(progress=progress)
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(self.cache_dir)
        env["UV_PYTHON_INSTALL_DIR"] = str(self.home / "python")
        env["UV_LINK_MODE"] = "copy"
        env["UV_HTTP_TIMEOUT"] = "600"
        env["UV_NO_PROGRESS"] = "1"
        python = self.python_executable()
        if not reinstall_app_only or not python.is_file():
            if progress:
                progress({"type": "step", "step": "venv", "message": f"Creating Python {PYTHON_VERSION} environment"})
            self.env_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run([str(uv), "venv", str(self.env_dir), "--python", PYTHON_VERSION, "--seed", "--allow-existing"], progress, env)
            if progress:
                progress({"type": "step", "step": "torch", "message": "Installing PyTorch (CUDA 12.8), Triton and engine packages (~3.5 GB download)"})
            self._run(
                [str(uv), "pip", "install", "--python", str(python), "-r", str(self.lock_file()), "--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"],
                progress,
                env,
            )
        wheel = self.app_wheel()
        if progress:
            progress({"type": "step", "step": "app", "message": "Installing RelayVSR Studio engine package"})
        if wheel is not None:
            self._run([str(uv), "pip", "install", "--python", str(python), "--no-deps", "--reinstall", str(wheel)], progress, env)
        else:
            root = self.source_root()
            if root is None:
                raise FileNotFoundError("Application wheel not found in the bundle.")
            if progress:
                progress({"type": "log", "message": f"Using source tree at {root} (PYTHONPATH)"})
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        status = self.status()
        if progress:
            progress({"type": "step", "step": "done", "message": "Environment ready" if status.ready else f"Environment check failed: {status.error}"})
        return status

    def remove(self) -> None:
        shutil.rmtree(self.env_dir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)


__all__ = ["EnvStatus", "EnvironmentManager", "LOCK_NAME", "PYTHON_VERSION", "TORCH_INDEX"]
