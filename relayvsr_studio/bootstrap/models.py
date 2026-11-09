"""Resumable model downloads from Hugging Face (no extra dependencies beyond stdlib).

Files are fetched with HTTP range requests into ``<name>.part`` and renamed when the
size matches the server's Content-Length. ``huggingface_hub`` is used when importable
(it adds Xet-accelerated transfers); otherwise the stdlib path is used.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

MODEL_FILES = {
    4: {"repo": "kopper/RelayVSR", "files": ["flash.safetensors", "wan.safetensors"], "folder": "x4"},
    2: {"repo": "lightx2v/RelayVSR_X2", "files": ["flash_x2.safetensors", "wan_x2.safetensors"], "folder": "x2"},
}

APPROX_SIZES = {
    "flash.safetensors": 127_759_784,
    "wan.safetensors": 10_066_399_024,
    "flash_x2.safetensors": 127_700_000,
    "wan_x2.safetensors": 10_127_000_000,
}

Progress = Callable[[dict], None]


def model_folder(models_dir: str | Path, scale: int) -> Path:
    return Path(models_dir) / MODEL_FILES[scale]["folder"]


def models_present(models_dir: str | Path, scale: int) -> bool:
    folder = model_folder(models_dir, scale)
    return all((folder / name).is_file() and (folder / name).stat().st_size > 1_000_000 for name in MODEL_FILES[scale]["files"])


def missing_bytes(models_dir: str | Path, scale: int) -> int:
    folder = model_folder(models_dir, scale)
    total = 0
    for name in MODEL_FILES[scale]["files"]:
        path = folder / name
        have = path.stat().st_size if path.is_file() else 0
        total += max(0, APPROX_SIZES.get(name, 0) - have)
    return total


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _download_stdlib(url: str, destination: Path, progress: Progress | None, label: str, cancelled: Callable[[], bool] | None) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "relayvsr-studio-downloader/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416:  # already complete
            part.rename(destination)
            return
        raise
    with response:
        status = response.getcode()
        if have and status != 206:
            have = 0  # server ignored the range; start over
        length = response.headers.get("Content-Length")
        total = have + int(length) if length else APPROX_SIZES.get(destination.name, 0)
        mode = "ab" if have else "wb"
        last = 0.0
        done = have
        with open(part, mode) as handle:
            while True:
                if cancelled and cancelled():
                    raise InterruptedError("Download cancelled")
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.time()
                if progress and (now - last > 0.25 or done == total):
                    last = now
                    progress({
                        "type": "progress",
                        "file": label,
                        "done_mb": done / 1e6,
                        "total_mb": total / 1e6,
                        "percent": 100.0 * done / total if total else 0.0,
                    })
    if total and done < total:
        raise OSError(f"Incomplete download of {label}: {done}/{total} bytes")
    os.replace(part, destination)


def download_models(models_dir: str | Path, scale: int, progress: Progress | None = None, cancelled: Callable[[], bool] | None = None) -> Path:
    spec = MODEL_FILES[scale]
    folder = model_folder(models_dir, scale)
    folder.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        destination = folder / name
        if destination.is_file() and destination.stat().st_size > 1_000_000:
            if progress:
                progress({"type": "skip", "file": name, "message": "already present"})
            continue
        if progress:
            progress({"type": "start", "file": name, "repo": spec["repo"]})
        for attempt in range(1, 6):
            try:
                _download_stdlib(hf_url(spec["repo"], name), destination, progress, name, cancelled)
                break
            except InterruptedError:
                raise
            except (OSError, urllib.error.URLError) as error:
                if attempt == 5:
                    raise
                if progress:
                    progress({"type": "retry", "file": name, "attempt": attempt, "error": str(error)})
                time.sleep(min(30, 2**attempt))
        if progress:
            progress({"type": "complete", "file": name, "path": str(destination)})
    return folder


__all__ = ["APPROX_SIZES", "MODEL_FILES", "download_models", "missing_bytes", "model_folder", "models_present"]
