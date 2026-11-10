"""Persisted user settings (JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..paths import default_env_dir, default_models_dir, settings_file


@dataclass
class Settings:
    language: str = "auto"
    models_dir: str = str(default_models_dir())
    env_dir: str = str(default_env_dir())
    scratch_dir: str = ""  # "" -> temp files next to the output, caches in the default locations
    output_dir: str = ""  # "" -> same folder as the input
    scale: int = 4
    size_mode: str = "native"  # native | fit1080 | fit1440 | fit2160 | width
    custom_width: int = 3840
    codec: str = "libx264"
    crf: int = 18
    preset: str = "fast"
    copy_audio: bool = True
    overwrite: bool = False
    wan_weights: str = "auto"
    vram_mode: str = "auto"  # auto | full | tiled | offload | low_vram | quality
    attention_backend: str = "auto"
    window_size: int = 16
    seed: int = 42
    detail: float = 1.0
    kv_window: int = 0  # 0 = checkpoint default
    preview: bool = True
    ffmpeg_path: str = ""
    use_external_ffmpeg: bool = False
    last_input_dir: str = ""
    window_geometry: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        path = settings_file()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in names})
            except (OSError, ValueError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        path = settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def models_path(self) -> Path:
        return Path(self.models_dir) if self.models_dir else default_models_dir()

    def env_path(self) -> Path:
        return Path(self.env_dir) if self.env_dir else default_env_dir()

    def scratch_path(self) -> Path | None:
        return Path(self.scratch_dir) if self.scratch_dir.strip() else None


__all__ = ["Settings"]
