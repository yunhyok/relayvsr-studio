"""Load RelayVSR checkpoints (native 4x ``kopper/RelayVSR`` or 2x ``lightx2v/RelayVSR_X2``).

Both safetensors files embed their architecture config in the metadata; the Wan file
also carries the fixed prompt embeddings. Loading options cover compute dtype, FP8
weight-only storage of the Wan blocks, CPU block offload, the attention backend and
an optional smaller keyframe KV window for low-VRAM GPUs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from . import backends
from .geometry import RelayGeometry
from .models.flash.model import FlashVSRConditionNet
from .models.wan.transformer import WanTransformer3DModel
from .offload import BlockOffloader
from .quant import quantize_module_linears_

log = logging.getLogger(__name__)

WAN_FORMATS = {"refvsr-wan-v1": 4, "relayvsr-wan-x2-v1": 2}
FLASH_FORMATS = {"refvsr-flash-v1": 4, "relayvsr-flash-x2-v1": 2}

# Canonical file names per scale, as published on Hugging Face.
MODEL_FILES = {
    4: {"repo": "kopper/RelayVSR", "wan": "wan.safetensors", "flash": "flash.safetensors"},
    2: {"repo": "lightx2v/RelayVSR_X2", "wan": "wan_x2.safetensors", "flash": "flash_x2.safetensors"},
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass(frozen=True)
class LoadOptions:
    device: str = "cuda"
    dtype: str = "bf16"
    wan_weights: str = "bf16"  # "bf16" (compute dtype) or "fp8" (weight-only E4M3)
    block_offload: bool = False
    attention_backend: str = "auto"  # auto | triton | sdpa
    kv_window: int | None = None  # override Wan temporal_context_frames (1..training_frames)
    pin_memory: bool = True


@dataclass
class CheckpointInfo:
    path: Path
    component: str  # "wan" | "flash"
    format: str
    scale: int
    config: dict
    size_bytes: int


@dataclass
class LoadedModels:
    wan: WanTransformer3DModel
    flash: FlashVSRConditionNet
    geometry: RelayGeometry
    wan_info: CheckpointInfo
    flash_info: CheckpointInfo
    options: LoadOptions
    device: torch.device
    dtype: torch.dtype
    offloader: BlockOffloader | None = None
    load_seconds: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def scale(self) -> int:
        return self.geometry.scale


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    fmt = metadata.get("format", "")
    if fmt in WAN_FORMATS:
        component, scale = "wan", WAN_FORMATS[fmt]
    elif fmt in FLASH_FORMATS:
        component, scale = "flash", FLASH_FORMATS[fmt]
    else:
        raise ValueError(f"{path.name} is not a RelayVSR checkpoint (format={fmt!r}).")
    config = json.loads(metadata.get("config", "{}"))
    return CheckpointInfo(path, component, fmt, scale, config, path.stat().st_size)


def find_checkpoints(models_dir: str | Path, scale: int) -> tuple[Path, Path]:
    """Locate ``wan``/``flash`` files for a scale under ``models_dir`` (``x4``/``x2`` subfolders or flat)."""
    models_dir = Path(models_dir)
    names = MODEL_FILES[scale]
    candidates = [models_dir / f"x{scale}", models_dir]
    for folder in candidates:
        wan, flash = folder / names["wan"], folder / names["flash"]
        if wan.is_file() and flash.is_file():
            return wan, flash
    # Fall back to scanning metadata of any safetensors in the tree.
    wan = flash = None
    for file in sorted(models_dir.rglob("*.safetensors")):
        try:
            info = inspect_checkpoint(file)
        except (ValueError, OSError):
            continue
        if info.scale != scale:
            continue
        if info.component == "wan" and wan is None:
            wan = file
        elif info.component == "flash" and flash is None:
            flash = file
    if wan is None or flash is None:
        raise FileNotFoundError(
            f"RelayVSR {scale}x checkpoints not found under {models_dir} "
            f"(expected {names['wan']} and {names['flash']} from {names['repo']})."
        )
    return wan, flash


def _validate_pair(wan: CheckpointInfo, flash: CheckpointInfo) -> None:
    if wan.component != "wan" or flash.component != "flash":
        raise ValueError("Pass the Wan checkpoint first and the Flash checkpoint second.")
    if wan.scale != flash.scale:
        raise ValueError(f"Cannot mix a {wan.scale}x Wan checkpoint with a {flash.scale}x Flash checkpoint.")
    cfg = flash.config
    if cfg.get("condition_mode", "endpoints") != "endpoints":
        raise ValueError("Flash checkpoint must use endpoint conditioning.")
    if cfg.get("temporal_rope_policy", "current_relative") != "current_relative":
        raise ValueError("Flash checkpoint must use current_relative temporal RoPE.")
    if int(cfg.get("upscale_factor", 4)) != wan.scale:
        raise ValueError("Flash upscale_factor disagrees with the checkpoint format.")


@torch.no_grad()
def load_models(wan_path: str | Path, flash_path: str | Path, options: LoadOptions | None = None) -> LoadedModels:
    options = options or LoadOptions()
    started = time.perf_counter()
    device = torch.device(options.device)
    dtype = DTYPES[options.dtype]
    wan_info = inspect_checkpoint(wan_path)
    flash_info = inspect_checkpoint(flash_path)
    _validate_pair(wan_info, flash_info)
    geometry = RelayGeometry(wan_info.scale)
    backends.force_backend(None if options.attention_backend == "auto" else options.attention_backend)

    wan_config = dict(wan_info.config)
    training_frames = int((wan_config.get("temporal_rope") or {}).get("training_frames", 8))
    if options.kv_window is not None:
        if not 1 <= options.kv_window <= training_frames:
            raise ValueError(f"kv_window must be within 1..{training_frames}.")
        wan_config["temporal_context_frames"] = int(options.kv_window)
    flash_config = dict(flash_info.config)
    variant = flash_config.pop("variant", "S")

    with safe_open(str(wan_info.path), framework="pt", device="cpu") as handle:
        prompt_shape = tuple(handle.get_slice("prompt_embeds").get_shape())
    if len(prompt_shape) != 3 or prompt_shape[0] != 1 or prompt_shape[2] != wan_config["text_dim"]:
        raise ValueError("Checkpoint prompt shape does not match the Wan model.")

    with init_empty_weights():
        wan = WanTransformer3DModel.from_config(wan_config)
        wan.lq_proj = wan.build_lq_proj(unshuffle=geometry.wan_unshuffle)
        wan.register_buffer("prompt_embeds", torch.empty(prompt_shape))
        flash = FlashVSRConditionNet.from_variant(variant, **flash_config)

    log.info("Loading Wan %dx checkpoint (%.2f GB) ...", geometry.scale, wan_info.size_bytes / 1e9)
    state = load_file(str(wan_info.path), device="cpu")
    wan.load_state_dict(state, strict=True, assign=True)
    del state
    flash.load_state_dict(load_file(str(flash_info.path), device="cpu"), strict=True, assign=True)

    wan.eval().requires_grad_(False)
    flash.eval().requires_grad_(False)

    # Non-block parts of Wan always live on the device in the compute dtype.
    blocks = wan.blocks
    wan.blocks = torch.nn.ModuleList()
    wan.to(device=device, dtype=dtype)
    wan.blocks = blocks

    fp8 = options.wan_weights == "fp8"
    quantized_linears = 0
    for block in blocks:
        block.to(dtype=dtype)
        if fp8:
            block.to(device=device)
            quantized_linears += quantize_module_linears_(block, device=device, scale_dtype=dtype)
            if options.block_offload:
                block.to(device="cpu")
        elif not options.block_offload:
            block.to(device=device)
    offloader = None
    if options.block_offload:
        offloader = BlockOffloader(blocks, device, pin_memory=options.pin_memory)
        wan.block_offloader = offloader
    flash.to(device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    details = {
        "scale": geometry.scale,
        "wan_weights": "fp8" if fp8 else options.dtype,
        "quantized_linears": quantized_linears,
        "block_offload": bool(options.block_offload),
        "kv_window": wan.temporal_context_frames,
        "attention_backend": backends.forced_backend() or ("triton" if backends.triton_available(device) else "sdpa"),
        "wan_params": sum(p.numel() for p in wan.parameters()),
        "flash_params": sum(p.numel() for p in flash.parameters()),
    }
    log.info("Models ready in %.1fs: %s", elapsed, details)
    return LoadedModels(wan, flash, geometry, wan_info, flash_info, options, device, dtype, offloader, elapsed, details)


__all__ = [
    "CheckpointInfo",
    "DTYPES",
    "LoadOptions",
    "LoadedModels",
    "MODEL_FILES",
    "find_checkpoints",
    "inspect_checkpoint",
    "load_models",
]
