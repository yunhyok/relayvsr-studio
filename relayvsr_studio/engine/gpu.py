"""GPU detection, VRAM budgeting and per-architecture presets.

The planner estimates the memory the Wan keyframe model needs for a given input size
and picks the cheapest configuration that fits the currently free VRAM:

1. BF16 weights, full-frame keyframes (reference quality, >= 24 GB class GPUs),
2. FP8 weight-only storage (RTX 40/50 with 16 GB, e.g. the RTX 5070 Ti),
3. FP8 + spatial tiling of the keyframe model with parked tile KV caches,
4. additionally CPU block offload of the Wan transformer (8-12 GB GPUs).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass

import torch

from .geometry import RelayGeometry

log = logging.getLogger(__name__)

GIB = 1024**3

WAN_PARAMS = 5.03e9
WAN_BLOCK_PARAMS = 4.9e9  # parameters living in the 30 transformer blocks
WAN_LAYERS = 30
WAN_DIM = 3072
FLASH_BYTES = 0.08 * GIB  # 32M params in BF16 + RoPE tables
CONTEXT_RESERVE = 0.75 * GIB  # CUDA context, cuBLAS/cuDNN workspaces, allocator slack


@dataclass
class GpuInfo:
    available: bool
    name: str = "CPU"
    index: int = 0
    capability: tuple[int, int] = (0, 0)
    total_bytes: int = 0
    free_bytes: int = 0
    driver: str = ""
    torch_version: str = torch.__version__
    cuda_version: str = torch.version.cuda or ""
    triton: bool = False
    platform: str = platform.platform()

    @property
    def architecture(self) -> str:
        major, minor = self.capability
        return {
            (12, 0): "Blackwell (RTX 50)",
            (10, 0): "Blackwell (B200)",
            (8, 9): "Ada Lovelace (RTX 40)",
            (8, 6): "Ampere (RTX 30)",
            (8, 0): "Ampere (A100)",
            (9, 0): "Hopper (H100)",
            (7, 5): "Turing (RTX 20)",
        }.get((major, minor), f"sm_{major}{minor}")

    @property
    def native_fp8(self) -> bool:
        return self.capability >= (8, 9)

    @property
    def supports_bf16(self) -> bool:
        return self.capability >= (8, 0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["architecture"] = self.architecture
        data["native_fp8"] = self.native_fp8
        data["total_gib"] = round(self.total_bytes / GIB, 2)
        data["free_gib"] = round(self.free_bytes / GIB, 2)
        return data


def query_memory(index: int = 0) -> tuple[int, int]:
    """(free, total) bytes of *physical* VRAM.

    On Windows (WDDM) ``cudaMemGetInfo`` ignores memory held by other processes and the
    driver silently pages overflow into system RAM, so NVML is preferred: it reports what
    is really resident. Falls back to torch when NVML is unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free), int(info.total)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - optional dependency / unsupported platform
        free, total = torch.cuda.mem_get_info(index)
        return int(free), int(total)


def gpu_info(index: int = 0, probe_triton: bool = True) -> GpuInfo:
    if not torch.cuda.is_available():
        return GpuInfo(available=False)
    props = torch.cuda.get_device_properties(index)
    free, total = query_memory(index)
    driver = ""
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - optional
        pass
    triton = False
    if probe_triton:
        from . import backends

        triton = backends.triton_available(torch.device("cuda", index))
    return GpuInfo(
        available=True,
        name=props.name,
        index=index,
        capability=(props.major, props.minor),
        total_bytes=total,
        free_bytes=free,
        driver=driver,
        triton=triton,
    )


@dataclass
class MemoryPlan:
    wan_weights: str  # "bf16" | "fp8"
    block_offload: bool
    kv_window: int
    tile_max_tokens: int  # 0 = no tiling
    tile_kv_offload: bool
    estimate_bytes: int
    free_bytes: int
    tokens: int
    tiles: int
    reason: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["estimate_gib"] = round(self.estimate_bytes / GIB, 2)
        data["free_gib"] = round(self.free_bytes / GIB, 2)
        return data


def kv_bytes_per_token(kv_window: int, dtype_bytes: int = 2) -> int:
    return WAN_LAYERS * kv_window * 2 * WAN_DIM * dtype_bytes


def estimate_bytes(
    geometry: RelayGeometry,
    height: int,
    width: int,
    *,
    wan_weights: str,
    block_offload: bool,
    kv_window: int,
    tile_tokens: int | None,
    dtype_bytes: int = 2,
) -> tuple[int, int]:
    """Return (estimated peak bytes, Wan tokens per frame)."""
    th, tw = geometry.wan_tokens(height, width)
    tokens = th * tw
    active_tokens = min(tokens, tile_tokens) if tile_tokens else tokens
    if block_offload:
        per_block = WAN_BLOCK_PARAMS / WAN_LAYERS * (1 if wan_weights == "fp8" else dtype_bytes)
        weights = 2 * per_block + (WAN_PARAMS - WAN_BLOCK_PARAMS) * dtype_bytes
    else:
        weights = WAN_BLOCK_PARAMS * (1 if wan_weights == "fp8" else dtype_bytes) + (WAN_PARAMS - WAN_BLOCK_PARAMS) * dtype_bytes
        if wan_weights == "fp8":
            weights += 0.02 * GIB  # scales
    kv = active_tokens * kv_bytes_per_token(kv_window, dtype_bytes)
    # Transformer activations (~a dozen token-sized tensors + FFN) and dequantized weight temp.
    activations = active_tokens * (WAN_DIM * dtype_bytes * 14 + 14336 * dtype_bytes * 2)
    if wan_weights == "fp8":
        activations += WAN_DIM * 14336 * dtype_bytes
    ph, pw = geometry.padded_size(height, width)
    flash_tokens = (ph // geometry.latent_stride) * (pw // geometry.latent_stride)
    flash = FLASH_BYTES + flash_tokens * 512 * dtype_bytes * 24 + flash_tokens * 128 * dtype_bytes * 2 * 4 * 10
    out_h, out_w = geometry.output_size(ph, pw)
    frames = out_h * out_w * 3 * (dtype_bytes * 4 + 2)
    total = weights + kv + activations + flash + frames + CONTEXT_RESERVE
    return int(total), tokens


def plan_memory(
    info: GpuInfo,
    geometry: RelayGeometry,
    height: int,
    width: int,
    *,
    prefer: str = "auto",  # auto | quality | balanced | low_vram
    wan_weights: str = "auto",  # auto | bf16 | fp8
    block_offload: str | bool = "auto",
    tiling: str = "auto",  # auto | off | on
    tile_max_tokens: int | None = None,
    kv_window: int | None = None,
    free_bytes: int | None = None,
) -> MemoryPlan:
    free = info.free_bytes if free_bytes is None else free_bytes
    kv_window = kv_window or 4
    th, tw = geometry.wan_tokens(height, width)
    tokens = th * tw

    def fits(weights, offload, tile):
        est, _ = estimate_bytes(geometry, height, width, wan_weights=weights, block_offload=offload, kv_window=kv_window, tile_tokens=tile)
        return est <= free, est

    candidates: list[tuple[str, bool, int, str]] = []
    weight_options = ["bf16", "fp8"] if wan_weights == "auto" else [wan_weights]
    if prefer == "low_vram" and wan_weights == "auto":
        weight_options = ["fp8"]
    offload_options = [False, True] if block_offload == "auto" else [bool(block_offload)]
    if prefer == "low_vram" and block_offload == "auto":
        offload_options = [True]
    for offload in offload_options:
        for weights in weight_options:
            if tiling in ("auto", "off"):
                candidates.append((weights, offload, 0, "full-frame keyframes"))
    for offload in offload_options:
        for weights in weight_options:
            if tiling in ("auto", "on"):
                candidates.append((weights, offload, -1, "tiled keyframes"))

    chosen = None
    for weights, offload, tile, label in candidates:
        if tile == 0:
            ok, est = fits(weights, offload, None)
            if ok or tiling == "off":
                chosen = MemoryPlan(weights, offload, kv_window, 0, False, est, free, tokens, 1, label if ok else label + " (may exceed VRAM)")
                if ok:
                    break
                continue
        else:
            # Largest tile that fits; search in Wan tokens.
            budget_tokens = tile_max_tokens
            if budget_tokens is None:
                est_min, _ = estimate_bytes(geometry, height, width, wan_weights=weights, block_offload=offload, kv_window=kv_window, tile_tokens=1)
                per_token = kv_bytes_per_token(kv_window) + WAN_DIM * 2 * 14 + 14336 * 2 * 2
                budget_tokens = int((free - est_min) // per_token) if free > est_min else 0
                budget_tokens = max(0, min(budget_tokens, tokens))
            if budget_tokens < 512:
                continue
            if budget_tokens >= tokens:
                ok, est = fits(weights, offload, None)
                chosen = MemoryPlan(weights, offload, kv_window, 0, False, est, free, tokens, 1, "full-frame keyframes")
                break
            ok, est = fits(weights, offload, budget_tokens)
            from .runtime.streaming import plan_tiles

            n_tiles = len(plan_tiles(th, tw, budget_tokens, 8))
            chosen = MemoryPlan(weights, offload, kv_window, budget_tokens, True, est, free, tokens, n_tiles, label)
            if ok:
                break
    if chosen is None:
        est, _ = estimate_bytes(geometry, height, width, wan_weights="fp8", block_offload=True, kv_window=kv_window, tile_tokens=512)
        chosen = MemoryPlan("fp8", True, kv_window, 512, True, est, free, tokens, 0, "minimum configuration (VRAM is very tight)")
    log.info("Memory plan: %s", chosen.to_dict())
    return chosen


def apply_runtime_tuning(info: GpuInfo) -> dict:
    """Backend knobs that help Blackwell/Ada without changing model numerics."""
    applied = {}
    if not info.available:
        return applied
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    applied["cudnn_benchmark"] = True
    applied["bf16_reduced_precision_reduction"] = True
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

        applied["sdpa_backends"] = ["flash", "efficient", "cudnn", "math"]
    except ImportError:  # pragma: no cover
        pass
    return applied


__all__ = ["GpuInfo", "MemoryPlan", "apply_runtime_tuning", "estimate_bytes", "gpu_info", "kv_bytes_per_token", "plan_memory", "query_memory"]
