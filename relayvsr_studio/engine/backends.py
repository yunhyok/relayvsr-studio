"""Attention backend selection: fused Triton kernels when available, SDPA otherwise.

The RelayVSR checkpoints use query-centred *local* attention in both networks:

* Wan keyframe model: a rectangular (22 x 40 token) window that is shifted inward at
  image borders ("shifted" semantics), over up to four cached frames.
* Flash decoder: truncated windows (24 x 24 for video history, 48 x 48 for the two
  endpoint conditions) over several KV planes.

Triton gives fused kernels for both. When Triton is missing (or fails to compile on
this machine) we fall back to a tiled gather + ``scaled_dot_product_attention``
formulation: queries are grouped in 16 x 16 tiles, each tile gathers the union of its
keys per plane, and a boolean mask restores the exact per-query window. This matches
the fused kernels numerically up to floating point rounding and is portable to any
CUDA GPU (it is several times slower than Triton, but still far faster than the
per-64-query reference path of the original release).

The tiled SDPA formulation follows LightX2V's RelayVSR SDPA backend (Apache-2.0).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

_TILE = 16
_CACHE_BYTES = 3 * 1024**3
_GATHER_BYTES = 2 * 1024**3

_lock = threading.Lock()
_triton_ok: bool | None = None
_forced: str | None = None  # "triton" | "sdpa" | None (auto)


def force_backend(name: str | None) -> None:
    """Force ``"triton"`` or ``"sdpa"``; ``None`` restores automatic selection."""
    global _forced
    if name not in (None, "auto", "triton", "sdpa"):
        raise ValueError(f"Unknown attention backend {name!r}")
    _forced = None if name == "auto" else name


def forced_backend() -> str | None:
    return _forced


def triton_available(device: torch.device | str | None = None) -> bool:
    """True when Triton imports and compiles a trivial kernel on this CUDA device."""
    global _triton_ok
    if os.environ.get("RELAYVSR_DISABLE_TRITON") == "1":
        return False
    with _lock:
        if _triton_ok is not None:
            return _triton_ok
        _triton_ok = False
        try:
            if not torch.cuda.is_available():
                return False
            import triton
            import triton.language as tl

            @triton.jit
            def _probe(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < n
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                tl.store(y_ptr + offsets, x * 2.0, mask=mask)

            dev = torch.device(device) if device is not None else torch.device("cuda")
            with torch.cuda.device(dev):
                x = torch.arange(256, device=dev, dtype=torch.float32)
                y = torch.empty_like(x)
                _probe[(1,)](x, y, 256, BLOCK=256)
                torch.cuda.synchronize(dev)
                _triton_ok = bool(torch.allclose(y, x * 2))
            if _triton_ok:
                log.info("Triton %s is available; fused local attention kernels enabled.", triton.__version__)
        except Exception as error:  # noqa: BLE001 - any failure means "no Triton"
            log.warning("Triton unavailable (%s: %s); using SDPA local attention.", type(error).__name__, error)
            _triton_ok = False
        return _triton_ok


def select_backend(query: torch.Tensor) -> str:
    """Resolve the backend for a given query tensor."""
    if _forced == "sdpa":
        return "sdpa"
    supported = query.is_cuda and query.dtype in (torch.bfloat16, torch.float16)
    if _forced == "triton":
        if not supported:
            raise RuntimeError("Triton local attention requires CUDA BF16/FP16 tensors.")
        return "triton"
    return "triton" if supported and triton_available(query.device) else "sdpa"


# ---------------------------------------------------------------------------
# Tiled SDPA local attention
# ---------------------------------------------------------------------------

_layouts: OrderedDict = OrderedDict()
_layout_bytes = 0


def clear_layout_cache() -> None:
    global _layout_bytes
    _layouts.clear()
    _layout_bytes = 0


def _layout(height, width, windows, shifted, chunk, device):
    """Per-tile gather indices and masks; cached per geometry."""
    global _layout_bytes
    cache_key = (height, width, windows, shifted, chunk, device.type, device.index)
    if cache_key in _layouts:
        _layouts.move_to_end(cache_key)
        return _layouts[cache_key][0]

    tile_y, tile_x = torch.meshgrid(
        torch.arange(0, height, _TILE, device=device),
        torch.arange(0, width, _TILE, device=device),
        indexing="ij",
    )
    tile_y, tile_x = tile_y.flatten(), tile_x.flatten()
    qr = torch.arange(_TILE * _TILE, device=device)
    n = height * width
    parts, storage_bytes = [], 0
    for start in range(0, tile_y.numel(), chunk):
        y, x = tile_y[start : start + chunk, None], tile_x[start : start + chunk, None]
        qy, qx = y + qr[None] // _TILE, x + qr[None] % _TILE
        qi = (qy * width + qx).clamp(0, n - 1).flatten()
        keys, masks = [], []
        for plane, (wh, ww) in enumerate(windows):
            rows, cols = wh + _TILE - 1, ww + _TILE - 1
            if shifted:
                rows, cols = min(height, rows), min(width, cols)
            kr = torch.arange(rows * cols, device=device)
            top, left = y - wh // 2, x - ww // 2
            qtop, qleft = qy - wh // 2, qx - ww // 2
            if shifted:
                top, left = top.clamp(0, height - wh), left.clamp(0, width - ww)
                qtop, qleft = qtop.clamp(0, height - wh), qleft.clamp(0, width - ww)
            ky, kx = top + kr[None] // cols, left + kr[None] % cols
            valid = (ky >= 0) & (ky < height) & (kx >= 0) & (kx < width)
            keys.append(plane * n + (ky * width + kx).clamp(0, n - 1))
            masks.append(
                valid[:, None]
                & (ky[:, None] >= qtop[:, :, None])
                & (ky[:, None] < qtop[:, :, None] + wh)
                & (kx[:, None] >= qleft[:, :, None])
                & (kx[:, None] < qleft[:, :, None] + ww)
            )
        ki = torch.cat(keys, -1)
        mask = torch.cat(masks, -1)[:, None].contiguous()
        owner = None
        if height % _TILE or width % _TILE:
            owner = ((qy < height) & (qx < width)).flatten().nonzero().flatten()
        part = (qi, ki.flatten(), mask, owner, int(qy.shape[0]), int(ki.shape[1]))
        parts.append(part)
        storage_bytes += sum(t.numel() * t.element_size() for t in part if isinstance(t, torch.Tensor))

    while _layouts and _layout_bytes + storage_bytes > _CACHE_BYTES:
        _, (_, old_bytes) = _layouts.popitem(last=False)
        _layout_bytes -= old_bytes
    if storage_bytes <= _CACHE_BYTES:
        _layouts[cache_key] = parts, storage_bytes
        _layout_bytes += storage_bytes
    return parts


def sdpa_local_attention(query, key, value, height, width, windows, *, shifted):
    """Local attention with SDPA.

    ``query``: [B, Hq, H*W, D] (queries of one frame), ``key``/``value``:
    [B, Hkv, planes*H*W, D]; ``windows`` is one (wh, ww) per KV plane.
    Returns [B, Hq, H*W, D].
    """
    b, heads, tokens, dim = query.shape
    kv_heads = key.shape[1]
    n = height * width
    if tokens != n or key.shape != value.shape or key.shape[0] != b:
        raise ValueError("Invalid local attention Q/K/V shape")
    if key.shape[2] != len(windows) * n or key.shape[3] != dim or heads % kv_heads:
        raise ValueError("Invalid local attention Q/K/V shape")
    windows = tuple((min(int(wh), height) if shifted else int(wh), min(int(ww), width) if shifted else int(ww)) for wh, ww in windows)
    lengths = [
        (min(height, wh + _TILE - 1) * min(width, ww + _TILE - 1)) if shifted else ((wh + _TILE - 1) * (ww + _TILE - 1))
        for wh, ww in windows
    ]
    bytes_per_tile = sum(lengths) * kv_heads * dim * key.element_size() * 2
    limit = max(1, min(128, _GATHER_BYTES // max(1, bytes_per_tile)))
    chunk = 1 << (limit.bit_length() - 1)
    parts = _layout(height, width, windows, shifted, chunk, query.device)
    output = query.new_empty((b, tokens, heads, dim))
    for batch in range(b):
        qs = query[batch].transpose(0, 1).contiguous()
        ks = key[batch].transpose(0, 1).contiguous()
        vs = value[batch].transpose(0, 1).contiguous()
        for qi, ki, mask, owner, count, key_length in parts:
            q = qs.index_select(0, qi).view(count, _TILE * _TILE, heads, dim).transpose(1, 2)
            k = ks.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            v = vs.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=heads != kv_heads)
            attended = attended.transpose(1, 2).reshape(-1, heads, dim)
            if owner is not None:
                attended = attended.index_select(0, owner)
                output[batch].index_copy_(0, qi.index_select(0, owner), attended)
            else:
                output[batch].index_copy_(0, qi, attended)
    return output.transpose(1, 2)


__all__ = [
    "clear_layout_cache",
    "force_backend",
    "forced_backend",
    "sdpa_local_attention",
    "select_backend",
    "triton_available",
]
