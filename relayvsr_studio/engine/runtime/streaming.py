"""Bounded-lookahead collaboration between the Wan keyframe model and the Flash decoder.

This is the official RelayVSR streaming schedule (endpoint windows of ``window_size``
frames: every 15th frame is a Wan keyframe, the Flash decoder fills the frames in
between while looking ahead to the next keyframe), extended with:

* native 2x / 4x geometry,
* optional spatial tiling of the keyframe model with per-tile KV caches, so any input
  resolution fits a fixed VRAM budget (tiles may park their KV caches in host memory
  between keyframes),
* per-frame timing spans for diagnostics.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice

import torch
import torch.nn.functional as F

from ..geometry import RelayGeometry
from ..models.wan.sampling import prepare_wan_latents
from .timing import DeviceSpan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFrame:
    index: int
    rgb: torch.Tensor  # uint8 CHW, native LR
    source_time: float = 0.0
    arrival_time: float = 0.0


@dataclass(frozen=True)
class OutputFrame:
    index: int
    rgb: torch.Tensor  # device float BCHW in [-1, 1]
    source_time: float
    arrival_time: float
    window_start: int
    window_end: int
    timings: tuple[tuple[str, DeviceSpan], ...] = ()


@dataclass(frozen=True)
class TilingConfig:
    """Keyframe (Wan) tiling in Wan-token units. ``max_tokens`` <= 0 disables tiling."""

    max_tokens: int = 0
    overlap: int = 8
    kv_offload: bool = False  # park inactive tiles' KV caches in host memory


@dataclass(frozen=True)
class Tile:
    y0: int  # token coordinates (inclusive/exclusive)
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def plan_tiles(tokens_h: int, tokens_w: int, max_tokens: int, overlap: int) -> list[Tile]:
    """Split a tokens_h x tokens_w grid into the fewest overlapping tiles of <= max_tokens."""
    if max_tokens <= 0 or tokens_h * tokens_w <= max_tokens:
        return [Tile(0, tokens_h, 0, tokens_w)]
    overlap = max(0, int(overlap))
    n_h = n_w = 1

    def tile_dims(nh, nw):
        th = math.ceil((tokens_h + (nh - 1) * overlap) / nh)
        tw = math.ceil((tokens_w + (nw - 1) * overlap) / nw)
        return min(th, tokens_h), min(tw, tokens_w)

    th, tw = tile_dims(n_h, n_w)
    while th * tw > max_tokens:
        # Split the longer side first; stop when tiles would get smaller than two overlaps.
        if th >= tw and th > 2 * overlap + 2:
            n_h += 1
        elif tw > 2 * overlap + 2:
            n_w += 1
        elif th > 2 * overlap + 2:
            n_h += 1
        else:
            break
        th, tw = tile_dims(n_h, n_w)

    def starts(total, count, size):
        if count == 1:
            return [0]
        return [round(i * (total - size) / (count - 1)) for i in range(count)]

    tiles = []
    for y0 in starts(tokens_h, n_h, th):
        for x0 in starts(tokens_w, n_w, tw):
            tiles.append(Tile(y0, y0 + th, x0, x0 + tw))
    return tiles


def _axis_weights(start: int, end: int, total: int, overlap: int, device, dtype) -> torch.Tensor:
    size = end - start
    weights = torch.ones(size, device=device, dtype=dtype)
    ramp = max(1, overlap)
    if start > 0:
        n = min(ramp, size)
        weights[:n] = torch.linspace(1.0 / (n + 1), n / (n + 1), n, device=device, dtype=dtype)
    if end < total:
        n = min(ramp, size)
        weights[-n:] = torch.minimum(
            weights[-n:], torch.linspace(n / (n + 1), 1.0 / (n + 1), n, device=device, dtype=dtype)
        )
    return weights


def prepare_lr(frame: torch.Tensor, *, device, dtype, stride: int) -> torch.Tensor:
    """Normalize native RGB to [-1, 1] and pad (replicate) to the Wan token grid."""
    if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8:
        raise ValueError("Streaming input must be uint8 RGB [3,H,W].")
    h, w = frame.shape[-2:]
    if min(h, w) <= 0:
        raise ValueError("Frame dimensions must be positive.")
    x = frame.unsqueeze(0).to(device=device, non_blocking=True).to(dtype)
    x = x.div(127.5).sub(1.0)
    return F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")


@dataclass
class _TileState:
    tile: Tile
    wan_state: object  # WanVSRStreamingState
    host: dict = field(default_factory=dict)  # pinned host buffers for parked KV caches
    on_device: bool = True


class CollaborativeFrameStream:
    """Model-only generator shared by the CLI, the GUI worker and benchmarks."""

    def __init__(
        self,
        *,
        wan_model,
        flash_model,
        wan_pipeline,
        prompt_embeds,
        device,
        compute_dtype,
        geometry: RelayGeometry,
        window_size: int = 16,
        seed: int | None = 42,
        tiling: TilingConfig | None = None,
        reuse_endpoint_kv: bool = True,
        precompute_rope: bool = True,
    ):
        if window_size < 2:
            raise ValueError("Collaborative window_size must be at least 2.")
        if flash_model.condition_mode != "endpoints":
            raise ValueError("Collaborative streaming requires endpoint conditions.")
        if flash_model.training or wan_model.training:
            raise ValueError("Streaming models must be in eval() mode.")
        if flash_model.upscale_factor != geometry.scale:
            raise ValueError("Flash decoder scale does not match the geometry.")
        self.wan = wan_model
        self.flash = flash_model
        self.pipeline = wan_pipeline
        self.prompt = prompt_embeds
        self.device = torch.device(device)
        self.dtype = compute_dtype
        self.geometry = geometry
        self.window_size = int(window_size)
        self.seed = seed
        self.tiling = tiling or TilingConfig()
        self.reuse_endpoint_kv = reuse_endpoint_kv
        self.precompute_rope = precompute_rope
        self.last_keyframe_indices: deque[int] = deque(maxlen=10000)
        self.keyframe_count = 0
        self.tiles: list[Tile] = []
        self._prepared_geometry = None

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def prepare(self, height: int, width: int) -> None:
        ph, pw = self.geometry.padded_size(height, width)
        key = (ph, pw, self.window_size, self.dtype, self.device)
        if self.precompute_rope and self._prepared_geometry != key:
            self.flash.prepare_streaming(ph, pw, self.window_size)
            self._prepared_geometry = key

    def plan(self, height: int, width: int) -> list[Tile]:
        th, tw = self.geometry.wan_tokens(height, width)
        return plan_tiles(th, tw, self.tiling.max_tokens, self.tiling.overlap)

    # ------------------------------------------------------------ tile states
    def _park(self, state: _TileState) -> None:
        if not state.on_device:
            return
        for index, cache in enumerate(state.wan_state.model_state.block_caches):
            if cache._storage is None:
                continue
            k, v = cache._storage
            host = state.host.get(index)
            if host is None:
                host = tuple(self._host_like(t) for t in (k, v))
                state.host[index] = host
            host[0].copy_(k, non_blocking=True)
            host[1].copy_(v, non_blocking=True)
            cache._storage = host
            cache.key_value = None
        torch.cuda.current_stream(self.device).synchronize()
        state.on_device = False

    def _restore(self, state: _TileState) -> None:
        if state.on_device:
            return
        for cache in state.wan_state.model_state.block_caches:
            if cache._storage is None:
                continue
            hk, hv = cache._storage
            k = hk.to(self.device, non_blocking=True)
            v = hv.to(self.device, non_blocking=True)
            cache._storage = (k, v)
            valid = min(cache.next_frame, cache.window_size) * cache.tokens_per_frame
            cache.key_value = (k[:, :valid], v[:, :valid])
        torch.cuda.current_stream(self.device).synchronize()
        state.on_device = True

    @staticmethod
    def _host_like(tensor: torch.Tensor) -> torch.Tensor:
        try:
            return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")

    # --------------------------------------------------------------- keyframe
    def _wan_latent(self, native: torch.Tensor, tile_states: list[_TileState], generator, ph: int, pw: int) -> torch.Tensor:
        """Run the one-step Wan model over the (padded) LR frame, tiled if configured."""
        geometry = self.geometry
        latent_stride = geometry.latent_stride
        scale_factor = 16 if geometry.scale == 4 else 8

        def run(lr_tile: torch.Tensor, state: _TileState, latents: torch.Tensor | None):
            if geometry.wan_upsample != 1:
                lr_tile = F.interpolate(lr_tile, scale_factor=geometry.wan_upsample, mode="bilinear", align_corners=False)
            return self.pipeline.sample_streaming_latent_step(
                model=self.wan,
                lq_frame=lr_tile.unsqueeze(2),
                spatial_scale_factor=scale_factor,
                prompt_embeds=self.prompt,
                streaming_state=state.wan_state,
                generator=generator,
                device=self.device,
                compute_dtype=self.dtype,
                latents=latents,
            )[:, :, 0]

        if len(tile_states) == 1:
            return run(native, tile_states[0], None).to(self.dtype)

        lh, lw = ph // latent_stride, pw // latent_stride
        noise = prepare_wan_latents(
            model=self.wan,
            spatial_scale_factor=scale_factor,
            batch_size=1,
            height=ph * geometry.wan_upsample,
            width=pw * geometry.wan_upsample,
            device=self.device,
            generator=generator,
        )
        channels = noise.shape[1]
        acc = torch.zeros((1, channels, lh, lw), device=self.device, dtype=torch.float32)
        weight_sum = torch.zeros((1, 1, lh, lw), device=self.device, dtype=torch.float32)
        stride = geometry.wan_stride
        tokens_h, tokens_w = ph // stride, pw // stride
        cells = stride // latent_stride  # latent cells per token (2)
        for state in tile_states:
            tile = state.tile
            if self.tiling.kv_offload:
                self._restore(state)
            lr_tile = native[..., tile.y0 * stride : tile.y1 * stride, tile.x0 * stride : tile.x1 * stride]
            latents = noise[..., tile.y0 * cells : tile.y1 * cells, tile.x0 * cells : tile.x1 * cells]
            latent = run(lr_tile, state, latents.contiguous()).float()
            wy = _axis_weights(tile.y0, tile.y1, tokens_h, self.tiling.overlap, self.device, torch.float32)
            wx = _axis_weights(tile.x0, tile.x1, tokens_w, self.tiling.overlap, self.device, torch.float32)
            weight = (wy[:, None] * wx[None, :]).repeat_interleave(cells, 0).repeat_interleave(cells, 1)
            acc[..., tile.y0 * cells : tile.y1 * cells, tile.x0 * cells : tile.x1 * cells] += latent * weight
            weight_sum[..., tile.y0 * cells : tile.y1 * cells, tile.x0 * cells : tile.x1 * cells] += weight
            if self.tiling.kv_offload:
                self._park(state)
        return (acc / weight_sum.clamp_min(1e-6)).to(self.dtype)

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def frames(self, source: Iterable[InputFrame]) -> Iterator[OutputFrame]:
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Cannot infer an empty stream.")
        if first.index != 0:
            raise ValueError("A new stream must start at frame 0.")
        shape = tuple(first.rgb.shape)
        h, w = shape[-2:]
        geometry = self.geometry
        stride = geometry.wan_stride
        ph, pw = geometry.padded_size(h, w)
        scale = geometry.scale
        self.prepare(h, w)
        self.tiles = self.plan(h, w)
        tile_states = [_TileState(tile, self.pipeline.create_streaming_state(self.wan)) for tile in self.tiles]
        if len(self.tiles) > 1:
            log.info(
                "Keyframe model tiled into %d tiles of up to %dx%d tokens (overlap %d, kv_offload=%s).",
                len(self.tiles), *self.tiles[0].shape, self.tiling.overlap, self.tiling.kv_offload,
            )
        generator = None if self.seed is None else [torch.Generator(device=self.device).manual_seed(self.seed)]
        self.last_keyframe_indices.clear()
        self.keyframe_count = 0
        pending_timings: list[tuple[str, DeviceSpan]] = []

        def keyframe(packet: InputFrame):
            span = DeviceSpan(self.device)
            native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
            pending_timings.append(("wan_input", span.finish()))
            span = DeviceSpan(self.device)
            latent = self._wan_latent(native, tile_states, generator, ph, pw)
            pending_timings.append(("wan", span.finish()))
            self.last_keyframe_indices.append(packet.index)
            self.keyframe_count += 1
            span = DeviceSpan(self.device)
            prepared = self.flash.prepare_condition(latent) if self.reuse_endpoint_kv else None
            pending_timings.append(("flash_condition", span.finish()))
            return latent, prepared

        start_latent, start_prepared = keyframe(first)
        start = first
        first_window = True
        while True:
            window = [start, *islice(iterator, self.window_size - 1)]
            for offset, packet in enumerate(window):
                if packet.index != start.index + offset or tuple(packet.rgb.shape) != shape:
                    raise ValueError("Streaming frames must be consecutive with constant dimensions.")
            if len(window) == 1 and not first_window:
                return
            end = window[-1]
            if end.index == start.index:
                end_latent, end_prepared = start_latent, start_prepared
            else:
                end_latent, end_prepared = keyframe(end)
            count = max(2, len(window))
            span = DeviceSpan(self.device)
            state = (
                self.flash.begin_stream(start_prepared, end_prepared, total_frames=count, input_height=ph, input_width=pw)
                if self.reuse_endpoint_kv
                else None
            )
            pending_timings.append(("flash_condition", span.finish()))
            for offset, packet in enumerate(window):
                span = DeviceSpan(self.device)
                native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
                kwargs = {}
                if offset == 0 and not self.reuse_endpoint_kv:
                    kwargs = {"condition_latent": start_latent, "end_condition_latent": end_latent, "total_frames": count}
                pending_timings.append(("flash_input", span.finish()))
                span = DeviceSpan(self.device)
                prediction, state = self.flash.decode_step(native, state, **kwargs)
                pending_timings.append(("flash", span.finish()))
                if first_window or offset:
                    timings = tuple(pending_timings)
                    pending_timings.clear()
                    yield OutputFrame(
                        packet.index,
                        prediction[..., : h * scale, : w * scale],
                        packet.source_time,
                        packet.arrival_time,
                        start.index,
                        end.index,
                        timings,
                    )
            first_window = False
            start, start_latent, start_prepared = end, end_latent, end_prepared
            if len(window) < self.window_size:
                return


__all__ = [
    "CollaborativeFrameStream",
    "InputFrame",
    "OutputFrame",
    "Tile",
    "TilingConfig",
    "plan_tiles",
    "prepare_lr",
]
