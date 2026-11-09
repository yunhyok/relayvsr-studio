"""High-level upscaling jobs: probe -> plan -> load -> stream -> encode -> mux."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F

from . import backends, media
from .checkpoints import DTYPES, LoadOptions, LoadedModels, find_checkpoints, load_models
from .geometry import RelayGeometry
from .gpu import MemoryPlan, apply_runtime_tuning, gpu_info, plan_memory
from .models.wan.sampling import WanVSRPipeline
from .runtime.streaming import CollaborativeFrameStream, InputFrame, TilingConfig

log = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]
CancelCheck = Callable[[], bool]


class JobCancelled(RuntimeError):
    pass


@dataclass
class JobConfig:
    input: str
    output: str
    scale: int = 4  # native model: 2 or 4
    models_dir: str = ""
    wan_checkpoint: str | None = None
    flash_checkpoint: str | None = None

    # Model execution
    device: str = "cuda"
    dtype: str = "bf16"
    wan_weights: str = "auto"  # auto | bf16 | fp8
    block_offload: str = "auto"  # auto | on | off
    tiling: str = "auto"  # auto | on | off
    tile_max_tokens: int | None = None
    tile_overlap: int = 8
    kv_window: int | None = None
    attention_backend: str = "auto"  # auto | triton | sdpa
    quality_preference: str = "auto"  # auto | quality | low_vram
    window_size: int = 16  # keyframe interval + 1
    seed: int = 42
    detail: float = 1.0

    # Output
    output_scale: float | None = None  # final size = input * output_scale (<= native scale)
    output_width: int | None = None
    output_height: int | None = None
    fit_short_side: int | None = None  # e.g. 1080 -> scale so the short side becomes 1080 (clamped to [1, native])
    output_format: str = "video"  # video | png
    codec: str = "libx264"
    crf: int = 18
    preset: str = "fast"
    pix_fmt: str = "yuv420p"
    encoder_threads: int = 0
    use_external_ffmpeg: bool = False
    ffmpeg_path: str | None = None
    copy_audio: bool = True
    overwrite: bool = False

    # Range
    start_frame: int = 0
    max_frames: int | None = None
    fps: str | None = None

    # Reporting
    preview_every: int = 0
    preview_dir: str | None = None
    progress_interval: float = 0.25

    @classmethod
    def from_dict(cls, data: dict) -> JobConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobResult:
    output: str
    frames: int
    seconds: float
    fps: float
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    plan: dict
    audio: bool
    peak_vram_gib: float
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_checkpoints(cfg: JobConfig) -> tuple[Path, Path]:
    if cfg.wan_checkpoint and cfg.flash_checkpoint:
        return Path(cfg.wan_checkpoint), Path(cfg.flash_checkpoint)
    if not cfg.models_dir:
        raise ValueError("models_dir (or explicit checkpoints) is required.")
    return find_checkpoints(cfg.models_dir, cfg.scale)


class Engine:
    """Keeps models resident between jobs when their load options are unchanged."""

    def __init__(self):
        self.models: LoadedModels | None = None
        self._key = None
        self.info = None

    def ensure_gpu(self, emit: EventCallback | None = None):
        if self.info is None:
            self.info = gpu_info()
            if not self.info.available:
                raise RuntimeError("No CUDA GPU detected. RelayVSR Studio requires an NVIDIA GPU with CUDA.")
            tuning = apply_runtime_tuning(self.info)
            if emit:
                emit({"type": "gpu", "gpu": self.info.to_dict(), "tuning": tuning})
        # Refresh free memory every time (NVML-based: counts other processes on Windows).
        from .gpu import query_memory

        free, total = query_memory(self.info.index)
        self.info.free_bytes, self.info.total_bytes = free, total
        return self.info

    def unload(self):
        if self.models is not None:
            self.models = None
            self._key = None
            backends.clear_layout_cache()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def plan(self, cfg: JobConfig, info_media: media.MediaInfo) -> tuple[MemoryPlan, Path, Path]:
        info = self.ensure_gpu()
        wan_path, flash_path = _resolve_checkpoints(cfg)
        geometry = RelayGeometry(cfg.scale)
        free = info.free_bytes
        # If models with the same options are already loaded, their memory is already
        # accounted for in ``free``; add it back so the planner reasons about totals.
        if self.models is not None:
            free += int(torch.cuda.memory_reserved(info.index))
        plan = plan_memory(
            info,
            geometry,
            info_media.height,
            info_media.width,
            prefer=cfg.quality_preference,
            wan_weights=cfg.wan_weights,
            block_offload={"auto": "auto", "on": True, "off": False}[cfg.block_offload],
            tiling=cfg.tiling,
            tile_max_tokens=cfg.tile_max_tokens,
            kv_window=cfg.kv_window,
            free_bytes=free,
        )
        return plan, wan_path, flash_path

    def load(self, cfg: JobConfig, plan: MemoryPlan, wan_path: Path, flash_path: Path, emit: EventCallback) -> LoadedModels:
        options = LoadOptions(
            device=cfg.device,
            dtype=cfg.dtype,
            wan_weights=plan.wan_weights,
            block_offload=plan.block_offload,
            attention_backend=cfg.attention_backend,
            kv_window=cfg.kv_window,
        )
        key = (str(wan_path), str(flash_path), options)
        if self.models is not None and self._key == key:
            return self.models
        self.unload()
        emit({"type": "stage", "stage": "load", "message": f"Loading RelayVSR {cfg.scale}x ({plan.wan_weights}{', block offload' if plan.block_offload else ''})"})
        self.models = load_models(wan_path, flash_path, options)
        self._key = key
        emit({"type": "models", "details": self.models.details, "load_seconds": round(self.models.load_seconds, 2)})
        return self.models

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run(self, cfg: JobConfig, emit: EventCallback, cancelled: CancelCheck | None = None) -> JobResult:
        cancelled = cancelled or (lambda: False)
        started = time.perf_counter()
        info_media = media.probe(cfg.input, cfg.fps)
        if info_media.kind != "image" and info_media.fps is None:
            raise ValueError("Image sequences require an explicit frame rate (fps).")
        emit({"type": "stage", "stage": "probe", "media": info_media.to_dict()})
        if cfg.output_format == "video" and info_media.kind != "image":
            output_path = Path(cfg.output)
            if output_path.suffix.lower() not in (".mp4", ".mkv", ".mov"):
                output_path = output_path.with_suffix(".mp4")
        else:
            output_path = Path(cfg.output)
        if output_path.exists() and not cfg.overwrite and output_path.is_file():
            raise FileExistsError(f"Output exists: {output_path}")
        if Path(cfg.input).resolve() == output_path.resolve():
            raise ValueError("Input and output must differ.")

        plan, wan_path, flash_path = self.plan(cfg, info_media)
        emit({"type": "plan", "plan": plan.to_dict()})
        models = self.load(cfg, plan, wan_path, flash_path, emit)
        geometry = models.geometry
        device, dtype = models.device, models.dtype
        models.flash.detail = float(cfg.detail)

        in_h, in_w = info_media.height, info_media.width
        native_h, native_w = geometry.output_size(in_h, in_w)
        out_h, out_w = native_h, native_w
        if cfg.output_width or cfg.output_height:
            if cfg.output_width and cfg.output_height:
                out_w, out_h = cfg.output_width, cfg.output_height
            elif cfg.output_width:
                out_w = cfg.output_width
                out_h = round(native_h * out_w / native_w)
            else:
                out_h = cfg.output_height
                out_w = round(native_w * out_h / native_h)
        elif cfg.fit_short_side:
            s = min(float(geometry.scale), max(1.0, cfg.fit_short_side / float(min(in_h, in_w))))
            out_h, out_w = round(in_h * s), round(in_w * s)
        elif cfg.output_scale and abs(cfg.output_scale - geometry.scale) > 1e-6:
            out_h, out_w = round(in_h * cfg.output_scale), round(in_w * cfg.output_scale)
        if cfg.output_format == "video":
            out_h, out_w = max(2, out_h // 2 * 2), max(2, out_w // 2 * 2)
        resize = (out_h, out_w) != (native_h, native_w)

        sampler = WanVSRPipeline(
            scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
            sampling={
                "execution_mode": "streaming",
                "num_inference_steps": 1,
                "timesteps": [1000],
                "guidance_scale": 1.0,
                "seed": cfg.seed,
                "kv_cache_window_size": models.wan.temporal_context_frames,
            },
        )
        sampler.prepare_streaming(model=models.wan, prompt_embeds=models.wan.prompt_embeds, device=device, compute_dtype=dtype)
        tiling = TilingConfig(max_tokens=plan.tile_max_tokens, overlap=cfg.tile_overlap, kv_offload=plan.tile_kv_offload)
        stream = CollaborativeFrameStream(
            wan_model=models.wan,
            flash_model=models.flash,
            wan_pipeline=sampler,
            prompt_embeds=models.wan.prompt_embeds,
            device=device,
            compute_dtype=dtype,
            geometry=geometry,
            window_size=cfg.window_size,
            seed=cfg.seed,
            tiling=tiling,
        )
        total = info_media.frames
        if cfg.start_frame:
            total = max(0, total - cfg.start_frame)
        if cfg.max_frames:
            total = min(total, cfg.max_frames) if total else cfg.max_frames
        fps = info_media.fps or Fraction(30)
        emit({"type": "stage", "stage": "run", "total": total, "input_size": [in_w, in_h], "output_size": [out_w, out_h], "tiles": len(stream.plan(in_h, in_w))})

        torch.cuda.reset_peak_memory_stats(device)
        writer = None
        temp_path: Path | None = None
        wrote_audio = False
        frames_done = 0
        last_report = 0.0
        run_started = time.perf_counter()
        timing_totals: dict[str, float] = {}
        timing_spans = []
        preview_dir = Path(cfg.preview_dir) if cfg.preview_dir else None
        encode = media.EncodeSettings(
            codec=cfg.codec, crf=cfg.crf, preset=cfg.preset, pix_fmt=cfg.pix_fmt, threads=cfg.encoder_threads,
            ffmpeg_path=cfg.ffmpeg_path, use_external_ffmpeg=cfg.use_external_ffmpeg,
        )
        external_audio_inline = False
        frames = media.iter_frames(cfg.input, start_frame=cfg.start_frame, max_frames=cfg.max_frames)
        recent_inputs: dict[int, torch.Tensor] = {}

        def _packets():
            for i, rgb in enumerate(frames):
                if preview_dir is not None and cfg.preview_every:
                    recent_inputs[i] = rgb
                    for old in [k for k in recent_inputs if k < i - cfg.window_size - 2]:
                        recent_inputs.pop(old, None)
                yield InputFrame(i, rgb, source_time=i / float(fps))

        packets = _packets()
        single_image = info_media.kind == "image"
        try:
            if single_image:
                writer = None
            elif cfg.output_format == "png":
                writer = media.ImageSequenceWriter(output_path)
            else:
                temp_path = media.temporary_path(output_path)
                use_ffmpeg = cfg.use_external_ffmpeg or (cfg.codec.endswith("_nvenc") and not media.pyav_encoder_available(cfg.codec))
                if use_ffmpeg and media.find_ffmpeg(cfg.ffmpeg_path):
                    audio_src = cfg.input if (cfg.copy_audio and info_media.has_audio and not cfg.max_frames and not cfg.start_frame) else None
                    writer = media.FfmpegPipeWriter(temp_path, fps, encode, audio_source=audio_src)
                    external_audio_inline = audio_src is not None
                else:
                    if use_ffmpeg:
                        log.warning("External ffmpeg not found; falling back to the built-in encoder.")
                        if cfg.codec.endswith("_nvenc"):
                            encode = dataclasses.replace(encode, codec="libx264")
                    writer = media.PyAVVideoWriter(temp_path, fps, encode)

            for result in stream.frames(packets):
                if cancelled():
                    raise JobCancelled("Cancelled by user.")
                rgb = result.rgb
                if resize:
                    rgb = F.interpolate(rgb.float().clamp(-1, 1), size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
                pixels = rgb[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
                if single_image:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    media.save_image(pixels, output_path)
                else:
                    writer.submit(pixels)
                frames_done += 1
                timing_spans.extend(result.timings)
                now = time.perf_counter()
                if preview_dir is not None and cfg.preview_every and (frames_done - 1) % cfg.preview_every == 0:
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = preview_dir / f"preview_{frames_done:06d}.jpg"
                    small = pixels
                    if small.shape[1] > 1600:
                        factor = 1600 / small.shape[1]
                        small = F.interpolate(rgb.float().clamp(-1, 1), scale_factor=factor, mode="bilinear", antialias=True)[0]
                        small = small.add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
                    media.save_image(small, preview_path)
                    input_path = None
                    source = recent_inputs.get(result.index)
                    if source is not None:
                        input_path = preview_dir / f"input_{frames_done:06d}.jpg"
                        media.save_image(source.permute(1, 2, 0).contiguous(), input_path)
                    emit({"type": "preview", "path": str(preview_path), "input_path": str(input_path) if input_path else None, "frame": frames_done, "index": result.index})
                if now - last_report >= cfg.progress_interval or frames_done == total:
                    last_report = now
                    elapsed = now - run_started
                    rate = frames_done / elapsed if elapsed > 0 else 0.0
                    eta = (total - frames_done) / rate if rate > 0 and total else None
                    emit({
                        "type": "progress",
                        "frame": frames_done,
                        "total": total,
                        "fps": round(rate, 2),
                        "elapsed": round(elapsed, 1),
                        "eta": round(eta, 1) if eta is not None else None,
                        "vram_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
                        "vram_peak_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
                        "keyframes": stream.keyframe_count,
                    })
            if writer is not None:
                emit({"type": "stage", "stage": "encode", "message": "Finishing encode"})
                writer.close()
                writer = None
            if temp_path is not None:
                if external_audio_inline:
                    wrote_audio = True
                    os.replace(temp_path, output_path)
                elif cfg.copy_audio and info_media.has_audio:
                    emit({"type": "stage", "stage": "mux", "message": "Copying audio"})
                    max_duration = frames_done / float(fps) if (cfg.max_frames or cfg.start_frame) else None
                    if cfg.start_frame:
                        max_duration = None  # audio offset would be wrong; skip audio for trimmed starts
                        wrote_audio = False
                        os.replace(temp_path, output_path)
                    else:
                        wrote_audio = media.mux_audio(cfg.input, temp_path, output_path, ffmpeg_path=cfg.ffmpeg_path, max_duration=max_duration)
                else:
                    os.replace(temp_path, output_path)
                temp_path = None
        finally:
            try:
                packets.close()
                frames.close()
            except Exception:  # noqa: BLE001
                pass
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        for name, span in timing_spans:
            try:
                timing_totals[name] = timing_totals.get(name, 0.0) + span.milliseconds()
            except RuntimeError:
                pass
        seconds = time.perf_counter() - started
        run_seconds = time.perf_counter() - run_started
        result = JobResult(
            output=str(output_path),
            frames=frames_done,
            seconds=round(seconds, 2),
            fps=round(frames_done / run_seconds, 2) if run_seconds > 0 else 0.0,
            input_size=(in_w, in_h),
            output_size=(out_w, out_h),
            plan=plan.to_dict(),
            audio=wrote_audio,
            peak_vram_gib=round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
            timings_ms={k: round(v, 1) for k, v in timing_totals.items()},
        )
        emit({"type": "done", "result": result.to_dict()})
        return result


__all__ = ["Engine", "JobCancelled", "JobConfig", "JobResult"]
