"""Command line interface: ``relayvsr-studio-cli upscale|probe|download|gui``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .paths import default_models_dir


def _add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", "-i", required=True, help="Video file, image file or image directory")
    parser.add_argument("--output", "-o", required=True, help="Output file (.mp4/.mkv/.mov, image) or folder for --format png")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4, help="Native model: 2 (RelayVSR_X2) or 4 (RelayVSR)")
    parser.add_argument("--models-dir", default=str(default_models_dir()))
    parser.add_argument("--wan-checkpoint")
    parser.add_argument("--flash-checkpoint")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--wan-weights", choices=("auto", "bf16", "fp8"), default="auto")
    parser.add_argument("--block-offload", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--tiling", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--tile-max-tokens", type=int)
    parser.add_argument("--tile-overlap", type=int, default=8)
    parser.add_argument("--kv-window", type=int, help="Wan keyframe KV window (1-8; default 4 from the checkpoint)")
    parser.add_argument("--attention", choices=("auto", "triton", "sdpa"), default="auto", dest="attention_backend")
    parser.add_argument("--prefer", choices=("auto", "quality", "low_vram"), default="auto", dest="quality_preference")
    parser.add_argument("--window-size", type=int, default=16, help="Keyframe interval + 1 (default 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detail", type=float, default=1.0, help="Residual detail strength (0..1.5)")
    parser.add_argument("--output-scale", type=float, help="Final output scale relative to input (<= native scale)")
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--format", choices=("video", "png"), default="video", dest="output_format")
    parser.add_argument("--codec", default="libx264", help="libx264 | libx265 | h264_nvenc | hevc_nvenc | av1_nvenc")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--external-ffmpeg", action="store_true", dest="use_external_ffmpeg")
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--no-audio", action="store_false", dest="copy_audio")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", help="Override FPS; required for image directories")


def cmd_upscale(args: argparse.Namespace) -> int:
    from .engine.pipeline import Engine, JobCancelled, JobConfig

    data = {k: v for k, v in vars(args).items() if k not in ("command", "func", "verbose")}
    cfg = JobConfig.from_dict(data)
    state = {"last": 0.0}

    def emit(event: dict) -> None:
        kind = event.get("type")
        if kind == "progress":
            now = time.perf_counter()
            if now - state["last"] < 0.5 and event["frame"] != event.get("total"):
                return
            state["last"] = now
            total = event.get("total") or 0
            eta = event.get("eta")
            eta_text = f" ETA {eta:.0f}s" if eta is not None else ""
            bar = f"{event['frame']}/{total}" if total else str(event["frame"])
            sys.stderr.write(f"\r  {bar} frames  {event['fps']:.2f} fps  VRAM {event['vram_gib']:.1f} GiB (peak {event['vram_peak_gib']:.1f}){eta_text}      ")
            sys.stderr.flush()
        elif kind == "stage":
            sys.stderr.write(f"\n[{event['stage']}] {event.get('message', '')}\n")
        elif kind == "plan":
            p = event["plan"]
            sys.stderr.write(
                f"[plan] weights={p['wan_weights']} offload={p['block_offload']} kv_window={p['kv_window']} "
                f"tiles={p['tiles']} (max {p['tile_max_tokens']} tokens, kv_offload={p['tile_kv_offload']}) "
                f"estimate={p['estimate_gib']} GiB free={p['free_gib']} GiB -> {p['reason']}\n"
            )
        elif kind == "gpu":
            g = event["gpu"]
            sys.stderr.write(f"[gpu] {g['name']} ({g['architecture']}), {g['total_gib']} GiB, free {g['free_gib']} GiB, triton={g['triton']}\n")
        elif kind == "models":
            sys.stderr.write(f"[models] {event['details']} in {event['load_seconds']}s\n")
        elif kind == "done":
            r = event["result"]
            sys.stderr.write(f"\n[done] {r['frames']} frames -> {r['output']} in {r['seconds']}s ({r['fps']} fps), peak VRAM {r['peak_vram_gib']} GiB, audio={r['audio']}\n")
            sys.stderr.write(f"       timings(ms): {r['timings_ms']}\n")

    engine = Engine()
    try:
        engine.run(cfg, emit)
    except JobCancelled:
        return 130
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from .engine import media
    from .engine.gpu import gpu_info

    info = gpu_info()
    payload = {"version": __version__, "gpu": info.to_dict(), "encoders": media.available_encoders(args.ffmpeg_path)}
    if args.input:
        payload["media"] = media.probe(args.input).to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .bootstrap.models import download_models

    def progress(event: dict) -> None:
        if event.get("type") == "progress":
            sys.stderr.write(f"\r  {event['file']}: {event['done_mb']:.0f}/{event['total_mb']:.0f} MB ({event['percent']:.1f}%)      ")
        else:
            sys.stderr.write(f"\n{event}\n")

    download_models(args.models_dir, args.scale, progress)
    sys.stderr.write("\n")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:  # noqa: ARG001
    from .app.main import main as gui_main

    return gui_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relayvsr-studio", description=f"RelayVSR Studio {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upscale", help="Upscale a video, image or image sequence")
    _add_job_arguments(p_up)
    p_up.set_defaults(func=cmd_upscale)

    p_probe = sub.add_parser("probe", help="Print GPU, encoder and media information as JSON")
    p_probe.add_argument("--input")
    p_probe.add_argument("--ffmpeg-path")
    p_probe.set_defaults(func=cmd_probe)

    p_dl = sub.add_parser("download", help="Download RelayVSR checkpoints from Hugging Face")
    p_dl.add_argument("--scale", type=int, choices=(2, 4), default=4)
    p_dl.add_argument("--models-dir", default=str(default_models_dir()))
    p_dl.set_defaults(func=cmd_download)

    p_gui = sub.add_parser("gui", help="Launch the desktop application")
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("av", "PIL", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
