"""Video / image I/O: prefetching decoder, asynchronous encoders, audio remux.

Decoding and encoding run on worker threads so the GPU never waits for FFmpeg. Output
is written to a temporary file next to the destination and renamed atomically after
the source audio has been muxed back in.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts", ".wmv", ".flv", ".mpg", ".mpeg", ".gif"}


@dataclass
class MediaInfo:
    path: str
    kind: str  # "video" | "image" | "image_dir"
    width: int
    height: int
    fps: Fraction | None
    frames: int  # best-effort count (0 if unknown)
    duration: float
    has_audio: bool
    codec: str = ""
    audio_codec: str = ""

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["fps"] = float(self.fps) if self.fps else None
        return data


def is_image_input(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES


def probe(path: str | Path, fps_override: str | float | None = None) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"No images found in {path}")
        with Image.open(files[0]) as image:
            width, height = image.size
        fps = Fraction(str(fps_override)) if fps_override else None
        return MediaInfo(str(path), "image_dir", width, height, fps, len(files), len(files) / float(fps) if fps else 0.0, False)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            width, height = image.size
        return MediaInfo(str(path), "image", width, height, None, 1, 0.0, False)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream.")
        stream = container.streams.video[0]
        fps = Fraction(str(fps_override)) if fps_override else (stream.average_rate or stream.guessed_rate or stream.base_rate)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if not duration and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        frames = int(stream.frames or 0)
        if not frames and fps and duration:
            frames = int(round(duration * float(fps)))
        audio = container.streams.audio[0] if container.streams.audio else None
        return MediaInfo(
            str(path),
            "video",
            stream.width,
            stream.height,
            Fraction(fps) if fps else None,
            frames,
            duration,
            audio is not None,
            stream.codec_context.name,
            audio.codec_context.name if audio is not None else "",
        )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _to_tensor(rgb: np.ndarray, pin: bool) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
    if pin:
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
    return tensor


def iter_frames(path: str | Path, *, start_frame: int = 0, max_frames: int | None = None, prefetch: int = 24, pin_memory: bool = True):
    """Yield uint8 CHW tensors; decoding happens on a background thread."""
    path = Path(path)
    if start_frame < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("Require start_frame >= 0 and max_frames >= 1.")
    sentinel = object()
    buffer: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop = threading.Event()
    error: list[BaseException] = []

    def images():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if path.is_dir() else [path]
        for file in files:
            with Image.open(file) as image:
                yield np.asarray(image.convert("RGB"))

    def video():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")

    def producer():
        count = 0
        try:
            source = images() if is_image_input(path) else video()
            for index, rgb in enumerate(source):
                if stop.is_set():
                    break
                if index < start_frame:
                    continue
                buffer.put(_to_tensor(rgb, pin_memory))
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            error.append(exc)
        finally:
            buffer.put(sentinel)

    thread = threading.Thread(target=producer, name="relayvsr-decode", daemon=True)
    thread.start()
    produced = 0
    try:
        while True:
            item = buffer.get()
            if item is sentinel:
                break
            produced += 1
            yield item
    finally:
        stop.set()
        # Drain so the producer can exit.
        while thread.is_alive():
            try:
                buffer.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.05)
    if error:
        raise error[0]
    if produced == 0:
        raise ValueError("No frames were decoded from the requested input range.")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@dataclass
class EncodeSettings:
    codec: str = "libx264"  # libx264 | libx265 | h264_nvenc | hevc_nvenc | av1_nvenc | png
    crf: int = 18
    preset: str = "fast"
    pix_fmt: str = "yuv420p"
    threads: int = 0  # 0 = auto
    ffmpeg_path: str | None = None  # external ffmpeg for NVENC (or when forced)
    use_external_ffmpeg: bool = False
    extra_args: tuple[str, ...] = ()


def find_ffmpeg(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffmpeg")


def pyav_encoder_available(name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:  # noqa: BLE001
        return False


def available_encoders(ffmpeg_path: str | None = None) -> dict[str, list[str]]:
    """Encoder names grouped by backend."""
    names = ["libx264", "libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc", "libsvtav1", "libaom-av1"]
    result = {"pyav": [n for n in names if pyav_encoder_available(n)], "ffmpeg": []}
    ffmpeg = find_ffmpeg(ffmpeg_path)
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15, creationflags=_no_window()).stdout
            result["ffmpeg"] = [n for n in names if f" {n} " in out]
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class _AsyncWriterBase:
    """GPU frame -> pinned host copy on a side stream -> encoder thread."""

    def __init__(self, queue_size: int = 12):
        self.pending: deque = deque()
        self.queue_size = queue_size
        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.frames = 0
        self.closed = False
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._loop, name="relayvsr-encode", daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            pixels, event = item
            try:
                if event is not None:
                    event.synchronize()
                self._write(pixels.numpy())
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                break

    def _write(self, rgb: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def submit(self, pixels: torch.Tensor) -> None:
        """``pixels``: uint8 HWC tensor (GPU or CPU)."""
        if self._error is not None:
            raise RuntimeError(f"Encoder failed: {self._error}") from self._error
        if pixels.is_cuda and self.copy_stream is not None:
            host = torch.empty(pixels.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
            self.copy_stream.wait_stream(torch.cuda.current_stream(pixels.device))
            with torch.cuda.stream(self.copy_stream):
                host.copy_(pixels, non_blocking=True)
                pixels.record_stream(self.copy_stream)
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            self._queue.put((host, event))
        else:
            self._queue.put((pixels.cpu().contiguous(), None))
        self.frames += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._queue.put(None)
        self._thread.join()
        self._finish()
        if self._error is not None:
            raise RuntimeError(f"Encoder failed: {self._error}") from self._error

    def _finish(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class PyAVVideoWriter(_AsyncWriterBase):
    def __init__(self, path: str | Path, fps: Fraction, settings: EncodeSettings, queue_size: int = 12):
        self.path = Path(path)
        self.fps = Fraction(fps)
        self.settings = settings
        self.container = None
        self.stream = None
        super().__init__(queue_size)

    def _open(self, height: int, width: int) -> None:
        self.container = av.open(str(self.path), mode="w")
        codec = self.settings.codec
        self.stream = self.container.add_stream(codec, rate=self.fps)
        self.stream.width, self.stream.height = width, height
        pix_fmt = self.settings.pix_fmt
        if pix_fmt == "yuv420p" and (height % 2 or width % 2):
            pix_fmt = "yuv444p"
        self.stream.pix_fmt = pix_fmt
        self.stream.time_base = 1 / self.fps
        options = {}
        threads = self.settings.threads or min(16, os.cpu_count() or 4)
        if codec in ("libx264", "libx265"):
            options.update({"crf": str(self.settings.crf), "preset": self.settings.preset})
            if codec == "libx265":
                options["x265-params"] = f"log-level=warning:pools={threads}"
                self.stream.codec_context.codec_tag = "hvc1"
            else:
                self.stream.codec_context.thread_count = threads
        elif codec.endswith("_nvenc"):
            options.update({"preset": "p5", "tune": "hq", "rc": "vbr", "cq": str(self.settings.crf)})
        self.stream.options = options

    def _write(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if self.container is None:
            self._open(h, w)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.frames_written
        frame.time_base = 1 / self.fps
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames_written += 1

    frames_written = 0

    def _finish(self) -> None:
        if self.container is None:
            return
        try:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()


class FfmpegPipeWriter(_AsyncWriterBase):
    """Pipe raw RGB frames to an external ffmpeg (hardware encoders, audio in one pass)."""

    def __init__(self, path: str | Path, fps: Fraction, settings: EncodeSettings, *, audio_source: str | None = None, queue_size: int = 12):
        self.path = Path(path)
        self.fps = Fraction(fps)
        self.settings = settings
        self.audio_source = audio_source
        self.process: subprocess.Popen | None = None
        self.ffmpeg = find_ffmpeg(settings.ffmpeg_path)
        if not self.ffmpeg:
            raise FileNotFoundError("ffmpeg executable not found for external encoding.")
        super().__init__(queue_size)

    def _open(self, height: int, width: int) -> None:
        s = self.settings
        pix_fmt = s.pix_fmt
        if pix_fmt == "yuv420p" and (height % 2 or width % 2):
            pix_fmt = "yuv444p"
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "pipe:0",
        ]
        if self.audio_source:
            cmd += ["-i", self.audio_source, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-shortest"]
        codec = s.codec
        cmd += ["-c:v", codec, "-pix_fmt", pix_fmt]
        if codec in ("libx264", "libx265"):
            cmd += ["-crf", str(s.crf), "-preset", s.preset]
            if codec == "libx265":
                cmd += ["-tag:v", "hvc1", "-x265-params", "log-level=warning"]
        elif codec.endswith("_nvenc"):
            cmd += ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(s.crf), "-b:v", "0"]
            if codec == "hevc_nvenc":
                cmd += ["-tag:v", "hvc1"]
        elif codec in ("libsvtav1", "libaom-av1"):
            cmd += ["-crf", str(s.crf)]
        cmd += list(s.extra_args)
        cmd += ["-movflags", "+faststart", str(self.path)]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_no_window())

    def _write(self, rgb: np.ndarray) -> None:
        if self.process is None:
            self._open(*rgb.shape[:2])
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(rgb.tobytes())

    def _finish(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        _, err = self.process.communicate()
        if self.process.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {self.process.returncode}: {err.decode(errors='replace')[-2000:]}")


class ImageSequenceWriter(_AsyncWriterBase):
    def __init__(self, folder: str | Path, pattern: str = "frame_{:06d}.png", queue_size: int = 12):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.pattern = pattern
        self.index = 0
        super().__init__(queue_size)

    def _write(self, rgb: np.ndarray) -> None:
        Image.fromarray(rgb).save(self.folder / self.pattern.format(self.index))
        self.index += 1

    def _finish(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Audio muxing
# ---------------------------------------------------------------------------


def mux_audio(source: str | Path, video: str | Path, destination: str | Path, *, ffmpeg_path: str | None = None, max_duration: float | None = None) -> bool:
    """Copy audio streams of ``source`` into ``video`` -> ``destination`` (stream copy).

    Returns True when audio was muxed, False when the output has no audio (the video
    file is then moved unchanged).
    """
    source, video, destination = Path(source), Path(video), Path(destination)
    try:
        with av.open(str(source)) as src:
            audio_streams = list(src.streams.audio)
            if not audio_streams:
                shutil.move(str(video), str(destination))
                return False
    except av.AVError:
        shutil.move(str(video), str(destination))
        return False
    try:
        _mux_with_pyav(source, video, destination, max_duration)
        return True
    except Exception as error:  # noqa: BLE001 - fall back to ffmpeg
        log.warning("PyAV audio remux failed (%s); trying ffmpeg.", error)
        if destination.exists():
            destination.unlink()
    ffmpeg = find_ffmpeg(ffmpeg_path)
    if ffmpeg:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c", "copy"]
        if max_duration:
            cmd += ["-t", f"{max_duration:.3f}"]
        cmd += ["-movflags", "+faststart", str(destination)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600, creationflags=_no_window())
            video.unlink(missing_ok=True)
            return True
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("ffmpeg audio remux failed (%s); writing video without audio.", error)
    shutil.move(str(video), str(destination))
    return False


def _add_stream_from_template(output, template):
    if hasattr(output, "add_stream_from_template"):
        return output.add_stream_from_template(template)
    return output.add_stream(template=template)


def _mux_with_pyav(source: Path, video: Path, destination: Path, max_duration: float | None) -> None:
    with av.open(str(video)) as vin, av.open(str(source)) as ain, av.open(str(destination), mode="w") as out:
        vstream = vin.streams.video[0]
        vout = _add_stream_from_template(out, vstream)
        astreams = list(ain.streams.audio)
        mapping = {s.index: _add_stream_from_template(out, s) for s in astreams}
        for packet in vin.demux(vstream):
            if packet.dts is None:
                continue
            packet.stream = vout
            out.mux(packet)
        for packet in ain.demux(astreams):
            if packet.dts is None:
                continue
            if max_duration is not None and packet.pts is not None and packet.time_base is not None:
                if float(packet.pts * packet.time_base) > max_duration + 0.5:
                    continue
            packet.stream = mapping[packet.stream.index]
            out.mux(packet)
    video.unlink(missing_ok=True)


def temporary_path(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=".relayvsr-", suffix=destination.suffix or ".mp4", dir=destination.parent, delete=False)
    handle.close()
    return Path(handle.name)


def save_image(rgb: torch.Tensor, path: str | Path) -> None:
    Image.fromarray(rgb.cpu().numpy()).save(path)


__all__ = [
    "EncodeSettings",
    "FfmpegPipeWriter",
    "IMAGE_SUFFIXES",
    "ImageSequenceWriter",
    "MediaInfo",
    "PyAVVideoWriter",
    "VIDEO_SUFFIXES",
    "available_encoders",
    "find_ffmpeg",
    "is_image_input",
    "iter_frames",
    "mux_audio",
    "probe",
    "pyav_encoder_available",
    "save_image",
    "temporary_path",
]
