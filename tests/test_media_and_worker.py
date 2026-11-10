import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(12))
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_probe_and_iter(tiny_video):
    from relayvsr_studio.engine import media

    info = media.probe(tiny_video)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == Fraction(12)
    assert info.frames == 10
    frames = list(media.iter_frames(tiny_video, start_frame=2, max_frames=5))
    assert len(frames) == 5
    assert tuple(frames[0].shape) == (3, 48, 64)


def test_pyav_writer_roundtrip(tiny_video, tmp_path):
    import torch

    from relayvsr_studio.engine import media

    out = tmp_path / "out.mp4"
    writer = media.PyAVVideoWriter(out, Fraction(12), media.EncodeSettings(codec="libx264", crf=20, preset="ultrafast"))
    for i in range(6):
        writer.submit(torch.full((96, 128, 3), i * 30, dtype=torch.uint8))
    writer.close()
    info = media.probe(out)
    assert (info.width, info.height) == (128, 96)
    assert info.frames == 6


def test_worker_probe_roundtrip():
    import torch

    has_cuda = torch.cuda.is_available()
    env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "relayvsr_studio.engine.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, encoding="utf-8")
    try:
        process.stdin.write(json.dumps({"cmd": "probe", "id": 1}) + "\n")
        process.stdin.flush()
        seen = []
        for _ in range(200):
            line = process.stdout.readline()
            if not line:
                break
            obj = json.loads(line)
            seen.append(obj["type"])
            if obj["type"] in ("probe", "error"):
                if has_cuda:
                    assert obj["type"] == "probe", obj
                    assert "gpu" in obj and "encoders" in obj
                else:  # CPU-only machine: the worker must answer with a clear error, not crash
                    assert obj["type"] == "error" and obj["id"] == 1 and "No CUDA GPU" in obj["message"], obj
                break
        assert "hello" in seen and ("probe" if has_cuda else "error") in seen
        process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        process.stdin.flush()
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
