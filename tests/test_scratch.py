import os
import time

from relayvsr_studio import paths


def test_scratch_environment_creates_subfolders(tmp_path):
    env = paths.scratch_environment(tmp_path)
    root = tmp_path / paths.SCRATCH_FOLDER
    assert env[paths.SCRATCH_ENV] == str(tmp_path)
    assert env["TEMP"] == env["TMP"] == env["TMPDIR"] == str(root / "tmp")
    assert env["TRITON_CACHE_DIR"] == str(root / "triton-cache")
    assert (root / "tmp").is_dir() and (root / "triton-cache").is_dir()


def test_scratch_unset_or_unusable(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.SCRATCH_ENV, raising=False)
    assert paths.scratch_root(None) is None
    assert paths.scratch_subdir("encode") is None
    assert paths.scratch_environment(None) == {}
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    assert paths.scratch_subdir("encode", blocker) is None  # a file cannot hold subfolders
    assert paths.scratch_environment(blocker) == {}


def test_scratch_from_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.SCRATCH_ENV, str(tmp_path))
    assert paths.scratch_subdir("encode") == tmp_path / paths.SCRATCH_FOLDER / "encode"


def test_clean_scratch_only_removes_stale_entries(tmp_path):
    encode = paths.scratch_subdir("encode", tmp_path)
    previews = paths.scratch_subdir("previews", tmp_path)
    cache = paths.scratch_subdir("triton-cache", tmp_path)
    old_file, new_file = encode / "old.mp4", encode / "new.mp4"
    old_dir, kept = previews / "job1", cache / "kernel.bin"
    for file in (old_file, new_file, kept):
        file.write_bytes(b"x")
    old_dir.mkdir()
    (old_dir / "p.jpg").write_bytes(b"x")
    stale = time.time() - 3 * 86400
    for entry in (old_file, old_dir, kept):
        os.utime(entry, (stale, stale))
    assert paths.clean_scratch(tmp_path) == 2
    assert not old_file.exists() and not old_dir.exists()
    assert new_file.exists() and kept.exists()  # recent files and caches are left alone


def test_temporary_path_and_move(tmp_path, monkeypatch):
    from relayvsr_studio.engine import media

    monkeypatch.delenv(paths.SCRATCH_ENV, raising=False)
    scratch, out = tmp_path / "scratch", tmp_path / "out" / "video.mp4"
    beside = media.temporary_path(out)
    assert beside.parent == out.parent
    beside.unlink()
    temp = media.temporary_path(out, scratch)
    assert temp.parent == scratch / paths.SCRATCH_FOLDER / "encode" and temp.suffix == ".mp4"
    temp.write_bytes(b"payload")
    media.move_into_place(temp, out)
    assert out.read_bytes() == b"payload" and not temp.exists()


def test_move_into_place_copies_when_rename_fails(tmp_path, monkeypatch):
    from relayvsr_studio.engine import media

    source, destination = tmp_path / "a.mp4", tmp_path / "b.mp4"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    real_replace = os.replace

    def fake_replace(src, dst):
        if os.fspath(src) == os.fspath(source):
            raise OSError(18, "cross-device link")  # what a move between drives reports
        return real_replace(src, dst)

    monkeypatch.setattr(media.os, "replace", fake_replace)
    media.move_into_place(source, destination)
    assert destination.read_bytes() == b"new" and not source.exists()
    assert not (tmp_path / ".b.mp4.part").exists()
