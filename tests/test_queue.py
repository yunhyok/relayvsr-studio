def test_queue_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAYVSR_HOME", str(tmp_path / "home"))
    from relayvsr_studio.app import settings

    assert settings.load_queue() == []  # no file yet
    paths = [r"D:\\영상\\a.mp4", "E:/clips/b.mkv"]
    settings.save_queue(paths)
    assert settings.load_queue() == paths
    assert not settings.queue_file().with_suffix(".tmp").exists()
    settings.queue_file().write_text("{broken", encoding="utf-8")
    assert settings.load_queue() == []  # a damaged file never blocks startup
    settings.save_queue([])
    assert settings.load_queue() == []
