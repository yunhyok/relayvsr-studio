"""JSON-lines worker process used by the GUI.

stdin  <- one JSON object per line: {"cmd": "probe" | "upscale" | "cancel" | "unload" | "shutdown", ...}
stdout -> one JSON object per line: {"type": ..., ...}; every reply to a command carries its "id".
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import traceback

from . import __doc__ as _doc  # noqa: F401 - keeps package import explicit

_write_lock = threading.Lock()


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class _EventLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            emit({"type": "log", "level": record.levelname.lower(), "logger": record.name, "message": self.format(record)})
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, handlers=[_EventLogHandler()], format="%(message)s")
    for noisy in ("av", "PIL", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    emit({"type": "hello", "pid": __import__("os").getpid(), "python": sys.version.split()[0]})

    import torch  # noqa: F401 - import early so failures surface immediately

    from .pipeline import Engine, JobCancelled, JobConfig

    engine = Engine()
    cancel_flag = threading.Event()
    busy = threading.Lock()

    def handle(command: dict) -> None:
        cmd = command.get("cmd")
        cid = command.get("id")
        if cmd == "probe":
            from . import media
            from .checkpoints import inspect_checkpoint

            info = engine.ensure_gpu()
            payload = {"type": "probe", "id": cid, "gpu": info.to_dict(), "encoders": media.available_encoders(command.get("ffmpeg_path"))}
            if command.get("models_dir"):
                from pathlib import Path

                models = {}
                for file in Path(command["models_dir"]).rglob("*.safetensors"):
                    try:
                        ck = inspect_checkpoint(file)
                        models[str(file)] = {"component": ck.component, "scale": ck.scale, "size": ck.size_bytes}
                    except (ValueError, OSError):
                        continue
                payload["models"] = models
            emit(payload)
        elif cmd == "media":
            from . import media

            info = media.probe(command["path"], command.get("fps"))
            emit({"type": "media", "id": cid, "media": info.to_dict()})
        elif cmd == "plan":
            from . import media

            cfg = JobConfig.from_dict(command["job"])
            info = media.probe(cfg.input, cfg.fps)
            plan, _, _ = engine.plan(cfg, info)
            emit({"type": "plan", "id": cid, "plan": plan.to_dict(), "media": info.to_dict()})
        elif cmd == "upscale":
            cfg = JobConfig.from_dict(command["job"])
            cancel_flag.clear()
            with busy:
                try:
                    result = engine.run(cfg, lambda e: emit({**e, "id": cid}), cancel_flag.is_set)
                    emit({"type": "finished", "id": cid, "ok": True, "result": result.to_dict()})
                except JobCancelled:
                    emit({"type": "finished", "id": cid, "ok": False, "cancelled": True})
                except Exception as error:  # noqa: BLE001
                    emit({"type": "finished", "id": cid, "ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                    if "out of memory" in str(error).lower():
                        engine.unload()
        elif cmd == "unload":
            engine.unload()
            emit({"type": "unloaded", "id": cid})
        else:
            emit({"type": "error", "id": cid, "message": f"Unknown command {cmd!r}"})

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as error:
                emit({"type": "error", "message": f"Bad JSON: {error}"})
                continue
            cmd = command.get("cmd")
            if cmd == "cancel":
                cancel_flag.set()
                emit({"type": "cancelling", "id": command.get("id")})
            elif cmd == "shutdown":
                cancel_flag.set()
                emit({"type": "bye"})
                sys.stdout.flush()
                import os

                os._exit(0)
            else:
                threading.Thread(target=_safe_handle, args=(command,), daemon=True).start()

    def _safe_handle(command: dict) -> None:
        try:
            handle(command)
        except Exception as error:  # noqa: BLE001
            emit({"type": "error", "id": command.get("id"), "message": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})

    reader()
    return 0


if __name__ == "__main__":
    sys.exit(main())
