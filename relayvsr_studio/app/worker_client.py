"""QProcess wrapper around the JSON-lines engine worker."""

from __future__ import annotations

import itertools
import json
import logging
import os

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..bootstrap.envmgr import EnvironmentManager

log = logging.getLogger(__name__)


class WorkerClient(QObject):
    event = Signal(dict)  # every JSON object from the worker
    log_line = Signal(str)  # raw stderr / unparsable stdout
    started = Signal()
    stopped = Signal(int)  # exit code (or -1 on crash)

    def __init__(self, envmgr: EnvironmentManager, parent: QObject | None = None):
        super().__init__(parent)
        self.envmgr = envmgr
        self.process: QProcess | None = None
        self._buffer = b""
        self._ids = itertools.count(1)
        self._expect_exit = False

    # ------------------------------------------------------------ lifecycle
    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.is_running():
            return
        cmd, env = self.envmgr.worker_command()
        process = QProcess(self)
        qenv = QProcessEnvironment()
        for key, value in env.items():
            qenv.insert(key, value)
        process.setProcessEnvironment(qenv)
        process.setProgram(cmd[0])
        process.setArguments(cmd[1:])
        process.setWorkingDirectory(os.path.expanduser("~"))
        process.readyReadStandardOutput.connect(self._on_stdout)
        process.readyReadStandardError.connect(self._on_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._buffer = b""
        self._expect_exit = False
        self.process = process
        process.start()
        if process.waitForStarted(15000):
            self.started.emit()
        else:
            self.log_line.emit(f"Failed to start worker: {process.errorString()}")

    def stop(self, timeout_ms: int = 5000) -> None:
        if not self.is_running():
            return
        self._expect_exit = True
        self.send({"cmd": "shutdown"})
        assert self.process is not None
        if not self.process.waitForFinished(timeout_ms):
            self.process.kill()
            self.process.waitForFinished(3000)

    def kill(self) -> None:
        if self.is_running():
            self._expect_exit = True
            assert self.process is not None
            self.process.kill()
            self.process.waitForFinished(3000)

    # ------------------------------------------------------------- commands
    def send(self, command: dict) -> int:
        if not self.is_running():
            self.start()
        cid = command.get("id") or next(self._ids)
        command["id"] = cid
        data = (json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8")
        assert self.process is not None
        self.process.write(data)
        return cid

    def probe(self, models_dir: str | None = None, ffmpeg_path: str | None = None) -> int:
        return self.send({"cmd": "probe", "models_dir": models_dir, "ffmpeg_path": ffmpeg_path})

    def upscale(self, job: dict) -> int:
        return self.send({"cmd": "upscale", "job": job})

    def plan(self, job: dict) -> int:
        return self.send({"cmd": "plan", "job": job})

    def cancel(self) -> None:
        if self.is_running():
            self.send({"cmd": "cancel"})
            QTimer.singleShot(20000, self._kill_if_still_busy)

    def unload(self) -> int:
        return self.send({"cmd": "unload"})

    _busy_after_cancel = False

    def _kill_if_still_busy(self) -> None:
        if self._busy_after_cancel and self.is_running():
            self.log_line.emit("Worker did not stop after cancel; killing it.")
            self.kill()

    # ------------------------------------------------------------------ IO
    def _on_stdout(self) -> None:
        assert self.process is not None
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if text.startswith("{"):
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    self.log_line.emit(text)
                    continue
                self.event.emit(obj)
            else:
                self.log_line.emit(text)

    def _on_stderr(self) -> None:
        assert self.process is not None
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log_line.emit(line.rstrip())

    def _on_finished(self, code: int, status) -> None:  # noqa: ANN001
        crashed = status == QProcess.ExitStatus.CrashExit
        self.stopped.emit(-1 if crashed else code)
        if not self._expect_exit:
            self.log_line.emit(f"Worker exited (code={code}, crashed={crashed}).")
        self.process = None

    def _on_error(self, error) -> None:  # noqa: ANN001
        if self.process is not None:
            self.log_line.emit(f"Worker process error: {self.process.errorString()}")


__all__ = ["WorkerClient"]
