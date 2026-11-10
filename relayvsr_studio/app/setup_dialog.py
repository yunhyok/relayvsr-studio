"""Environment / model setup dialog."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bootstrap import models as model_files
from ..bootstrap.envmgr import EnvironmentManager, EnvStatus
from .i18n import tr
from .settings import Settings


class _InstallThread(QThread):
    progress = Signal(dict)
    finished_status = Signal(object)
    failed = Signal(str)

    def __init__(self, envmgr: EnvironmentManager, parent=None, app_only: bool = False):
        super().__init__(parent)
        self.envmgr = envmgr
        self.app_only = app_only

    def run(self) -> None:
        try:
            status = self.envmgr.create(self.progress.emit, reinstall_app_only=self.app_only)
            self.finished_status.emit(status)
        except InterruptedError:
            self.failed.emit("cancelled")
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"{type(error).__name__}: {error}")


class _StatusThread(QThread):
    result = Signal(object)

    def __init__(self, envmgr: EnvironmentManager, parent=None):
        super().__init__(parent)
        self.envmgr = envmgr

    def run(self) -> None:
        self.result.emit(self.envmgr.status())


class _DownloadThread(QThread):
    progress = Signal(dict)
    done = Signal(int)
    failed = Signal(int, str)

    def __init__(self, models_dir: str, scale: int, parent=None):
        super().__init__(parent)
        self.models_dir, self.scale = models_dir, scale
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            model_files.download_models(self.models_dir, self.scale, self.progress.emit, lambda: self._cancel)
            self.done.emit(self.scale)
        except InterruptedError:
            self.failed.emit(self.scale, "cancelled")
        except Exception as error:  # noqa: BLE001
            self.failed.emit(self.scale, f"{type(error).__name__}: {error}")


def _gb(num: float) -> str:
    return f"{num / 1024**3:.1f} GB"


class SetupDialog(QDialog):
    environment_changed = Signal()
    models_changed = Signal()

    def __init__(self, settings: Settings, envmgr: EnvironmentManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings
        self.envmgr = envmgr
        self.setWindowTitle(tr("Environment setup"))
        self.resize(760, 640)
        self._install_thread: _InstallThread | None = None
        self._download_threads: dict[int, _DownloadThread] = {}
        self._build()
        self.refresh_status()
        self.refresh_models()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)

        env_box = QGroupBox(tr("Engine"))
        form = QFormLayout(env_box)
        self.env_status = QLabel(tr("Checking environment…"))
        self.env_status.setWordWrap(True)
        form.addRow(tr("Status"), self.env_status)
        self.env_dir_edit = QLineEdit(self.settings.env_dir)
        env_row = QHBoxLayout()
        env_row.addWidget(self.env_dir_edit)
        browse_env = QPushButton(tr("Browse…"))
        browse_env.clicked.connect(lambda: self._browse(self.env_dir_edit))
        env_row.addWidget(browse_env)
        form.addRow(tr("Engine folder"), env_row)
        self.disk_label = QLabel("")
        form.addRow(tr("Free disk space"), self.disk_label)
        buttons = QHBoxLayout()
        self.install_button = QPushButton(tr("Install / repair engine"))
        self.install_button.clicked.connect(lambda: self.install())
        buttons.addWidget(self.install_button)
        self.cancel_install_button = QPushButton(tr("Cancel"))
        self.cancel_install_button.setEnabled(False)
        self.cancel_install_button.clicked.connect(self.cancel_install)
        buttons.addWidget(self.cancel_install_button)
        self.remove_button = QPushButton(tr("Remove environment"))
        self.remove_button.clicked.connect(self.remove_environment)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        form.addRow(buttons)
        layout.addWidget(env_box)

        models_box = QGroupBox(tr("Models"))
        grid = QGridLayout(models_box)
        self.models_dir_edit = QLineEdit(self.settings.models_dir)
        browse_models = QPushButton(tr("Browse…"))
        browse_models.clicked.connect(lambda: self._browse(self.models_dir_edit, models=True))
        grid.addWidget(QLabel(tr("Models folder")), 0, 0)
        grid.addWidget(self.models_dir_edit, 0, 1, 1, 2)
        grid.addWidget(browse_models, 0, 3)
        self.model_rows: dict[int, dict] = {}
        for row, scale in enumerate((4, 2), start=1):
            spec = model_files.MODEL_FILES[scale]
            name = QLabel(f"{tr('4x (RelayVSR)') if scale == 4 else tr('2x (RelayVSR_X2)')}  <span style='color:#888'>{spec['repo']}</span>")
            name.setTextFormat(Qt.TextFormat.RichText)
            status = QLabel("")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(True)
            button = QPushButton(tr("Download"))
            button.clicked.connect(lambda _=False, s=scale: self.download(s))
            grid.addWidget(name, row, 0)
            grid.addWidget(status, row, 1)
            grid.addWidget(bar, row, 2)
            grid.addWidget(button, row, 3)
            self.model_rows[scale] = {"status": status, "bar": bar, "button": button}
        layout.addWidget(models_box)

        scratch_box = QGroupBox(tr("Scratch disk"))
        scratch_form = QFormLayout(scratch_box)
        self.scratch_dir_edit = QLineEdit(self.settings.scratch_dir)
        self.scratch_dir_edit.setPlaceholderText(tr("Not set: temporary files stay next to the output file"))
        self.scratch_dir_edit.setClearButtonEnabled(True)
        self.scratch_dir_edit.textChanged.connect(self._show_scratch_space)
        scratch_row = QHBoxLayout()
        scratch_row.addWidget(self.scratch_dir_edit)
        browse_scratch = QPushButton(tr("Browse…"))
        browse_scratch.clicked.connect(lambda: self._browse(self.scratch_dir_edit, title=tr("Scratch folder")))
        scratch_row.addWidget(browse_scratch)
        scratch_form.addRow(tr("Scratch folder"), scratch_row)
        self.scratch_space_label = QLabel("")
        scratch_form.addRow(tr("Free disk space"), self.scratch_space_label)
        scratch_hint = QLabel(
            tr(
                "Videos being encoded, preview images, temporary files, the Triton kernel cache and the engine "
                "download cache are written to a RelayVSR-scratch subfolder here. Finished videos are then moved "
                "to the output folder. Pick a fast drive with free space of at least the size of the output video."
            )
        )
        scratch_hint.setWordWrap(True)
        scratch_hint.setStyleSheet("color:#999")
        scratch_form.addRow(scratch_hint)
        layout.addWidget(scratch_box)
        self._show_scratch_space()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText(tr("Setup log"))
        layout.addWidget(self.log, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        apply_button = QPushButton(tr("Apply"))
        apply_button.clicked.connect(self.apply_paths)
        bottom.addWidget(apply_button)
        close_button = QPushButton(tr("Close"))
        close_button.clicked.connect(self.accept)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)

    def _browse(self, edit: QLineEdit, models: bool = False, title: str | None = None) -> None:
        folder = QFileDialog.getExistingDirectory(self, title or (tr("Models folder") if models else tr("Engine folder")), edit.text() or str(Path.home()))
        if folder:
            edit.setText(folder)

    def _show_scratch_space(self) -> None:
        text = self.scratch_dir_edit.text().strip()
        if not text:
            self.scratch_space_label.setText("—")
            return
        try:
            anchor = Path(text).anchor
            if not anchor:
                raise OSError("relative path")
            self.scratch_space_label.setText(f"{_gb(shutil.disk_usage(anchor).free)} ({anchor})")
        except OSError:
            self.scratch_space_label.setText("?")

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    # -------------------------------------------------------------- status
    def apply_paths(self) -> None:
        changed = False
        if self.settings.models_dir != self.models_dir_edit.text().strip():
            self.settings.models_dir = self.models_dir_edit.text().strip()
            changed = True
        if self.settings.env_dir != self.env_dir_edit.text().strip():
            self.settings.env_dir = self.env_dir_edit.text().strip()
            self.envmgr.env_dir = Path(self.settings.env_dir)
            changed = True
        scratch = self.scratch_dir_edit.text().strip()
        if self.settings.scratch_dir != scratch:
            if scratch:
                try:
                    if not Path(scratch).is_absolute():
                        raise OSError("not an absolute path")
                    probe = Path(scratch) / "RelayVSR-scratch"
                    probe.mkdir(parents=True, exist_ok=True)
                    test_file = probe / ".write-test"
                    test_file.write_bytes(b"")
                    test_file.unlink()
                except OSError as error:
                    QMessageBox.warning(self, tr("Scratch disk"), f"{tr('This folder cannot be used as a scratch folder.')}\n{scratch}\n{error}")
                    self.scratch_dir_edit.setText(self.settings.scratch_dir)
                    scratch = self.settings.scratch_dir
            if self.settings.scratch_dir != scratch:
                self.settings.scratch_dir = scratch
                self.envmgr.scratch_dir = Path(scratch) if scratch else None
                self.append_log(f"{tr('Scratch folder')}: {scratch or '—'}")
                changed = True
        if changed:
            self.settings.save()
            self.refresh_status()
            self.refresh_models()
            self.environment_changed.emit()
            self.models_changed.emit()

    def refresh_status(self) -> None:
        self.env_status.setText(tr("Checking environment…"))
        self._status_thread = _StatusThread(self.envmgr, self)
        self._status_thread.result.connect(self._show_status)
        self._status_thread.start()
        try:
            usage = shutil.disk_usage(Path(self.env_dir_edit.text() or Path.home()).anchor or str(Path.home()))
            self.disk_label.setText(_gb(usage.free))
        except OSError:
            self.disk_label.setText("?")

    def _show_status(self, status: EnvStatus) -> None:
        if status.mode == "dev":
            text = f"{tr('Engine ready')} (dev: {status.python})"
        elif status.ready:
            text = f"{tr('Engine ready')} — Python {status.details.get('python', '?')}, torch {status.torch} (CUDA {status.cuda}), Triton {status.triton or 'n/a'}"
        elif status.exists:
            text = f"{tr('Engine not installed')} — {status.error or 'incomplete'}"
            if status.torch and status.app_version and status.app_version != __import__('relayvsr_studio').__version__:
                text = f"Engine package {status.app_version} needs update"
        else:
            text = tr("Engine not installed")
        if status.gpu:
            text += f"\n{tr('GPU')}: {status.gpu}"
        self.env_status.setText(text)
        self.install_button.setEnabled(status.mode != "dev")
        self.remove_button.setEnabled(status.mode != "dev" and status.exists)

    def refresh_models(self) -> None:
        models_dir = self.models_dir_edit.text().strip() or self.settings.models_dir
        for scale, row in self.model_rows.items():
            present = model_files.models_present(models_dir, scale)
            missing = model_files.missing_bytes(models_dir, scale)
            if present:
                row["status"].setText(tr("Installed"))
                row["bar"].setValue(1000)
                row["bar"].setFormat("100%")
                row["button"].setEnabled(False)
            else:
                row["status"].setText(f"{tr('Not downloaded')} ({_gb(missing)})")
                row["button"].setEnabled(scale not in self._download_threads)

    # ------------------------------------------------------------- install
    def install(self, app_only: bool = False) -> None:
        """Install or repair the engine; ``app_only`` just refreshes the app's own engine package."""
        self.apply_paths()
        if self._install_thread is not None:
            return
        self.install_button.setEnabled(False)
        self.cancel_install_button.setEnabled(True)
        self.append_log("=== engine install ===")
        self._install_thread = _InstallThread(self.envmgr, self, app_only=bool(app_only))
        self._install_thread.progress.connect(self._install_progress)
        self._install_thread.finished_status.connect(self._install_done)
        self._install_thread.failed.connect(self._install_failed)
        self._install_thread.start()

    def cancel_install(self) -> None:
        self.envmgr.cancel()

    def _install_progress(self, event: dict) -> None:
        if event.get("type") == "step":
            self.append_log(f"--- {event.get('message')}")
            self.env_status.setText(event.get("message", ""))
        else:
            self.append_log(event.get("message", ""))

    def _install_done(self, status: EnvStatus) -> None:
        self._install_thread = None
        self.install_button.setEnabled(True)
        self.cancel_install_button.setEnabled(False)
        self._show_status(status)
        self.environment_changed.emit()

    def _install_failed(self, message: str) -> None:
        self._install_thread = None
        self.install_button.setEnabled(True)
        self.cancel_install_button.setEnabled(False)
        self.append_log(f"!!! {message}")
        self.env_status.setText(f"{tr('Failed')}: {message}")
        self.refresh_status()

    def remove_environment(self) -> None:
        if QMessageBox.question(self, tr("Remove environment"), f"{self.envmgr.env_dir}\n\n{tr('Remove environment')}?") != QMessageBox.StandardButton.Yes:
            return
        self.envmgr.remove()
        self.refresh_status()
        self.environment_changed.emit()

    # ------------------------------------------------------------ download
    def download(self, scale: int) -> None:
        self.apply_paths()
        if scale in self._download_threads:
            return
        row = self.model_rows[scale]
        row["button"].setEnabled(False)
        row["status"].setText(tr("Downloading…"))
        thread = _DownloadThread(self.settings.models_dir, scale, self)
        thread.progress.connect(lambda e, s=scale: self._download_progress(s, e))
        thread.done.connect(self._download_done)
        thread.failed.connect(self._download_failed)
        self._download_threads[scale] = thread
        thread.start()

    def _download_progress(self, scale: int, event: dict) -> None:
        row = self.model_rows[scale]
        kind = event.get("type")
        if kind == "progress":
            row["bar"].setValue(int(event["percent"] * 10))
            row["bar"].setFormat(f"{event['file']}  {event['done_mb']:.0f}/{event['total_mb']:.0f} MB")
        elif kind in ("start", "complete", "retry", "skip"):
            self.append_log(f"[{scale}x] {kind}: {event.get('file')} {event.get('error', '')}")

    def _download_done(self, scale: int) -> None:
        self._download_threads.pop(scale, None)
        self.refresh_models()
        self.models_changed.emit()

    def _download_failed(self, scale: int, message: str) -> None:
        self._download_threads.pop(scale, None)
        self.append_log(f"!!! [{scale}x] {message}")
        self.model_rows[scale]["status"].setText(f"{tr('Failed')}: {message[:60]}")
        self.model_rows[scale]["button"].setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        for thread in list(self._download_threads.values()):
            thread.cancel()
        super().closeEvent(event)

    @staticmethod
    def open_folder(path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        else:  # pragma: no cover
            subprocess.Popen(["xdg-open", str(path)])


__all__ = ["SetupDialog"]
