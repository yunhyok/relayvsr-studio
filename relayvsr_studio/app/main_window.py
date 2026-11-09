"""Main window: job queue, output settings, progress, previews and log."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..bootstrap import models as model_files
from ..bootstrap.envmgr import EnvironmentManager, EnvStatus
from ..paths import app_home, logs_dir
from .i18n import tr
from .settings import Settings
from .setup_dialog import SetupDialog
from .worker_client import WorkerClient

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts", ".wmv", ".flv", ".mpg", ".mpeg", ".gif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

CODECS = [
    ("libx264", "H.264 (libx264, CPU)"),
    ("libx265", "HEVC (libx265, CPU)"),
    ("h264_nvenc", "H.264 (NVENC, GPU)"),
    ("hevc_nvenc", "HEVC (NVENC, GPU)"),
    ("av1_nvenc", "AV1 (NVENC, RTX 40/50)"),
]
PRESETS_X26X = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"]


@dataclass
class Job:
    input: Path
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    output: Path | None = None
    status: str = "waiting"  # waiting | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    command_id: int | None = None
    scale: int = 4


class _StatusThread(QThread):
    result = Signal(object)

    def __init__(self, envmgr: EnvironmentManager, parent=None):
        super().__init__(parent)
        self.envmgr = envmgr

    def run(self) -> None:
        self.result.emit(self.envmgr.status())


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, envmgr: EnvironmentManager):
        super().__init__()
        self.settings = settings
        self.envmgr = envmgr
        self.client = WorkerClient(envmgr, self)
        self.client.event.connect(self.on_event)
        self.client.log_line.connect(self.append_log)
        self.client.stopped.connect(self.on_worker_stopped)
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self.env_status: EnvStatus | None = None
        self.gpu: dict | None = None
        self.encoders: dict | None = None
        self.pause_requested = False
        self._run_started = 0.0
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1280, 860)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self._load_settings_into_ui()
        self.statusBar().showMessage(tr("Checking environment…"))
        QTimer.singleShot(200, self.check_environment)

    # ------------------------------------------------------------------ UI
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        add_files = QAction(tr("Add files…"), self)
        add_files.setShortcut("Ctrl+O")
        add_files.triggered.connect(self.add_files)
        file_menu.addAction(add_files)
        add_folder = QAction(tr("Add folder…"), self)
        add_folder.triggered.connect(self.add_folder)
        file_menu.addAction(add_folder)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        setup_action = QAction(tr("Environment setup"), self)
        setup_action.triggered.connect(self.open_setup)
        tools_menu.addAction(setup_action)
        models_action = QAction(tr("Models folder"), self)
        models_action.triggered.connect(lambda: SetupDialog.open_folder(self.settings.models_path()))
        tools_menu.addAction(models_action)
        logs_action = QAction(tr("Log"), self)
        logs_action.triggered.connect(lambda: SetupDialog.open_folder(logs_dir()))
        tools_menu.addAction(logs_action)
        tools_menu.addSeparator()
        lang_menu = tools_menu.addMenu(tr("Language"))
        for code, label in (("auto", "Auto"), ("en", "English"), ("ko", "한국어")):
            action = QAction(label, self)
            action.triggered.connect(lambda _=False, c=code: self.set_language(c))
            lang_menu.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")
        page = QAction("RelayVSR project page", self)
        page.triggered.connect(lambda: webbrowser.open("https://kopperx.github.io/RelayVSR/"))
        help_menu.addAction(page)
        repo = QAction("RelayVSR Studio on GitHub", self)
        repo.triggered.connect(lambda: webbrowser.open("https://github.com/yunhyok/relayvsr-studio"))
        help_menu.addAction(repo)
        about = QAction(tr("About"), self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)

        header = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        header.addWidget(title)
        subtitle = QLabel("RelayVSR · Wan keyframes + Flash decoder")
        subtitle.setObjectName("muted")
        header.addWidget(subtitle)
        header.addStretch(1)
        self.gpu_label = QLabel("")
        self.gpu_label.setObjectName("muted")
        header.addWidget(self.gpu_label)
        self.env_button = QPushButton(tr("Environment"))
        self.env_button.clicked.connect(self.open_setup)
        header.addWidget(self.env_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- queue
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.addWidget(QLabel(tr("Queue")))
        self.queue = QListWidget()
        self.queue.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.queue.setMinimumWidth(320)
        self.queue.itemDoubleClicked.connect(self.open_job_output)
        queue_layout.addWidget(self.queue, 1)
        hint = QLabel(tr("Drop videos or images here"))
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        queue_layout.addWidget(hint)
        queue_buttons = QHBoxLayout()
        for text, slot in ((tr("Add files…"), self.add_files), (tr("Add folder…"), self.add_folder), (tr("Remove"), self.remove_selected), (tr("Clear finished"), self.clear_finished)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            queue_buttons.addWidget(button)
        queue_layout.addLayout(queue_buttons)
        splitter.addWidget(queue_widget)

        # ---- settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_widget = QWidget()
        settings_scroll.setWidget(settings_widget)
        form_layout = QVBoxLayout(settings_widget)
        form_layout.setContentsMargins(6, 0, 6, 0)

        model_box = QGroupBox(tr("Model"))
        model_form = QFormLayout(model_box)
        self.scale_combo = QComboBox()
        self.scale_combo.addItem(tr("4x (RelayVSR)"), 4)
        self.scale_combo.addItem(tr("2x (RelayVSR_X2)"), 2)
        self.scale_combo.currentIndexChanged.connect(self.refresh_model_status)
        model_form.addRow(tr("Model"), self.scale_combo)
        self.model_status = QLabel("")
        self.model_status.setObjectName("muted")
        self.model_status.setWordWrap(True)
        self.model_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_form.addRow("", self.model_status)
        self.size_combo = QComboBox()
        for key, label in (("native", tr("Native scale")), ("fit1080", tr("Fit 1080p")), ("fit1440", tr("Fit 1440p")), ("fit2160", tr("Fit 4K (2160p)")), ("width", tr("Custom width"))):
            self.size_combo.addItem(label, key)
        self.size_combo.currentIndexChanged.connect(lambda: self.width_spin.setEnabled(self.size_combo.currentData() == "width"))
        model_form.addRow(tr("Output size"), self.size_combo)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 15360)
        self.width_spin.setSingleStep(2)
        self.width_spin.setEnabled(False)
        model_form.addRow(tr("Width"), self.width_spin)
        form_layout.addWidget(model_box)

        output_box = QGroupBox(tr("Output"))
        output_form = QFormLayout(output_box)
        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(tr("Same as input"))
        out_row.addWidget(self.output_dir_edit)
        browse_out = QPushButton(tr("Browse…"))
        browse_out.clicked.connect(self.browse_output_dir)
        out_row.addWidget(browse_out)
        output_form.addRow(tr("Output folder"), out_row)
        self.codec_combo = QComboBox()
        for code, label in CODECS:
            self.codec_combo.addItem(label, code)
        self.codec_combo.currentIndexChanged.connect(self._codec_changed)
        output_form.addRow(tr("Codec"), self.codec_combo)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        output_form.addRow(tr("Quality (CRF / CQ)"), self.crf_spin)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS_X26X)
        output_form.addRow(tr("Preset"), self.preset_combo)
        self.audio_check = QCheckBox(tr("Copy audio"))
        output_form.addRow("", self.audio_check)
        self.overwrite_check = QCheckBox(tr("Overwrite existing files"))
        output_form.addRow("", self.overwrite_check)
        form_layout.addWidget(output_box)

        adv_box = QGroupBox(tr("Advanced"))
        adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv_inner = QWidget()
        adv_form = QFormLayout(adv_inner)
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_layout = QVBoxLayout(adv_box)
        adv_layout.addWidget(adv_inner)
        adv_box.toggled.connect(adv_inner.setVisible)
        adv_inner.setVisible(False)
        self.weights_combo = QComboBox()
        for key, label in (("auto", tr("Auto")), ("fp8", "FP8 (E4M3 weight-only, ~5 GB)"), ("bf16", "BF16 (~10 GB)")):
            self.weights_combo.addItem(label, key)
        adv_form.addRow(tr("Wan weights"), self.weights_combo)
        self.vram_combo = QComboBox()
        for key, label in (
            ("auto", tr("Auto")),
            ("quality", tr("Quality first (BF16 if it fits)")),
            ("full", tr("Full frame")),
            ("tiled", tr("Tiled keyframes")),
            ("offload", tr("Block offload")),
            ("low_vram", tr("Low VRAM")),
        ):
            self.vram_combo.addItem(label, key)
        adv_form.addRow(tr("VRAM strategy"), self.vram_combo)
        self.attention_combo = QComboBox()
        for key, label in (("auto", tr("Auto")), ("triton", "Triton"), ("sdpa", "SDPA")):
            self.attention_combo.addItem(label, key)
        adv_form.addRow(tr("Attention kernels"), self.attention_combo)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 64)
        self.window_spin.setToolTip("Frames per keyframe window (16 = a Wan keyframe every 15 frames)")
        adv_form.addRow(tr("Keyframe interval"), self.window_spin)
        self.kv_spin = QSpinBox()
        self.kv_spin.setRange(0, 8)
        self.kv_spin.setSpecialValueText("checkpoint (4)")
        adv_form.addRow(tr("Keyframe KV window"), self.kv_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        adv_form.addRow(tr("Seed"), self.seed_spin)
        self.detail_spin = QDoubleSpinBox()
        self.detail_spin.setRange(0.0, 1.5)
        self.detail_spin.setSingleStep(0.05)
        adv_form.addRow(tr("Detail strength"), self.detail_spin)
        range_row = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 10_000_000)
        self.start_spin.setPrefix(tr("Start frame") + " ")
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 10_000_000)
        self.max_spin.setSpecialValueText(tr("all"))
        self.max_spin.setPrefix(tr("Max frames") + " ")
        range_row.addWidget(self.start_spin)
        range_row.addWidget(self.max_spin)
        adv_form.addRow(tr("Frame range"), range_row)
        self.preview_check = QCheckBox(tr("Show previews"))
        adv_form.addRow("", self.preview_check)
        self.ffmpeg_edit = QLineEdit()
        self.ffmpeg_edit.setPlaceholderText("ffmpeg.exe")
        adv_form.addRow(tr("External ffmpeg path (optional)"), self.ffmpeg_edit)
        self.external_ffmpeg_check = QCheckBox(tr("Use external ffmpeg for encoding"))
        adv_form.addRow("", self.external_ffmpeg_check)
        form_layout.addWidget(adv_box)
        form_layout.addStretch(1)
        splitter.addWidget(settings_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # ---- progress / preview
        bottom = QVBoxLayout()
        preview_row = QHBoxLayout()
        self.preview_in = QLabel(tr("Before"))
        self.preview_out = QLabel(tr("After"))
        for label in (self.preview_in, self.preview_out):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(170)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            label.setStyleSheet("border: 1px solid #3a3f47; border-radius: 6px; color: #777;")
            preview_row.addWidget(label, 1)
        self.preview_frame = QWidget()
        self.preview_frame.setLayout(preview_row)
        self.preview_frame.setMaximumHeight(240)
        bottom.addWidget(self.preview_frame)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        self.progress_label = QLabel(tr("Ready"))
        bottom.addWidget(self.progress_label)

        controls = QHBoxLayout()
        self.start_button = QPushButton(tr("Start"))
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumWidth(140)
        self.start_button.clicked.connect(self.start_queue)
        controls.addWidget(self.start_button)
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        controls.addWidget(self.cancel_button)
        self.pause_check = QCheckBox(tr("Pause after current"))
        controls.addWidget(self.pause_check)
        controls.addStretch(1)
        open_output = QPushButton(tr("Open output folder"))
        open_output.clicked.connect(self.open_output_folder)
        controls.addWidget(open_output)
        self.log_toggle = QPushButton(tr("Log"))
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(lambda checked: self.log_view.setVisible(checked))
        controls.addWidget(self.log_toggle)
        bottom.addLayout(controls)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setMaximumHeight(180)
        self.log_view.setVisible(False)
        bottom.addWidget(self.log_view)
        root.addLayout(bottom)

    def _codec_changed(self) -> None:
        codec = self.codec_combo.currentData()
        nvenc = str(codec).endswith("_nvenc")
        self.preset_combo.setEnabled(not nvenc)
        self.crf_spin.setToolTip("NVENC constant quality (cq)" if nvenc else "x264/x265 CRF (lower = better quality)")

    # ------------------------------------------------------------ settings
    def _load_settings_into_ui(self) -> None:
        s = self.settings
        self._select(self.scale_combo, s.scale)
        self._select(self.size_combo, s.size_mode)
        self.width_spin.setValue(s.custom_width)
        self.width_spin.setEnabled(s.size_mode == "width")
        self.output_dir_edit.setText(s.output_dir)
        self._select(self.codec_combo, s.codec)
        self.crf_spin.setValue(s.crf)
        if s.preset in PRESETS_X26X:
            self.preset_combo.setCurrentText(s.preset)
        self.audio_check.setChecked(s.copy_audio)
        self.overwrite_check.setChecked(s.overwrite)
        self._select(self.weights_combo, s.wan_weights)
        self._select(self.vram_combo, s.vram_mode)
        self._select(self.attention_combo, s.attention_backend)
        self.window_spin.setValue(s.window_size)
        self.kv_spin.setValue(s.kv_window)
        self.seed_spin.setValue(s.seed)
        self.detail_spin.setValue(s.detail)
        self.preview_check.setChecked(s.preview)
        self.ffmpeg_edit.setText(s.ffmpeg_path)
        self.external_ffmpeg_check.setChecked(s.use_external_ffmpeg)
        self._codec_changed()
        self.refresh_model_status()

    @staticmethod
    def _select(combo: QComboBox, data) -> None:
        index = combo.findData(data)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _save_settings_from_ui(self) -> None:
        s = self.settings
        s.scale = int(self.scale_combo.currentData())
        s.size_mode = str(self.size_combo.currentData())
        s.custom_width = self.width_spin.value()
        s.output_dir = self.output_dir_edit.text().strip()
        s.codec = str(self.codec_combo.currentData())
        s.crf = self.crf_spin.value()
        s.preset = self.preset_combo.currentText()
        s.copy_audio = self.audio_check.isChecked()
        s.overwrite = self.overwrite_check.isChecked()
        s.wan_weights = str(self.weights_combo.currentData())
        s.vram_mode = str(self.vram_combo.currentData())
        s.attention_backend = str(self.attention_combo.currentData())
        s.window_size = self.window_spin.value()
        s.kv_window = self.kv_spin.value()
        s.seed = self.seed_spin.value()
        s.detail = round(self.detail_spin.value(), 3)
        s.preview = self.preview_check.isChecked()
        s.ffmpeg_path = self.ffmpeg_edit.text().strip()
        s.use_external_ffmpeg = self.external_ffmpeg_check.isChecked()
        s.save()

    def set_language(self, code: str) -> None:
        self.settings.language = code
        self.settings.save()
        QMessageBox.information(self, tr("Language"), "Restart the application to apply the language.")

    # --------------------------------------------------------- environment
    def check_environment(self) -> None:
        self._status_thread = _StatusThread(self.envmgr, self)
        self._status_thread.result.connect(self._environment_checked)
        self._status_thread.start()

    def _environment_checked(self, status: EnvStatus) -> None:
        self.env_status = status
        if status.ready:
            self.statusBar().showMessage(tr("Engine ready"))
            self.env_button.setText(tr("Environment") + " ✓")
            if not self.client.is_running():
                self.client.start()
                self.client.probe(models_dir=str(self.settings.models_path()), ffmpeg_path=self.settings.ffmpeg_path or None)
        else:
            self.statusBar().showMessage(tr("Engine not installed"))
            self.env_button.setText(tr("Environment") + " !")
            if status.mode == "managed":
                answer = QMessageBox.question(self, tr("Environment setup"), tr("The inference engine is not installed yet. Install it now? (downloads about 3.5 GB)"))
                dialog = self.open_setup()
                if answer == QMessageBox.StandardButton.Yes:
                    dialog.install()

    def open_setup(self) -> SetupDialog:
        dialog = SetupDialog(self.settings, self.envmgr, self)
        dialog.environment_changed.connect(self._environment_changed)
        dialog.models_changed.connect(self.refresh_model_status)
        dialog.setModal(False)
        dialog.show()
        self._setup_dialog = dialog
        return dialog

    def _environment_changed(self) -> None:
        self.client.stop()
        self.check_environment()

    def refresh_model_status(self) -> None:
        scale = int(self.scale_combo.currentData())
        if model_files.models_present(self.settings.models_path(), scale):
            self.model_status.setText(f"{tr('Installed')} · {self.settings.models_path() / model_files.MODEL_FILES[scale]['folder']}")
        else:
            self.model_status.setText(f"{tr('Not downloaded')} · {model_files.MODEL_FILES[scale]['repo']} ({model_files.missing_bytes(self.settings.models_path(), scale) / 1024**3:.1f} GB)")

    # ---------------------------------------------------------------- jobs
    def add_paths(self, paths: list[str | Path]) -> None:
        added = 0
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                files = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT)
                images = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT]
                if not files and images:
                    files = [path]  # image sequence folder
                for file in files:
                    self._add_job(file)
                    added += 1
            elif path.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
                self._add_job(path)
                added += 1
        if added:
            self.settings.last_input_dir = str(Path(paths[0]).parent)
            self.settings.save()

    def _add_job(self, path: Path) -> None:
        if any(j.input == path and j.status in ("waiting", "running") for j in self.jobs):
            return
        job = Job(input=path)
        self.jobs.append(job)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, job.id)
        self.queue.addItem(item)
        self._update_item(job)

    def _item_for(self, job: Job) -> QListWidgetItem | None:
        for i in range(self.queue.count()):
            item = self.queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == job.id:
                return item
        return None

    def _update_item(self, job: Job) -> None:
        item = self._item_for(job)
        if item is None:
            return
        labels = {"waiting": tr("Waiting"), "running": tr("Running"), "done": tr("Done"), "failed": tr("Failed"), "cancelled": tr("Cancelled")}
        icons = {"waiting": "•", "running": "▶", "done": "✓", "failed": "✗", "cancelled": "–"}
        text = f"{icons[job.status]} {job.input.name}\n    {labels[job.status]}"
        if job.status == "running" and job.message:
            text += f" · {job.message}"
        elif job.status == "done" and job.result:
            text += f" · {job.result.get('output_size', ['?', '?'])[0]}x{job.result.get('output_size', ['?', '?'])[1]} · {job.result.get('fps', 0)} fps"
        elif job.status == "failed" and job.error:
            text += f" · {job.error[:80]}"
        item.setText(text)
        item.setSizeHint(QSize(0, 40))

    def add_files(self) -> None:
        start = self.settings.last_input_dir or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, tr("Add files…"), start, "Media (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.mts *.m2ts *.wmv *.flv *.mpg *.mpeg *.gif *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)")
        if files:
            self.add_paths(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Add folder…"), self.settings.last_input_dir or str(Path.home()))
        if folder:
            self.add_paths([folder])

    def remove_selected(self) -> None:
        for item in self.queue.selectedItems():
            job_id = item.data(Qt.ItemDataRole.UserRole)
            job = next((j for j in self.jobs if j.id == job_id), None)
            if job is not None and job.status == "running":
                continue
            self.jobs = [j for j in self.jobs if j.id != job_id]
            self.queue.takeItem(self.queue.row(item))

    def clear_finished(self) -> None:
        for job in [j for j in self.jobs if j.status in ("done", "failed", "cancelled")]:
            item = self._item_for(job)
            if item is not None:
                self.queue.takeItem(self.queue.row(item))
        self.jobs = [j for j in self.jobs if j.status in ("waiting", "running")]

    def browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Output folder"), self.output_dir_edit.text() or str(Path.home()))
        if folder:
            self.output_dir_edit.setText(folder)

    def open_output_folder(self) -> None:
        folder = self.output_dir_edit.text().strip()
        if not folder:
            job = self.current or next((j for j in reversed(self.jobs) if j.output), None)
            folder = str(job.output.parent) if job and job.output else str(Path.home())
        SetupDialog.open_folder(folder)

    def open_job_output(self, item: QListWidgetItem) -> None:
        job = next((j for j in self.jobs if j.id == item.data(Qt.ItemDataRole.UserRole)), None)
        if job and job.output and job.output.exists():
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(job.output)])
            else:  # pragma: no cover
                webbrowser.open(str(job.output.parent))

    # ---------------------------------------------------------- job config
    def _output_path(self, job: Job, scale: int) -> Path:
        folder = Path(self.output_dir_edit.text().strip()) if self.output_dir_edit.text().strip() else job.input.parent
        if job.input.is_dir():
            stem = job.input.name
        else:
            stem = job.input.stem
        mode = self.size_combo.currentData()
        tag = {"native": f"x{scale}", "fit1080": "1080p", "fit1440": "1440p", "fit2160": "2160p", "width": f"w{self.width_spin.value()}"}[mode]
        suffix = ".png" if (job.input.suffix.lower() in IMAGE_EXT and not job.input.is_dir()) else ".mp4"
        candidate = folder / f"{stem}_relayvsr_{tag}{suffix}"
        if not self.overwrite_check.isChecked():
            counter = 1
            while candidate.exists():
                candidate = folder / f"{stem}_relayvsr_{tag}_{counter}{suffix}"
                counter += 1
        return candidate

    def build_job_config(self, job: Job) -> dict:
        scale = int(self.scale_combo.currentData())
        job.scale = scale
        job.output = self._output_path(job, scale)
        vram = self.vram_combo.currentData()
        cfg = {
            "input": str(job.input),
            "output": str(job.output),
            "scale": scale,
            "models_dir": str(self.settings.models_path()),
            "wan_weights": self.weights_combo.currentData(),
            "block_offload": {"offload": "on", "full": "off", "tiled": "off"}.get(vram, "auto"),
            "tiling": {"tiled": "on", "full": "off", "offload": "auto"}.get(vram, "auto"),
            "quality_preference": {"quality": "quality", "low_vram": "low_vram"}.get(vram, "auto"),
            "attention_backend": self.attention_combo.currentData(),
            "window_size": self.window_spin.value(),
            "kv_window": self.kv_spin.value() or None,
            "seed": self.seed_spin.value(),
            "detail": round(self.detail_spin.value(), 3),
            "codec": self.codec_combo.currentData(),
            "crf": self.crf_spin.value(),
            "preset": self.preset_combo.currentText(),
            "copy_audio": self.audio_check.isChecked(),
            "overwrite": True,
            "start_frame": self.start_spin.value(),
            "max_frames": self.max_spin.value() or None,
            "use_external_ffmpeg": self.external_ffmpeg_check.isChecked(),
            "ffmpeg_path": self.ffmpeg_edit.text().strip() or None,
            "preview_every": 24 if self.preview_check.isChecked() else 0,
            "preview_dir": str(app_home() / "previews" / job.id) if self.preview_check.isChecked() else None,
            "output_format": "video",
        }
        if vram == "quality":
            cfg["wan_weights"] = "bf16" if cfg["wan_weights"] == "auto" else cfg["wan_weights"]
            cfg["quality_preference"] = "quality"
        mode = self.size_combo.currentData()
        if mode == "fit1080":
            cfg["fit_short_side"] = 1080
        elif mode == "fit1440":
            cfg["fit_short_side"] = 1440
        elif mode == "fit2160":
            cfg["fit_short_side"] = 2160
        elif mode == "width":
            cfg["output_width"] = self.width_spin.value()
        if job.input.is_dir():
            cfg["fps"] = "24"
        return cfg

    # ------------------------------------------------------------- running
    def start_queue(self) -> None:
        self._save_settings_from_ui()
        if self.env_status is None or not self.env_status.ready:
            QMessageBox.warning(self, APP_NAME, tr("Install the engine first (Environment)."))
            self.open_setup()
            return
        if not any(j.status == "waiting" for j in self.jobs):
            QMessageBox.information(self, APP_NAME, tr("Add at least one input file."))
            return
        scale = int(self.scale_combo.currentData())
        if not model_files.models_present(self.settings.models_path(), scale):
            if QMessageBox.question(self, APP_NAME, tr("Model files for this scale are not downloaded yet. Download them now?")) == QMessageBox.StandardButton.Yes:
                dialog = self.open_setup()
                dialog.download(scale)
            return
        self.pause_check.setChecked(False)
        self.run_next()

    def run_next(self) -> None:
        if self.current is not None:
            return
        if self.pause_check.isChecked():
            self.pause_check.setChecked(False)
            self.progress_label.setText(tr("Ready"))
            return
        job = next((j for j in self.jobs if j.status == "waiting"), None)
        if job is None:
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.progress_label.setText(tr("Done"))
            self.statusBar().showMessage(tr("Done"))
            return
        try:
            cfg = self.build_job_config(job)
        except Exception as error:  # noqa: BLE001
            job.status, job.error = "failed", str(error)
            self._update_item(job)
            self.run_next()
            return
        self.current = job
        job.status = "running"
        job.message = tr("Loading models…")
        job.progress = 0.0
        self._update_item(job)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.progress_label.setText(f"{job.input.name} — {tr('Loading models…')}")
        self.preview_in.setPixmap(QPixmap())
        self.preview_out.setPixmap(QPixmap())
        self.preview_in.setText(tr("Before"))
        self.preview_out.setText(tr("After"))
        self._run_started = time.time()
        self.append_log(f"=== {job.input} -> {job.output}")
        self.append_log(f"config: {cfg}")
        self.client._busy_after_cancel = True
        job.command_id = self.client.upscale(cfg)

    def cancel_current(self) -> None:
        if self.current is not None:
            self.progress_label.setText(tr("Cancelled") + "…")
            self.client.cancel()

    def on_worker_stopped(self, code: int) -> None:
        if self.current is not None:
            job = self.current
            job.status = "failed"
            job.error = tr("Engine process exited unexpectedly") + f" (code {code})"
            self._update_item(job)
            self.current = None
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.progress_label.setText(job.error)

    # -------------------------------------------------------------- events
    def on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "hello":
            self.append_log(f"worker pid {event.get('pid')} python {event.get('python')}")
        elif kind == "probe":
            self.gpu = event.get("gpu")
            self.encoders = event.get("encoders")
            self._show_gpu()
        elif kind == "gpu":
            self.gpu = event.get("gpu")
            self._show_gpu()
        elif kind == "log":
            if event.get("level") in ("warning", "error"):
                self.append_log(f"[{event.get('level')}] {event.get('message')}")
            else:
                self.append_log(str(event.get("message")))
        elif kind == "stage":
            self._on_stage(event)
        elif kind == "plan":
            plan = event.get("plan", {})
            text = (
                f"{tr('Plan')}: {plan.get('wan_weights')}, "
                f"{'offload, ' if plan.get('block_offload') else ''}"
                f"{plan.get('tiles', 1)} tile(s), KV {plan.get('kv_window')}, "
                f"{tr('Estimated VRAM')} {plan.get('estimate_gib')} / {plan.get('free_gib')} GiB — {plan.get('reason')}"
            )
            self.append_log(text)
            self.statusBar().showMessage(text)
        elif kind == "models":
            self.append_log(f"models: {event.get('details')} ({event.get('load_seconds')}s)")
        elif kind == "progress":
            self._on_progress(event)
        elif kind == "preview":
            self._on_preview(event)
        elif kind == "finished":
            self._on_finished(event)
        elif kind == "error":
            self.append_log(f"[error] {event.get('message')}")
            if event.get("traceback"):
                self.append_log(str(event["traceback"]))

    def _show_gpu(self) -> None:
        if not self.gpu:
            return
        g = self.gpu
        if not g.get("available"):
            self.gpu_label.setText(tr("No CUDA GPU detected"))
            return
        text = f"{g.get('name')} · {g.get('total_gib')} GiB · {g.get('architecture')} · Triton {'✓' if g.get('triton') else '✗'}"
        if g.get("capability") and tuple(g["capability"]) >= (12, 0):
            text += " · " + tr("Preset: RTX 50 series (Blackwell) tuned")
        self.gpu_label.setText(text)

    def _on_stage(self, event: dict) -> None:
        stage = event.get("stage")
        job = self.current
        if job is None:
            return
        if stage == "load":
            job.message = tr("Loading models…")
        elif stage == "run":
            size_in = event.get("input_size", [0, 0])
            size_out = event.get("output_size", [0, 0])
            job.message = f"{size_in[0]}x{size_in[1]} → {size_out[0]}x{size_out[1]}"
            self.progress_label.setText(f"{job.input.name} — {job.message}")
        elif stage == "encode":
            self.progress_label.setText(f"{job.input.name} — {event.get('message', '')}")
        elif stage == "mux":
            self.progress_label.setText(f"{job.input.name} — {event.get('message', '')}")
        self._update_item(job)

    def _on_progress(self, event: dict) -> None:
        job = self.current
        if job is None:
            return
        total = event.get("total") or 0
        frame = event.get("frame", 0)
        if total:
            job.progress = frame / total
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(job.progress * 1000))
        else:
            self.progress.setRange(0, 0)
        eta = event.get("eta")
        eta_text = f" · {tr('ETA')} {int(eta // 60)}:{int(eta % 60):02d}" if eta is not None else ""
        self.progress_label.setText(
            f"{job.input.name} — {frame}/{total or '?'} {tr('frames')} · {event.get('fps', 0):.2f} fps{eta_text} · "
            f"VRAM {event.get('vram_gib', 0):.1f} GiB ({tr('peak')} {event.get('vram_peak_gib', 0):.1f}) · {tr('Keyframes')} {event.get('keyframes', 0)}"
        )
        job.message = f"{frame}/{total or '?'} · {event.get('fps', 0):.1f} fps"
        self._update_item(job)

    def _on_preview(self, event: dict) -> None:
        for key, label in (("input_path", self.preview_in), ("path", self.preview_out)):
            path = event.get(key)
            if path and Path(path).exists():
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    label.setText("")

    def _on_finished(self, event: dict) -> None:
        job = self.current
        self.client._busy_after_cancel = False
        if job is None:
            return
        if event.get("ok"):
            job.status = "done"
            job.result = event.get("result")
            job.progress = 1.0
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            r = job.result or {}
            self.progress_label.setText(f"{tr('Output written to')} {r.get('output')} · {r.get('frames')} {tr('frames')} · {r.get('fps')} fps · {tr('peak')} VRAM {r.get('peak_vram_gib')} GiB")
            self.append_log(f"done: {r}")
        elif event.get("cancelled"):
            job.status = "cancelled"
            self.progress_label.setText(tr("Cancelled"))
        else:
            job.status = "failed"
            job.error = event.get("error", "error")
            self.progress_label.setText(f"{tr('Failed')}: {job.error}")
            self.append_log(f"failed: {job.error}\n{event.get('traceback', '')}")
        self._update_item(job)
        self.current = None
        self._cleanup_previews(job)
        QTimer.singleShot(50, self.run_next)

    def _cleanup_previews(self, job: Job) -> None:
        folder = app_home() / "previews" / job.id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    # ------------------------------------------------------------- helpers
    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        log.info(text)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            tr("About"),
            f"<b>{APP_NAME} {__version__}</b><br>"
            "Windows desktop video upscaler built on <a href='https://kopperx.github.io/RelayVSR/'>RelayVSR</a> "
            "(Wang et al., USTC): a one-step Wan 2.2 keyframe model relays sparse references to a lightweight "
            "streaming decoder.<br><br>"
            "Model code: Apache-2.0 (RelayVSR authors). Weights: kopper/RelayVSR, lightx2v/RelayVSR_X2.<br>"
            "App: Apache-2.0.",
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.current is not None:
            if QMessageBox.question(self, APP_NAME, tr("Cancel") + "?") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._save_settings_from_ui()
        super().closeEvent(event)

    def shutdown(self) -> None:
        self.client.stop()


__all__ = ["MainWindow"]
