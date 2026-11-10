"""Application entry point."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .. import APP_NAME, __version__
from ..paths import logs_dir


def _setup_logging() -> None:
    folder = logs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(folder / "studio.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler(sys.stderr)], format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _dark_palette(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    base = QColor(30, 32, 36)
    panel = QColor(39, 42, 48)
    text = QColor(228, 230, 235)
    accent = QColor(86, 156, 255)
    palette.setColor(QPalette.ColorRole.Window, base)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, panel)
    palette.setColor(QPalette.ColorRole.AlternateBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipBase, panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, panel)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 90, 90))
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 145, 155))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 124, 130))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 124, 130))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(120, 124, 130))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QGroupBox { border: 1px solid #3a3f47; border-radius: 6px; margin-top: 10px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fb3d1; }
        QProgressBar { border: 1px solid #3a3f47; border-radius: 4px; text-align: center; height: 18px; }
        QProgressBar::chunk { background-color: #569cff; border-radius: 3px; }
        QPushButton { padding: 5px 12px; border-radius: 4px; border: 1px solid #3a3f47; background: #2f333a; }
        QPushButton:hover { background: #3a3f48; }
        QPushButton:disabled { color: #777; }
        QPushButton#primary { background: #2f6fd6; border-color: #2f6fd6; color: white; font-weight: 600; }
        QPushButton#primary:hover { background: #3b7ee6; }
        QPushButton#danger { background: #7a2e2e; border-color: #7a2e2e; color: white; }
        QListWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #3a3f47; border-radius: 4px; padding: 2px; }
        QLabel#muted { color: #9aa3b2; }
        QLabel#title { font-size: 18px; font-weight: 600; }
        """
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    log = logging.getLogger("relayvsr_studio.app")
    log.info("%s %s starting", APP_NAME, __version__)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from ..bootstrap.envmgr import EnvironmentManager
    from ..paths import bundle_dir
    from .i18n import set_language
    from .main_window import MainWindow
    from .settings import Settings

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("RelayVSR Studio")
    app.setStyle("Fusion")
    _dark_palette(app)
    for candidate in (bundle_dir() / "relayvsr_studio" / "app" / "resources" / "icon.ico", bundle_dir() / "resources" / "icon.ico"):
        if candidate.is_file():
            app.setWindowIcon(QIcon(str(candidate)))
            break

    settings = Settings.load()
    set_language(settings.language)
    envmgr = EnvironmentManager(settings.env_path(), settings.scratch_path())
    window = MainWindow(settings, envmgr)
    window.show()
    code = app.exec()
    window.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
