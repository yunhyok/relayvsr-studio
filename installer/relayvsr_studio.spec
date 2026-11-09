# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the GUI launcher (no torch inside; the engine lives in a managed env).

Environment variables (set by installer/build.ps1):
  RVS_BUILD_DIR  - folder containing wheels/ (app wheel) and tools/uv.exe
"""

import os
from pathlib import Path

spec_dir = Path(SPECPATH)
repo = spec_dir.parent
build_dir = Path(os.environ.get("RVS_BUILD_DIR", repo / "build"))

datas = [
    (str(repo / "requirements" / "engine-cu128.txt"), "requirements"),
    (str(repo / "relayvsr_studio" / "app" / "resources"), "relayvsr_studio/app/resources"),
    (str(repo / "LICENSE"), "."),
    (str(repo / "THIRD_PARTY_NOTICES.md"), "."),
]
wheels = sorted((build_dir / "wheels").glob("relayvsr_studio-*.whl")) if (build_dir / "wheels").is_dir() else []
if wheels:
    datas.append((str(wheels[-1]), "wheels"))
uv = build_dir / "tools" / "uv.exe"
if uv.is_file():
    datas.append((str(uv), "tools"))

a = Analysis(
    [str(spec_dir / "launcher.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=datas,
    hiddenimports=["relayvsr_studio.app.main", "relayvsr_studio.app.main_window", "relayvsr_studio.app.setup_dialog"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "triton", "diffusers", "accelerate", "av", "numpy", "PIL", "safetensors", "einops",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
        "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "PySide6.QtPdf", "PySide6.QtDesigner", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtOpenGL",
        "PySide6.QtNetwork", "PySide6.QtSvg", "tkinter", "unittest", "pydoc", "doctest",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RelayVSR Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(repo / "relayvsr_studio" / "app" / "resources" / "icon.ico"),
    version=str(build_dir / "version_info.txt") if (build_dir / "version_info.txt").is_file() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RelayVSR Studio",
)
