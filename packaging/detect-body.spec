# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（任务 10.2）。

用法（仓库根目录）：
    pyinstaller packaging/detect-body.spec --noconfirm

- onedir：启动快、动态库排查容易；产物在 dist/DetectBodyInVideo/
- 前端静态文件、models/*.task、图标打进 _MEIPASS，运行时由 settings.RESOURCE_ROOT 定位
- 没有安装 mediapipe / onnxruntime 时自动跳过，便于先在干净环境验证壳
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
CONSOLE = os.environ.get("DETECT_BODY_CONSOLE", "") == "1"

datas = []
binaries = []
hiddenimports = []
for package in ("uvicorn", "h11", "anyio"):
    try:
        hiddenimports += collect_submodules(package)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] 跳过 hiddenimports {package}: {exc}")

# 桌面壳（任务 10.1）：pywebview 在 guilib 里用 importlib 动态加载平台后端，
# 静态分析看不到，必须显式声明，否则打包后窗口打不开只剩浏览器降级。
hiddenimports += [
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "webview.platforms.gtk",
    "webview.platforms.qt",
    "webview.platforms.cocoa",
    "clr_loader",
    "pythonnet",
]
try:
    # 自带 hook-webview.py 会收集 lib/*.dll（WebView2 托管程序集）；显式再收集一次更稳
    datas += collect_data_files("webview", subdir="lib")
    datas += collect_data_files("webview", subdir="js")
except Exception as exc:  # noqa: BLE001
    print(f"[spec] 跳过 webview 数据文件: {exc}")

static_dir = ROOT / "app" / "backend" / "static"
if (static_dir / "index.html").is_file():
    datas.append((str(static_dir), "app/backend/static"))

models_dir = ROOT / "models"
if models_dir.is_dir():
    for model in models_dir.glob("*.task"):
        datas.append((str(model), "models"))

for package in ("mediapipe", "onnxruntime"):
    try:
        datas += collect_data_files(package)
        binaries += collect_dynamic_libs(package)
        hiddenimports += collect_submodules(package)
    except Exception as exc:  # noqa: BLE001 - 可选依赖缺失不阻断壳构建
        print(f"[spec] 跳过 {package}: {exc}")

icon_png = ROOT / "app" / "desktop" / "assets" / "icon.png"
icon_ico = ROOT / "app" / "desktop" / "assets" / "icon.ico"
icon_icns = ROOT / "app" / "desktop" / "assets" / "icon.icns"
if icon_png.is_file():
    datas.append((str(icon_png), "app/desktop/assets"))
if icon_ico.is_file() and sys.platform == "win32":
    datas.append((str(icon_ico), "app/desktop/assets"))

exe_icon = str(icon_ico) if sys.platform == "win32" and icon_ico.is_file() else None

a = Analysis(
    [str(ROOT / "app" / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DetectBodyInVideo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DetectBodyInVideo",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DetectBodyInVideo.app",
        icon=str(icon_icns) if icon_icns.is_file() else None,
        bundle_identifier="ai.detectbody.desktop",
        info_plist={
            "CFBundleDisplayName": "DetectBodyInVideo",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
