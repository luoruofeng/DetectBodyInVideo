"""桌面运行时依赖检测（任务 10.3）。

Windows 上 pywebview 依赖 WebView2 Runtime；没装时给出可操作的提示，
并允许降级到系统浏览器，避免用户只看到黑窗口。
"""

from __future__ import annotations

import platform
from pathlib import Path

WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"

_WEBVIEW2_GUID_NAME = "WebView2"


def is_windows() -> bool:
    return platform.system() == "Windows"


def _registry_webview2_version() -> str | None:
    try:
        import winreg  # noqa: PLC0415 - 仅 Windows
    except ImportError:
        return None

    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
    ]
    for root, path in roots:
        try:
            key = winreg.OpenKey(root, path)
        except OSError:
            continue
        with key:
            index = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(key, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(key, sub_name) as sub:
                        try:
                            display_name = winreg.QueryValueEx(sub, "name")[0]
                        except OSError:
                            continue
                        if _WEBVIEW2_GUID_NAME not in str(display_name):
                            continue
                        try:
                            return str(winreg.QueryValueEx(sub, "pv")[0])
                        except OSError:
                            return "unknown"
                except OSError:
                    continue
    return None


def _filesystem_webview2() -> bool:
    bases = [
        Path(r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application"),
        Path(r"C:\Program Files\Microsoft\EdgeWebView\Application"),
    ]
    for base in bases:
        if any(base.glob("*/msedgewebview2.exe")):
            return True
    return False


def webview2_status() -> tuple[bool, str | None]:
    """返回 ``(是否可用, 版本或 None)``；非 Windows 视为可用。"""
    if not is_windows():
        return True, None
    version = _registry_webview2_version()
    if version:
        return True, version
    if _filesystem_webview2():
        return True, "unknown"
    return False, None


def webview2_hint() -> str:
    return (
        "未检测到 Microsoft Edge WebView2 Runtime，桌面窗口可能无法创建。\n"
        f"请安装后重试：{WEBVIEW2_DOWNLOAD_URL}\n"
        "也可以先使用 --browser 参数在系统浏览器中打开。"
    )


def notify_windows(message: str, title: str = "DetectBodyInVideo") -> None:
    """Windows 原生消息框；其它平台只写日志。"""
    if not is_windows():
        return
    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x00000040 | 0x00000000)
    except Exception:  # noqa: BLE001 - 提示失败不影响降级
        pass


__all__ = [
    "WEBVIEW2_DOWNLOAD_URL",
    "is_windows",
    "notify_windows",
    "webview2_hint",
    "webview2_status",
]
