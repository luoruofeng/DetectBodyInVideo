"""前端 ↔ 桌面壳的原生能力桥（pywebview ``js_api``）。

目前只暴露一件浏览器做不到的事：导出时弹系统「另存为」对话框，让用户自己选
文件名与保存位置。默认目录是系统桌面（Windows 会读注册表里的 Shell Folders，
从而兼容 OneDrive 重定向后的桌面路径）。

前端调用方式（pywebview 5.x 会把返回值包成 Promise）：

```js
const path = await window.pywebview.api.choose_export_path('clip_20260101-120000_posefull.json')
```

用户取消时返回空字符串，前端据此放弃本次导出，不写任何文件。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("detect_body.desktop")

_JSON_FILTER = ("JSON 文件 (*.json)",)


def desktop_dir() -> Path:
    """系统桌面目录；找不到时退回用户主目录，保证对话框总能打开。"""
    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.extend(_windows_desktop_candidates())
    home = Path.home()
    candidates.extend([home / "Desktop", home / "桌面", home / "OneDrive" / "Desktop"])
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:  # 权限不足/路径异常时继续尝试下一个
            continue
    return home


def _windows_desktop_candidates() -> list[Path]:
    """从注册表读桌面路径：中文系统或 OneDrive 重定向后 ``~/Desktop`` 并不存在。"""
    try:
        import winreg  # noqa: PLC0415 - 仅 Windows
    except ImportError:
        return []
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, key_path) as key:
                value = winreg.QueryValueEx(key, "Desktop")[0]
        except OSError:
            continue
        if isinstance(value, str) and value.strip():
            return [Path(os.path.expandvars(value))]
    return []


def _clean_filename(name: str) -> str:
    """只取文件名部分，避免前端传来的名字里混进路径分隔符。"""
    cleaned = Path(str(name or "")).name.strip()
    if not cleaned or cleaned in {".", ".."}:
        return "export.json"
    return cleaned if cleaned.lower().endswith(".json") else f"{cleaned}.json"


def _first_path(result: object) -> str:
    """不同平台后端返回 str / tuple / list / None，这里统一成字符串。"""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return str(first) if first else ""
    return ""


class DesktopBridge:
    """挂到 ``webview.create_window(js_api=...)`` 上的对象，方法即前端 API。"""

    def __init__(self) -> None:
        self.window: object | None = None

    def choose_export_path(self, default_name: str = "export.json") -> str:
        """弹系统「另存为」对话框，返回绝对路径；取消或失败返回空字符串。"""
        window = self.window
        if window is None:
            return ""
        try:
            import webview  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - 无 pywebview 时让前端走默认目录
            logger.warning("无法导入 pywebview，跳过保存对话框：%s", exc)
            return ""
        try:
            result = window.create_file_dialog(  # type: ignore[attr-defined]
                webview.SAVE_DIALOG,
                directory=str(desktop_dir()),
                save_filename=_clean_filename(default_name),
                file_types=_JSON_FILTER,
            )
        except Exception as exc:  # noqa: BLE001 - 打开失败不该让导出整体失败
            logger.warning("打开「另存为」对话框失败：%s", exc)
            return ""
        path = _first_path(result)
        if path:
            logger.info("用户选择导出路径：%s", path)
        return path


__all__ = ["DesktopBridge", "desktop_dir"]
