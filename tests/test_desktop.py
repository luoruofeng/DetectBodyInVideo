"""桌面壳辅助函数（任务 10.1 / 10.3）。"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.desktop import bridge as desktop_bridge
from app.desktop import checks
from app.desktop import main as desktop


def test_find_free_port_prefers_available() -> None:
    port = desktop.find_free_port(8765)
    assert 1024 < port <= 65535


def test_wait_for_health_reads_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls = {"count": 0}

    def fake_urlopen(url, timeout=0):
        calls["count"] += 1
        assert url.endswith("/api/health")
        return FakeResponse(b'{"status": "ok"}')

    monkeypatch.setattr(desktop.urllib.request, "urlopen", fake_urlopen)
    assert desktop.wait_for_health("http://127.0.0.1:8000", timeout=1) is True
    assert calls["count"] == 1


def test_wait_for_health_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, timeout=0):
        raise OSError("refused")

    monkeypatch.setattr(desktop.urllib.request, "urlopen", boom)
    assert desktop.wait_for_health("http://127.0.0.1:1", timeout=0.3) is False


def test_webview2_status_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks.platform, "system", lambda: "Darwin")
    assert checks.webview2_status() == (True, None)
    assert "WebView2" in checks.webview2_hint()


def test_icon_path_resolves_generated_icon() -> None:
    path = desktop.icon_path()
    # 仓库内已由 scripts/make_icon.py 生成；若被清理则允许返回 None
    assert path is None or Path(path).is_file()


def test_window_title_includes_version() -> None:
    settings = SimpleNamespace(app_name="DetectBodyInVideo", version="1.2.3")
    assert desktop.window_title(settings) == "DetectBodyInVideo 1.2.3"
    assert desktop.window_title() == desktop.APP_TITLE


def test_stop_backend_cancels_jobs_then_exits() -> None:
    calls: list[str] = []

    class FakeJobs:
        def cancel_all(self) -> int:
            calls.append("cancel")
            return 2

    class FakeThread:
        def join(self, timeout: float | None = None) -> None:
            calls.append(f"join:{timeout}")

        def is_alive(self) -> bool:
            return False

    class FakeServer:
        should_exit = False

    server = FakeServer()
    server.config = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(analysis_jobs=FakeJobs())))

    desktop.stop_backend(server, FakeThread())  # type: ignore[arg-type]

    assert calls[0] == "cancel"  # 先取消任务，再停服务
    assert server.should_exit is True
    assert calls[-1] == f"join:{desktop.SHUTDOWN_TIMEOUT}"


def test_ensure_std_streams_fills_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """窗口模式下 sys.stdout/stderr 为 None，必须补成可用流，否则 uvicorn 启动即崩。"""

    class FakeSys:
        stdout = None
        stderr = None

    monkeypatch.setattr(desktop, "sys", FakeSys)
    desktop.ensure_std_streams()
    assert FakeSys.stdout is not None
    assert FakeSys.stderr is not None
    # uvicorn 会调 isatty()/write()，补齐后必须可用（Windows 上 nul 设备 isatty 为 True）
    FakeSys.stdout.write("smoke\n")
    FakeSys.stdout.flush()
    FakeSys.stderr.isatty()


def test_desktop_bridge_returns_empty_without_window() -> None:
    """未挂窗口时（浏览器模式）不能抛异常，前端据此退回默认导出目录。"""
    assert desktop_bridge.DesktopBridge().choose_export_path("a.json") == ""


def test_desktop_bridge_has_no_public_attributes() -> None:
    """js_api 对象上不能挂公开的非可调用属性。

    pywebview 注入 JS API 时会递归遍历 js_api 的所有公开属性（``get_functions``），
    一旦把 pywebview 的 Window 挂成公开属性，递归会走进 ``Window.native`` 的
    WinForms/.NET 对象图，最后卡在 Accessibility 的跨进程 COM 调用上；该线程持有
    GIL，桌面壳 UI 线程随之卡在 PyGILState_Ensure，整个窗口「未响应」。
    """

    bridge = desktop_bridge.DesktopBridge()
    public = [name for name in dir(bridge) if not name.startswith("_")]
    for name in public:
        attr = getattr(bridge, name)
        assert callable(attr), f"{name} 是公开非可调用属性，会被 pywebview 递归遍历"


def test_desktop_dir_exists() -> None:
    assert desktop_bridge.desktop_dir().is_dir()


def test_clean_filename_keeps_json_and_strips_paths() -> None:
    assert desktop_bridge._clean_filename("clip.json") == "clip.json"
    assert desktop_bridge._clean_filename("dir/clip") == "clip.json"
    assert desktop_bridge._clean_filename("..\\clip.JSON") == "clip.JSON"
    assert desktop_bridge._clean_filename("") == "export.json"


def test_first_path_accepts_str_and_sequences() -> None:
    assert desktop_bridge._first_path("C:/a.json") == "C:/a.json"
    assert desktop_bridge._first_path(("C:/a.json",)) == "C:/a.json"
    assert desktop_bridge._first_path(None) == ""
    assert desktop_bridge._first_path(()) == ""


def test_stop_backend_survives_missing_app_state() -> None:
    class FakeThread:
        joined = False

        def join(self, timeout: float | None = None) -> None:
            self.joined = True

        def is_alive(self) -> bool:
            return False

    thread = FakeThread()
    desktop.stop_backend(SimpleNamespace(), thread)  # type: ignore[arg-type]
    assert thread.joined is True
