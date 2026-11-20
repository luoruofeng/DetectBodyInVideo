"""桌面外壳启动器（任务 10.1）。

流程：选空闲端口 → 后台线程起 uvicorn → 轮询 /api/health → 打开 pywebview 窗口。
打包后资源从 ``sys._MEIPASS`` 读取；WebView2 缺失或 pywebview 不可用时降级为系统浏览器。
窗口关闭时先取消分析任务、再停 uvicorn，确保线程与显存尽快释放。

单端口：前端构建产物由 ``app/backend/static`` 托管，窗口与浏览器都只访问
``http://127.0.0.1:<port>``，不需要第二个静态服务器。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from app.backend.core.settings import RESOURCE_ROOT, Settings, get_settings

logger = logging.getLogger("detect_body.desktop")

APP_TITLE = "DetectBodyInVideo"
WINDOW_SIZE = (1440, 900)
WINDOW_MIN_SIZE = (1024, 700)
SHUTDOWN_TIMEOUT = 5.0


def ensure_std_streams() -> None:
    """窗口模式（PyInstaller runw）下 sys.stdout/sys.stderr 是 None。

    uvicorn 等三方库会直接调用 ``sys.stderr.isatty()``，None 会让启动直接崩掉，
    因此这里把缺失的标准流接到 devnull，日志仍完整写入 ``~/.detect-body/logs``。
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
            except OSError:
                pass


def find_free_port(preferred: int) -> int:
    """优先使用配置端口，被占用则让系统分配。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    url = f"{base_url}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    json.loads(response.read().decode("utf-8"))
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.15)
    return False


def start_backend(port: int, settings: Settings) -> tuple[object, threading.Thread]:
    """在后台线程启动 uvicorn，返回 (server, thread)。"""
    import uvicorn

    from app.backend.main import create_app

    config = uvicorn.Config(
        create_app(settings),
        host="127.0.0.1",
        port=port,
        log_level=settings.log_level.lower(),
        access_log=False,
        # 复用 app.backend.core.logging 的根 logger：uvicorn 自带的 dictConfig 在
        # 窗口模式（sys.stderr=None）下会因 isatty 崩溃，且只写控制台不落盘。
        log_config=None,
        # 固定实现，避免 PyInstaller 漏掉 uvicorn 的动态协议导入
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    return server, thread


def icon_path() -> Path | None:
    """打包时由 ``scripts/make_icon.py`` 生成，缺失则用系统默认图标。"""
    for name in ("icon.ico", "icon.png"):
        candidate = RESOURCE_ROOT / "app" / "desktop" / "assets" / name
        if candidate.is_file():
            return candidate
    return None


def window_title(settings: Settings | None = None) -> str:
    """标题栏文字：应用名 + 版本，便于用户确认自己跑的是哪个版本。"""
    if settings is None:
        return APP_TITLE
    return f"{settings.app_name} {settings.version}".strip()


def open_browser(url: str, *, wait: bool = True) -> None:
    webbrowser.open(url)
    if not wait:
        return
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _on_window_closed() -> None:
    logger.info("窗口已关闭，开始释放资源")


def run_window(url: str, settings: Settings) -> bool:
    """返回 True 表示窗口正常关闭；False 表示无法创建窗口，调用方可降级。"""
    try:
        import webview  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - 缺 WebView 运行时统一降级
        logger.warning("pywebview 不可用（%s），降级为系统浏览器模式", exc)
        return False

    from app.desktop.bridge import DesktopBridge

    icon = icon_path()
    bridge = DesktopBridge()
    try:
        window = webview.create_window(
            window_title(settings),
            url,
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=WINDOW_MIN_SIZE,
            background_color="#0B0F14",
            text_select=True,
            # 导出时前端通过 window.pywebview.api.choose_export_path() 弹系统另存为
            js_api=bridge,
        )
        # 窗口关闭事件只做日志：真正的清理在 main() 的 finally 里统一执行，
        # 保证「窗口关闭 → 取消任务 → 停服务」顺序不因事件线程时序而错乱。
        if window is not None:
            bridge.window = window
            window.events.closed += _on_window_closed
        webview.start(
            debug=bool(os.environ.get("DETECT_BODY_WEBVIEW_DEBUG")),
            private_mode=False,
            storage_path=str(settings.data_dir / "webview"),
            # Windows 窗口图标来自 exe 资源（PyInstaller icon=），此处仅 GTK/QT 生效
            icon=str(icon) if icon else None,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - WebKitGTK/WebView2 缺少运行时
        logger.exception("桌面窗口启动失败：%s", exc)
        return False


def stop_backend(server: object, thread: threading.Thread) -> None:
    """先取消分析任务，再请求 uvicorn 退出并等待线程收尾。"""
    try:
        jobs = server.config.app.state.analysis_jobs  # type: ignore[attr-defined]
        cancelled = jobs.cancel_all()
        if cancelled:
            logger.info("退出前取消 %d 个分析任务", cancelled)
    except Exception:  # noqa: BLE001 - 应用未就绪时没有 jobs
        pass
    try:
        server.should_exit = True  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    thread.join(timeout=SHUTDOWN_TIMEOUT)
    if thread.is_alive():
        logger.warning("uvicorn 线程 %.0fs 内未退出，进程结束时将被回收", SHUTDOWN_TIMEOUT)


def _static_ready(settings: Settings) -> bool:
    index = settings.static_dir / "index.html"
    if index.is_file():
        return True
    logger.warning(
        "未找到前端构建产物 %s，窗口可能显示 404；先运行 scripts/build_frontend 再启动",
        index,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DetectBodyInVideo 桌面版")
    parser.add_argument("--port", type=int, default=None, help="后端端口，默认取配置或自动分配")
    parser.add_argument("--browser", action="store_true", help="强制使用系统浏览器打开")
    parser.add_argument("--no-window", action="store_true", help="只启动服务，不开窗口（调试用）")
    args = parser.parse_args(argv)

    from app.backend.core.logging import setup_logging
    from app.desktop.checks import is_windows, notify_windows, webview2_hint, webview2_status

    ensure_std_streams()
    settings = get_settings()
    setup_logging(settings)

    if is_windows() and not args.browser:
        ready, version = webview2_status()
        if ready:
            logger.info("WebView2 Runtime 已就绪（%s）", version)
        else:
            logger.error(webview2_hint())
            if not args.no_window:
                notify_windows(webview2_hint())

    _static_ready(settings)

    port = args.port or find_free_port(settings.port)
    base_url = f"http://127.0.0.1:{port}"
    server, thread = start_backend(port, settings)

    if not wait_for_health(base_url):
        logger.error("后端健康检查超时：%s", base_url)
        if not args.no_window:
            notify_windows(f"后端服务启动失败，详见日志目录：\n{settings.logs_dir}")
        stop_backend(server, thread)
        return 1
    logger.info("服务就绪：%s", base_url)

    try:
        if args.no_window:
            logger.info("仅服务模式，按 Ctrl+C 退出（%s）", base_url)
            while True:
                time.sleep(1)
        elif args.browser:
            open_browser(base_url)
        else:
            if not run_window(base_url, settings):
                # 窗口创建失败时不要让用户面对空屏：退到浏览器
                notify_windows(
                    "桌面窗口创建失败，已改用系统浏览器。\n"
                    "如需修复桌面模式，请安装 WebView2 Runtime（Windows）"
                    "或 WebKitGTK（Linux）。"
                )
                open_browser(base_url)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出")
    finally:
        stop_backend(server, thread)
        logger.info("已退出")
    return 0


def _report_fatal(exc: BaseException) -> None:
    """窗口模式没有控制台，异常必须落到日志 + 原生消息框，否则表现为「双击没反应」。"""
    logger.exception("桌面壳异常退出：%s", exc)
    try:
        from app.desktop.checks import notify_windows

        notify_windows(f"DetectBodyInVideo 启动失败：\n{exc}\n\n详见日志目录：{get_settings().logs_dir}")
    except Exception:  # noqa: BLE001 - 兜底提示失败不影响退出码
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 打包后无控制台，统一兜底
        _report_fatal(exc)
        raise SystemExit(1) from exc
