@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title DetectBodyInVideo

echo.
echo   DetectBodyInVideo - 双击启动（源码模式）
echo   ==========================================
echo.

set "VENV_PY=.venv\Scripts\python.exe"

if exist "dist\DetectBodyInVideo\DetectBodyInVideo.exe" (
    echo [提示] 检测到打包程序，也可直接双击 dist\DetectBodyInVideo\DetectBodyInVideo.exe
    echo         打包版无需 Python / Node；本脚本始终从源码启动。
    echo.
)

if not exist "%VENV_PY%" (
    echo [1/3] 未检测到虚拟环境，开始首次准备（仅一次，可能需要几分钟）...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [错误] 未找到 Python。请先安装 Python 3.9+，安装时勾选 "Add Python to PATH"。
        echo.
        pause
        exit /b 1
    )
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
    if errorlevel 1 (
        echo [错误] 环境准备失败，请在终端运行 scripts\bootstrap.ps1 查看详细日志。
        echo.
        pause
        exit /b 1
    )
)

if not exist "app\backend\static\index.html" (
    echo [2/3] 未检测到前端产物，正在构建（需 Node.js 18+）...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_frontend.ps1"
    if errorlevel 1 (
        echo [错误] 前端构建失败，请确认 Node.js 18+ 已安装并加入 PATH。
        echo.
        pause
        exit /b 1
    )
)

echo [3/3] 启动 DetectBodyInVideo...
"%VENV_PY%" -m app.desktop.main %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [提示] 程序退出码 %EXITCODE%。日志目录：%USERPROFILE%\.detect-body\logs
    echo         WebView 未就绪时会降级为系统浏览器；请确认已安装 WebView2 Runtime。
    echo.
    pause
)

exit /b %EXITCODE%
