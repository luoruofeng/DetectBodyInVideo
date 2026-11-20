#!/usr/bin/env bash
# DetectBodyInVideo 双击启动入口（Linux）。
# 首次使用前需执行一次：chmod +x start.sh
# 文件管理器中可设为「作为程序执行」后双击；终端也可运行：./start.sh [--browser]
cd "$(dirname "$0")" || exit 1

VENV_PY=".venv/bin/python"

pause_on_error() {
  echo
  echo "按回车键关闭窗口..."
  read -r _ || true
}

if [[ -f "dist/DetectBodyInVideo.app/Contents/Info.plist" || -f "dist/DetectBodyInVideo/DetectBodyInVideo" ]]; then
  echo "[提示] 检测到打包产物：dist/DetectBodyInVideo/；打包版无需 Python / Node。"
  echo "        本入口始终从源码启动。"
  echo
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "[1/3] 未检测到虚拟环境，开始首次准备（仅一次，可能需要几分钟）..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 未找到 python3，请先安装 Python 3.9+。"
    pause_on_error
    exit 1
  fi
  if ! bash scripts/bootstrap.sh; then
    echo "[错误] 环境准备失败，请在终端运行 scripts/bootstrap.sh 查看详细日志。"
    pause_on_error
    exit 1
  fi
fi

if [[ ! -f "app/backend/static/index.html" ]]; then
  echo "[2/3] 未检测到前端产物，正在构建（需 Node.js 18+）..."
  if ! bash scripts/build_frontend.sh; then
    echo "[错误] 前端构建失败，请确认 Node.js 18+ 与 WebKitGTK/GTK3 已安装。"
    pause_on_error
    exit 1
  fi
fi

echo "[3/3] 启动 DetectBodyInVideo..."
"$VENV_PY" -m app.desktop.main "$@"
CODE=$?

if [[ "$CODE" -ne 0 ]]; then
  echo
  echo "[提示] 程序退出码 $CODE。日志目录：$HOME/.detect-body/logs"
  pause_on_error
fi

exit "$CODE"
