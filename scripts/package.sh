#!/usr/bin/env bash
# 打包 macOS / Linux 桌面版（任务 10.2）：前端构建 → 图标 → PyInstaller onedir。
# 用法：./scripts/package.sh [--skip-frontend] [--console] [--clean]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 双击入口必须是可执行文件；这里补一次执行位，避免从压缩包取出后无法双击。
chmod +x "$ROOT/start.sh" "$ROOT/start.command" 2>/dev/null || true
DIST="$ROOT/dist"
WORK="$ROOT/build/pyinstaller"
PYTHON="${PYTHON:-python3}"

SKIP_FRONTEND=0
CONSOLE=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --console) CONSOLE=1 ;;
    --clean) CLEAN=1 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

echo "[package] Python: $($PYTHON --version 2>&1)"
if ! "$PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[package] 安装 pyinstaller"
  "$PYTHON" -m pip install --upgrade "pyinstaller>=6.6"
fi

if [[ "$SKIP_FRONTEND" == "0" ]]; then
  "$ROOT/scripts/build_frontend.sh"
fi

echo "[package] 生成图标"
"$PYTHON" "$ROOT/scripts/make_icon.py"

if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$DIST" "$WORK"
fi

export DETECT_BODY_CONSOLE="$CONSOLE"
echo "[package] PyInstaller（onedir）"
"$PYTHON" -m PyInstaller "$ROOT/packaging/detect-body.spec" \
  --noconfirm --distpath "$DIST" --workpath "$WORK"

if [[ "$(uname -s)" == "Darwin" ]]; then
  TARGET="$DIST/DetectBodyInVideo.app"
else
  TARGET="$DIST/DetectBodyInVideo/DetectBodyInVideo"
fi
if [[ ! -e "$TARGET" ]]; then
  echo "打包失败：缺少 $TARGET（加 --console 重跑可看到详细错误）" >&2
  exit 1
fi
echo ""
echo "[package] 完成：$TARGET"
echo "        模型：首次启动可在界面下载，或放到 ~/.detect-body/models"
