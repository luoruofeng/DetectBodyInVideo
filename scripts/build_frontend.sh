#!/usr/bin/env bash
# 构建前端到 app/backend/static（任务 10.2）
# 用法：./scripts/build_frontend.sh [--clean] [--skip-install]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/app/frontend"
STATIC="$ROOT/app/backend/static"

CLEAN=0
SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

if ! command -v npm >/dev/null 2>&1; then
  echo "未找到 npm，请先安装 Node.js 18+" >&2
  exit 1
fi

cd "$FRONTEND"
if [[ "$CLEAN" == "1" && -d node_modules ]]; then
  echo "[frontend] 清理 node_modules"
  rm -rf node_modules
fi
if [[ "$SKIP_INSTALL" == "0" ]]; then
  if [[ -f package-lock.json ]]; then
    echo "[frontend] npm ci"
    npm ci
  else
    echo "[frontend] npm install"
    npm install
  fi
fi
echo "[frontend] npm run build → $STATIC"
npm run build

if [[ ! -f "$STATIC/index.html" ]]; then
  echo "构建产物缺失：$STATIC/index.html" >&2
  exit 1
fi
echo "[frontend] 完成：$STATIC/index.html"
