#!/usr/bin/env bash
# 一键启动开发环境：后端 uvicorn（热重载）+ 前端 vite。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

echo "[dev] 项目根目录: $ROOT"
echo "[dev] 后端: http://127.0.0.1:$PORT"

cd "$ROOT"
"$PY" -m uvicorn app.backend.main:app --host 127.0.0.1 --port "$PORT" --reload &
BACKEND_PID=$!

cleanup() {
  echo "[dev] 停止后端进程 $BACKEND_PID"
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/app/frontend"
if [[ ! -d node_modules ]]; then
  echo "[dev] 安装前端依赖..."
  npm install
fi
echo "[dev] 前端: http://127.0.0.1:5173"
npm run dev
