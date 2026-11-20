#!/usr/bin/env bash
# 首次运行准备：建 venv、装 Python/前端依赖、下载 MediaPipe 姿态模型。
#   用法: ./scripts/bootstrap.sh [--cuda] [--skip-frontend] [--skip-models]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
# 双击入口必须是可执行文件；这里补一次执行位，避免从压缩包取出后无法双击。
chmod +x "$ROOT/start.sh" "$ROOT/start.command" 2>/dev/null || true
EXTRAS="dev"
SKIP_FRONTEND=0
SKIP_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --cuda) EXTRAS="dev,cuda" ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$VENV" ]]; then
  echo "[bootstrap] 创建虚拟环境 $VENV"
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
echo "[bootstrap] 升级 pip"
"$PY" -m pip install --upgrade pip wheel

echo "[bootstrap] 安装 Python 依赖: $ROOT[$EXTRAS]"
"$PY" -m pip install -e "$ROOT[$EXTRAS]"

if [[ "$SKIP_FRONTEND" -eq 0 ]]; then
  echo "[bootstrap] 安装前端依赖"
  ( cd "$ROOT/app/frontend" && npm install )
fi

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  mkdir -p "$ROOT/models"
  BASE="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
  for tier in lite full heavy; do
    FILE="pose_landmarker_${tier}.task"
    DEST="$ROOT/models/$FILE"
    if [[ -f "$DEST" ]]; then
      echo "[bootstrap] 已存在 $FILE"
      continue
    fi
    echo "[bootstrap] 下载模型 $FILE ..."
    curl -fL --retry 2 -o "$DEST" "$BASE/pose_landmarker_${tier}/float16/1/$FILE" || {
      echo "[bootstrap] 下载失败: $FILE（可稍后重试或手动放入 models/）" >&2
      rm -f "$DEST"
    }
  done
  # 手/脸增强模型（默认开启，缺失时增强会自动降级）
  for KIND in hand face; do
    FILE="${KIND}_landmarker.task"
    DEST="$ROOT/models/$FILE"
    if [[ -f "$DEST" ]]; then
      echo "[bootstrap] 已存在 $FILE"
      continue
    fi
    echo "[bootstrap] 下载模型 $FILE ..."
    curl -fL --retry 2 -o "$DEST" "$BASE/${KIND}_landmarker/${KIND}_landmarker/float16/1/$FILE" || {
      echo "[bootstrap] 下载失败: $FILE（可稍后重试或手动放入 models/）" >&2
      rm -f "$DEST"
    }
  done
fi

echo
echo "[bootstrap] 完成。启动开发环境："
echo "    source .venv/bin/activate"
echo "    ./scripts/dev.sh"
