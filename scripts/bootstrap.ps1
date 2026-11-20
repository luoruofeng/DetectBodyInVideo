<#
.SYNOPSIS
    首次运行准备：建 venv、装 Python/前端依赖、下载 MediaPipe 姿态模型。
.EXAMPLE
    ./scripts/bootstrap.ps1
    ./scripts/bootstrap.ps1 -Cuda -SkipModels
#>
param(
    [switch]$Cuda,
    [switch]$SkipFrontend,
    [switch]$SkipModels,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

if (-not (Test-Path $venv)) {
    Write-Host "[bootstrap] 创建虚拟环境 $venv" -ForegroundColor Cyan
    & $Python -m venv $venv
}

$venvPy = Join-Path $venv "Scripts\python.exe"
Write-Host "[bootstrap] 升级 pip" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip wheel

$extras = "dev"
if ($Cuda) { $extras = "dev,cuda" }
$spec = "{0}[{1}]" -f $root, $extras
Write-Host "[bootstrap] 安装 Python 依赖: $spec" -ForegroundColor Cyan
& $venvPy -m pip install -e $spec

if (-not $SkipFrontend) {
    $frontend = Join-Path $root "app\frontend"
    Write-Host "[bootstrap] 安装前端依赖" -ForegroundColor Cyan
    Push-Location $frontend
    try {
        npm install
    }
    finally {
        Pop-Location
    }
}

if (-not $SkipModels) {
    $modelsDir = Join-Path $root "models"
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    $models = [ordered]@{
        "pose_landmarker_lite.task"  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        "pose_landmarker_full.task"  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        "pose_landmarker_heavy.task" = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        "hand_landmarker.task"       = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        "face_landmarker.task"       = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    }
    foreach ($name in $models.Keys) {
        $dest = Join-Path $modelsDir $name
        if (Test-Path $dest) {
            Write-Host "[bootstrap] 已存在 $name" -ForegroundColor DarkGray
            continue
        }
        Write-Host "[bootstrap] 下载模型 $name ..." -ForegroundColor Cyan
        try {
            Invoke-WebRequest -Uri $models[$name] -OutFile $dest -UseBasicParsing
        }
        catch {
            Write-Warning "模型下载失败（$name）：$($_.Exception.Message)。可稍后重试，或手动放到 models/ 目录。"
        }
    }
}

Write-Host ""
Write-Host "[bootstrap] 完成。启动开发环境：" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    ./scripts/dev.ps1"
