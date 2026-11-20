<#
.SYNOPSIS
    一键启动开发环境：后端 uvicorn（热重载）+ 前端 vite。
.EXAMPLE
    ./scripts/dev.ps1
    ./scripts/dev.ps1 -BackendPort 8001
#>
param(
    [int]$BackendPort = 8000,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# 优先使用 .venv 内的解释器
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPy) { $venvPy } else { "python" }

Write-Host "[dev] 项目根目录: $root" -ForegroundColor Cyan
Write-Host "[dev] 后端: http://127.0.0.1:$BackendPort" -ForegroundColor Cyan

$backendArgs = @(
    "-m", "uvicorn", "app.backend.main:app",
    "--host", "127.0.0.1",
    "--port", "$BackendPort",
    "--reload"
)
$backend = Start-Process -FilePath $python `
    -ArgumentList $backendArgs `
    -WorkingDirectory $root `
    -PassThru -NoNewWindow

try {
    if (-not $SkipFrontend) {
        $frontend = Join-Path $root "app\frontend"
        Push-Location $frontend
        try {
            if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
                Write-Host "[dev] 安装前端依赖..." -ForegroundColor Yellow
                npm install
            }
            Write-Host "[dev] 前端: http://127.0.0.1:5173" -ForegroundColor Cyan
            npm run dev
        }
        finally {
            Pop-Location
        }
    }
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Write-Host "[dev] 停止后端进程 $($backend.Id)" -ForegroundColor Yellow
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
