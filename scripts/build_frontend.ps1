<#
.SYNOPSIS
    构建前端到 app/backend/static（任务 10.2）。
.EXAMPLE
    ./scripts/build_frontend.ps1
    ./scripts/build_frontend.ps1 -Clean
#>
param(
    [switch]$Clean,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "app\frontend"
$static = Join-Path $root "app\backend\static"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "未找到 npm，请先安装 Node.js 18+"
}

Push-Location $frontend
try {
    if ($Clean -and (Test-Path "node_modules")) {
        Write-Host "[frontend] 清理 node_modules" -ForegroundColor DarkGray
        Remove-Item -Recurse -Force "node_modules"
    }
    if (-not $SkipInstall) {
        if (Test-Path "package-lock.json") {
            Write-Host "[frontend] npm ci" -ForegroundColor Cyan
            npm ci
        }
        else {
            Write-Host "[frontend] npm install" -ForegroundColor Cyan
            npm install
        }
    }
    Write-Host "[frontend] npm run build → $static" -ForegroundColor Cyan
    npm run build
}
finally {
    Pop-Location
}

$index = Join-Path $static "index.html"
if (-not (Test-Path $index)) {
    throw "构建产物缺失：$index"
}
Write-Host "[frontend] 完成：$index" -ForegroundColor Green
