<#
.SYNOPSIS
    打包 Windows 桌面版（任务 10.2）：前端构建 → 图标 → PyInstaller onedir。
.EXAMPLE
    ./scripts/package.ps1
    ./scripts/package.ps1 -SkipFrontend -Console
#>
param(
    [switch]$SkipFrontend,
    [switch]$Console,
    [switch]$Clean,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"
$work = Join-Path $root "build\pyinstaller"

Write-Host "[package] Python: $(& $Python --version)" -ForegroundColor Cyan
& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[package] 安装 pyinstaller" -ForegroundColor Cyan
    & $Python -m pip install --upgrade "pyinstaller>=6.6"
}

if (-not $SkipFrontend) {
    & (Join-Path $PSScriptRoot "build_frontend.ps1")
}

Write-Host "[package] 生成图标" -ForegroundColor Cyan
& $Python (Join-Path $PSScriptRoot "make_icon.py")

if ($Clean) {
    Remove-Item -Recurse -Force $dist, $work -ErrorAction SilentlyContinue
}

$env:DETECT_BODY_CONSOLE = if ($Console) { "1" } else { "0" }
Write-Host "[package] PyInstaller（onedir）" -ForegroundColor Cyan
& $Python -m PyInstaller (Join-Path $root "packaging\detect-body.spec") `
    --noconfirm --distpath $dist --workpath $work

$exe = Join-Path $dist "DetectBodyInVideo\DetectBodyInVideo.exe"
if (-not (Test-Path $exe)) {
    throw "打包失败：缺少 $exe（加 -Console 重跑可看到详细错误）"
}
Write-Host ""
Write-Host "[package] 完成：$exe" -ForegroundColor Green
Write-Host "        运行：& '$exe'"
Write-Host "        模型：首次启动可在界面下载，或放到 %USERPROFILE%\.detect-body\models"
