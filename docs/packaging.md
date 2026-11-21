# 桌面打包与三平台说明

任务 10 的落地文档：Windows / macOS / Linux 的构建、签名、运行时依赖与常见故障。
打包采用 PyInstaller **onedir**（启动快、动态库易排查），窗口用 pywebview。

## 0. 共同前提

```powershell
# 一次性准备（含前端依赖与姿态模型）
./scripts/bootstrap.ps1
# 构建前端 → app/backend/static
./scripts/build_frontend.ps1
# 打包（会自动构建前端、生成图标、安装 pyinstaller）
./scripts/package.ps1
```

```bash
./scripts/bootstrap.sh
./scripts/build_frontend.sh
./scripts/package.sh
```

产物：

| 平台 | 路径 |
| --- | --- |
| Windows | `dist/DetectBodyInVideo/DetectBodyInVideo.exe` |
| macOS | `dist/DetectBodyInVideo.app` |
| Linux | `dist/DetectBodyInVideo/DetectBodyInVideo` |

调试壳本身时加 `-Console` / `--console`，可看到 PyInstaller 与后端日志。

## 0.1 桌面壳行为（任务 10.1）

`app/desktop/main.py` 是唯一入口，源码与打包产物行为一致：

1. 选端口：优先 `DETECT_BODY_PORT`（默认 8000），被占用则让系统分配空闲端口。
2. 后台线程起 uvicorn（`loop=asyncio`、`http=h11`、`ws=none`，避免打包后动态导入缺失）。
3. 轮询 `GET /api/health`，成功后打开 pywebview 窗口；30s 未就绪则报错退出（退出码 1）。
4. 前端构建产物由后端 `/` 静态挂载，**窗口与浏览器都只访问同一个端口**。
5. 窗口：1440×900、最小 1024×700、标题 `DetectBodyInVideo <版本>`；
   图标 Windows 取自 exe 资源（PyInstaller `icon=`），Linux/macOS 走 `webview.start(icon=...)`。
6. 关闭顺序：窗口关闭 → 取消进行中的分析任务 → `server.should_exit` → 等 uvicorn 线程收尾
   （5s 超时后由进程退出回收）→ 端口释放。后端跑在守护线程里，**没有 Python 子进程**需要杀。
7. 降级链：pywebview 导入失败 → 系统浏览器；窗口创建失败 → 原生提示 + 系统浏览器；
   `--browser` 强制浏览器，`--no-window` 只跑服务（服务器/CI 用）。

打包注意：pywebview 的平台后端是用 `importlib` 动态加载的，spec 里已显式加
`webview.platforms.*` 与 `pythonnet` / `clr_loader` hiddenimports，并收集
`webview/lib/*.dll`（WebView2 托管程序集）。漏掉这些的表现是「打包后只能降级浏览器」。

## 0.2 双击入口（源码模式）

仓库根目录的 `start.bat`（Windows）、`start.command`（macOS）、`start.sh`（Linux）是日常使用的双击入口，走源码模式而非打包产物：

- 缺 `.venv` 时自动调用 `bootstrap.*`（含模型下载，失败不阻断）；
- 缺 `app/backend/static/index.html` 时自动调用 `build_frontend.*`；
- 随后用 `.venv` 里的 Python 运行 `python -m app.desktop.main "$@"`，失败时保留控制台/终端显示日志位置。

macOS / Linux 首次需 `chmod +x start.command`（或 `start.sh`）；`bootstrap.sh` 与 `package.sh` 也会自动补上执行位。Windows 的 `.bat` 有意保留控制台，若需要完全无控制台，请双击 `dist/DetectBodyInVideo/DetectBodyInVideo.exe`。面向分发的原生入口仍是上一节产物表中的路径。

## 1. Windows 10 / 11

- **WebView2 Runtime**：启动时读取注册表 `EdgeUpdate\Clients`（名称含 “WebView2”），
  找不到会弹系统提示并给出下载地址；窗口创建失败会自动降级到默认浏览器。
  离线机器可在镜像中预装 Evergreen Runtime。
- **VC++ 运行库**：MediaPipe / onnxruntime 依赖 `msvcp140.dll`、`vcruntime140.dll`；
  发行版机器若缺失，安装 Microsoft Visual C++ 2015–2022 Redistributable。
- **SmartScreen**：未签名 exe 首次运行会被拦截；内部分发可用自签名证书，
  或在文档中说明“更多信息 → 仍要运行”。
- **数据目录**：`%USERPROFILE%\.detect-body`（模型、导出、断点、日志）。
  打包后模型默认写这里，避免 Program Files 只读。

## 2. macOS（Apple Silicon 优先）

- pywebview 使用系统 WKWebView，无需额外运行时。
- 构建：`./scripts/package.sh`，产物为 `.app`；如需 Universal 二进制，
  在 arm64 机器上分别构建后用 `lipo -create` 合并，或直接在目标架构构建。
- 图标：当前只生成 PNG/ICO；如需 `.icns`，用 `iconutil` 从 iconset 生成后放到
  `app/desktop/assets/icon.icns`，spec 会自动使用。
- 签名与公证：

```bash
codesign --deep --force --options runtime \
  --sign "Developer ID Application: Your Name (TEAMID)" \
  dist/DetectBodyInVideo.app

xcrun notarytool submit dist/DetectBodyInVideo.zip \
  --apple-id you@example.com --team-id TEAMID --password "app-specific-password" --wait
xcrun stapler staple dist/DetectBodyInVideo.app
```

- 无签名分发时，用户首次打开需右键 → 打开，或执行：

```bash
xattr -dr com.apple.quarantine /Applications/DetectBodyInVideo.app
```

## 3. Linux

- pywebview 需要 GTK + WebKitGTK：

```bash
# Debian / Ubuntu
sudo apt install libwebkit2gtk-4.1-0 libgtk-3-0
# Fedora
sudo dnf install webkit2gtk4.1 gtk3
```

- 缺 WebKitGTK 时启动会自动降级为系统浏览器模式（`--browser` 可强制）。
- 建议产物二选一：
  - **AppImage**：用 `linuxdeploy` 打包 `dist/DetectBodyInVideo/` 目录，
    产出单文件，用户 `chmod +x` 后直接运行；需在文档中注明 WebKitGTK 依赖。
  - **.deb**：把 onedir 目录放到 `/opt/detect-body`，写一个 `/usr/bin/detect-body`
    启动脚本，并用 `dpkg-deb --build` 生成安装包。
- 无显示环境（服务器）用 `DetectBodyInVideo --no-window --port 8000` 只跑服务。

## 4. 模型与离线安装

- 内置 MediaPipe 官方直链，可离线预置到 `~/.detect-body/models` 或打包进 `models/`。
- 国内网络可设置镜像前缀（只替换 host，路径保持一致）：

```bash
export DETECT_BODY_MODEL_MIRROR_URL=https://your-mirror.example.com/mediapipe-models
```

- 首启界面会检测模型缺失并显示下载入口（精简 / 完整 / 高精度 + 镜像开关 + 进度）。
- ONNX（RTMPose / RTMO）权重仍由用户自行放入 `models/`，或设置
  `DETECT_BODY_ONNX_POSE_MODEL` 指向具体文件。

## 5. 常见问题

| 现象 | 处理 |
| --- | --- |
| 双击后没窗口 | 用 `-Console` / `--console` 重跑；Windows 先装 WebView2；Linux 装 WebKitGTK |
| 提示缺少 `msvcp140.dll` | 安装 VC++ 2015–2022 Redistributable |
| 打包后 404 / 白屏 | 确认 `app/backend/static/index.html` 存在，先跑 `build_frontend` |
| 模型下载超时 | 配置 `DETECT_BODY_MODEL_MIRROR_URL`，或手动放入 `~/.detect-body/models` |
| 杀毒误报 | PyInstaller 产物常见误报；对 exe 做代码签名或加白名单 |
| 端口被占用 | 桌面壳会自动选空闲端口；服务模式可传 `--port` |

## 6. 验收清单

- [ ] 干净 Windows 10/11（无 Python、无 Node）双击可启动并完成一次导出
- [ ] 无 NVIDIA 的机器能跑 CPU 路径，硬件面板显示 CPU 模式
- [ ] macOS Apple Silicon 首次打开通过 Gatekeeper，CoreML 状态可见
- [ ] Linux 安装 WebKitGTK 后窗口正常，缺失时自动降级浏览器
- [ ] 首启无模型时能完成下载并立即开始分析

---

## 7. 米制（metric）增强的许可决策

对应 `docs/todo-kick-depth-accuracy.md` §7。结论先写：**发行版不内置任何米制权重，
米制只作为"源码模式可选增强"，默认关闭**。

### 7.1 候选模型与许可

| 候选 | 权重许可 | 能否随发行版分发 | 说明 |
| --- | --- | --- | --- |
| VideoPose3D（H36M 微调） | 研究/非商业（Facebook 原仓库声明，数据来自 Human3.6M） | ❌ | 本项目只做 ONNX 导出脚本，权重由使用者自行准备 |
| MotionBERT | 研究/非商业（原仓库声明 + H36M） | ❌ | 同上；`scripts/export_motionbert_onnx.py` |
| AGFormer | 研究/非商业（H36M） | ❌ | 契约与上面两者相同（`LIFTER_FILENAMES` 已预留） |
| RTMPose / RTMO | Apache-2.0（MMPose 生态） | ✅（但体积大） | **已支持**：`DETECT_BODY_ONNX_POSE_MODEL`，属 2D 姿态引擎，不是米制 3D |
| MediaPipe PoseLandmarker | Apache-2.0 | ✅ | 默认引擎；其 `world_landmarks` 只是"以髋为原点的近似米制"，不是真米制 3D |
| MeTRAbs（单图米制 3D） | 代码 MIT；权重由作者发布，SMPL-X 系身体模型需另行注册并遵守其研究许可 | ⚠️ 取决于权重来源 | 精度上最接近"真实厘米"，但**强依赖 SMPL-X 网格与相机内参**；体积约 +300~500 MB（含身体模型），Windows 发行版再叠加 torch 后 ≈ +1.5 GB |
| Multi-HMR / CameraHMR | 研究许可（H36M + 合成数据训练） | ❌ | 单图多人的米制重建；输出也是 SMPL-X 参数，需要同一套身体模型与新渲染链路 |

判定原则：**许可不明的权重一律不进安装包**。收益与风险不对等 —— 内置一次违规分发
的法律成本远高于"让用户自己下一个权重文件"的体验损失。

精度与体积的取舍（同 `docs/todo-kick-depth-accuracy.md` §7.1 的评估结论）：

| 方案 | 能否给出"真实厘米" | 额外依赖 | 结论 |
| --- | --- | --- | --- |
| 现状（MediaPipe + 骨长约束 + 符号先验） | 否（只保证方向与角度） | 无 | 默认路径 |
| 单目抬升器（VideoPose3D / MotionBERT） | 否（root-relative、无尺度，只取符号） | onnxruntime（已有） | 可选，需自备权重 |
| MeTRAbs / Multi-HMR（SMPL-X 系） | 是（需相机内参，弱透视下仍有系统偏差） | torch + SMPL-X 身体模型 | **不立项**：体积 +1.5 GB、许可需注册、对"踢腿幅度"的边际收益不如补拍一个正侧机位 |

### 7.2 三种落地形态与取舍

| 形态 | 用户操作 | 体积影响 | 结论 |
| --- | --- | --- | --- |
| 内置权重 | 无 | Windows onedir 约 +150~300 MB（含 torch 则 +700 MB 以上） | **不采用**（许可不允许） |
| 首启自动下载 | 点一下确认 | 安装包不变；离线机器需手动放 | 可采用，但需镜像与失败兜底；当前未实现 |
| 用户自备 + 源码模式启用 | 放 `models/`，设两个环境变量 | 0 | **当前方案**（见下） |

### 7.3 当前实现：可选 extra + 显式开关

```bash
pip install -e ".[metric]"                      # 只有源码模式才会装 torch
export DETECT_BODY_ENABLE_METRIC_ENGINE=1       # 不设这个变量，引擎根本不注册
export DETECT_BODY_METRIC_MODEL=models/metric-h36m.onnx
```

- `app/backend/engine/metric_engine.py` 目前**只实现契约与降级路径**：缺依赖/缺权重时
  `availability()` 为 False、`prepare()` 抛 `EngineUnavailableError`，`select_engine()`
  自动回落 MediaPipe；即使注册成功，`infer()` 也明确报"推理内核未实现"。
- 打包脚本不安装 `[metric]` extra，`models/` 里也不会出现 `metric-h36m.onnx`，
  因此发行产物的体积与许可风险都不变。
- 深度符号先验的 `model` 后端走的是 ONNX + onnxruntime（`depth_sign.OnnxLifter`），
  **不需要 torch**；缺权重时自动退回 `temporal`，不影响可用性。

### 7.4 商用前的检查清单

- [ ] 确认所选米制模型的许可允许商用（H36M 训练数据的许可要一并确认）
- [ ] 若内置权重：更新本文件与 `README` 的许可段落，并在安装包里附许可原文
- [ ] 若自动下载：加"仅用于研究"的显式确认，并提供关闭入口
- [ ] 重新采三档基线（见 `docs/benchmark.md` 踢腿专项），确认收益值得体积与合规成本


