# DetectBodyInVideo

DetectBodyInVideo 是一个**本地运行**的跨平台视频人体姿态分析工具：把一段包含人物的普通单目视频拖进界面，逐帧检测 33 个人体关键点，经过跟踪、平滑、补帧、骨长约束、朝向估计与质量检查后，导出可直接驱动 iOS / macOS RealityKit / USDZ 模型的姿态 JSON，或导出 iOS App 可播放的动作 JSON。

它不是云端服务：视频、推理、后处理和导出都在本机完成。程序由三部分组成：

- **Python 后端**：FastAPI + OpenCV + MediaPipe / ONNX Runtime，负责完整分析链路。
- **React 前端**：Vite + TypeScript + Tailwind + three.js，负责上传、预览、参数、时间轴、质检与导出。
- **桌面外壳**：pywebview，负责把本地服务包装成一个窗口；WebView2 / WebKitGTK 不可用时自动降级为系统浏览器。

支持的导出：

| 格式 | 文件后缀 | 主要用途 | 坐标约定 |
| --- | --- | --- | --- |
| 姿态 JSON | `*_pose{granularity}.json` | 复核、二次处理、RealityKit 驱动 | `arkit-right-handed-y-up`，米制 |
| 动作 JSON | `*_motion.json` | iOS App 的 `MotionFrame` 解码器直接播放 | `hip-local-meters-yup`，10 Hz |

> **重要前提**：单目 RGB 视频只能可靠地给出画面内的 `x/y`。前后方向 `z` 是基于人体先验的估计值，不是绝对深度。项目通过骨长约束、2D 重投影拟合和 Z 轴平滑尽量压住误差，但无法把估计变成真实测量。导出文件中始终写有 `depthIsEstimated: true`、`depthConfidence` 与 `warnings`。

---

## 目录

- [1. 先跑起来](#1-先跑起来)
  - [1.5 双击启动入口](#15-双击启动入口)
- [2. 第一次使用：从上传到导出](#2-第一次使用从上传到导出)
- [3. 环境与依赖](#3-环境与依赖)
- [三种运行方式](#三种运行方式)
- [项目结构](#项目结构)
- [端到端处理流程](#端到端处理流程)
- [推理引擎、硬件加速与模型](#推理引擎硬件加速与模型)
  - [应用内 GPU 运行时安装](#应用内-gpu-运行时安装)
- [界面使用说明](#界面使用说明)
- [配置参考](#配置参考)
- [HTTP API](#http-api)
- [命令行与脚本](#命令行与脚本)
- [导出格式](#导出格式)
- [坐标、深度与精度边界](#坐标深度与精度边界)
- [拍摄建议](#拍摄建议)
- [数据目录](#数据目录)
- [测试与基准](#测试与基准)
- [打包与部署](#打包与部署)
- [故障排查](#故障排查)
- [已知限制与维护说明](#已知限制与维护说明)
- [文档索引](#文档索引)
- [许可证](#许可证)

---

## 1. 先跑起来

下面以**开发模式**为例：后端 FastAPI 监听 `127.0.0.1:8000`，前端 Vite 监听 `127.0.0.1:5173`，浏览器打开前端地址，`/api` 请求由 Vite 代理到后端。这是修改代码、调试界面时最常用、也是最推荐的起步方式。

### 1.1 Windows（PowerShell）

在项目根目录执行：

```powershell
# 首次准备：创建 .venv、安装 Python dev 依赖、安装 npm 依赖、下载姿态与手/脸模型
./scripts/bootstrap.ps1

# 并行启动后端（:8000，热重载）和前端（:5173，热更新）
./scripts/dev.ps1
```

打开 <http://127.0.0.1:5173>。可用 `curl http://127.0.0.1:8000/api/health` 确认后端已就绪，正常返回 `{"status":"ok",...}`。

首次准备常用变体：

```powershell
./scripts/bootstrap.ps1 -Cuda                 # NVIDIA 机器，安装 onnxruntime-gpu 与 pynvml
./scripts/bootstrap.ps1 -SkipModels           # 网络不通，稍后在界面或 API 下载模型
./scripts/bootstrap.ps1 -SkipFrontend         # 只准备 Python 环境
./scripts/bootstrap.ps1 -Python "C:\Python311\python.exe"   # 指定 Python 解释器路径
```

开发启动变体：

```powershell
./scripts/dev.ps1 -SkipFrontend               # 只启动后端
./scripts/dev.ps1 -BackendPort 8001           # 后端换端口（见下方注意事项）
```

> **重要**：`vite.config.ts` 里的开发代理目标固定为 `http://127.0.0.1:8000`。`dev.ps1 -BackendPort 8001` 会让后端真的监听 8001，但前端 `/api` 仍代理到 8000，除非你同步修改 `app/frontend/vite.config.ts` 的 `server.proxy['/api'].target`。只想换端口且不改代码时，请直接用后文的**单端口源码模式**或桌面壳。

如果 PowerShell 因执行策略拒绝脚本，可在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1
```

### 1.2 macOS / Linux

```bash
# 首次准备（解释器固定为 python3）
./scripts/bootstrap.sh

# 并行启动后端与前端
./scripts/dev.sh
```

打开 <http://127.0.0.1:5173>。可用变体：

```bash
./scripts/bootstrap.sh --cuda
./scripts/bootstrap.sh --skip-frontend
./scripts/bootstrap.sh --skip-models
PORT=8001 ./scripts/dev.sh          # 同样受 Vite 代理固定 :8000 的限制
```

脚本没有执行权限时先 `chmod +x scripts/*.sh`。

### 1.3 不使用 bootstrap 的手动启动

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -e ".[dev]"        # NVIDIA: -e ".[dev,cuda]"

cd app/frontend
npm install
cd ../..

# 终端 A
python -m uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
# 终端 B
cd app/frontend && npm run dev
```

安装 `python -m pip install -e .` 后还会获得 `detect-body` 命令，它是桌面壳入口，参数见[三种运行方式](#三种运行方式)。

模型可以在启动后由界面下载，也可以由 bootstrap 预先放到仓库根目录 `models/`。本工作区当前已有 `pose_landmarker_lite.task`、`pose_landmarker_full.task`（`heavy` 仍缺失）以及手/脸增强所需的 `hand_landmarker.task`、`face_landmarker.task`；所有 `*.task` 都被 `.gitignore` 排除，新克隆后需要重新下载。

### 1.4 只想用成品，不开发前端

不启动 Vite，先构建前端产物，再由后端单端口托管：

```powershell
./scripts/build_frontend.ps1
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

```bash
./scripts/build_frontend.sh
python3 -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>。前提是 `app/backend/static/index.html` 已生成；`FastAPI` 只在静态目录非空时把它挂载到 `/`。这种模式下没有热更新，改前端需要重新 `build_frontend`。

### 1.5 双击启动入口

不想每次开终端，可以直接双击仓库根目录的入口文件。它们按各平台习惯提供，使用**源码模式**，并在首次运行时自动完成环境准备，之后每次双击直接打开桌面窗口：

| 平台 | 双击文件 | 首次使用 | 运行表现 |
| --- | --- | --- | --- |
| Windows 10/11 | `start.bat` | 无需额外操作 | 保留一个控制台显示准备/启动日志，随后弹出桌面窗口 |
| macOS | `start.command` | 先执行一次 `chmod +x start.command` | 在「终端」中运行并弹出桌面窗口 |
| Linux | `start.sh` | 先执行一次 `chmod +x start.sh` | 文件管理器中设为「作为程序执行」后双击 |

入口脚本的工作顺序：

1. 检查 `.venv`：不存在则调用 `bootstrap.ps1` / `bootstrap.sh` 创建虚拟环境并安装依赖（含模型；模型下载失败不阻断后续启动）。
2. 检查 `app/backend/static/index.html`：不存在则调用 `build_frontend.ps1` / `build_frontend.sh` 构建前端（需 Node.js 18+）。
3. 用 `.venv` 里的 Python 执行 `python -m app.desktop.main`，并原样透传命令行参数（例如终端里执行 `./start.sh --browser`）。
4. 启动失败时保留控制台/终端窗口并显示日志位置，不会“一闪而过”。

> **macOS 执行位**：从压缩包或 Git 取出的 `.command` 默认没有执行位，需在终端执行一次 `chmod +x start.command`。通过 Git 提交时保留执行位，新克隆可免做；若已先运行过 `bootstrap.sh` 或 `package.sh`，这两个脚本也会自动补上执行位。

> **Windows 控制台**：`start.bat` 有意保留控制台便于看到依赖安装与后端报错。若希望完全没有控制台，请用下面的打包产物 `DetectBodyInVideo.exe`。

分发场景请直接双击**打包产物**，它们才是各平台的原生入口（目标机无需 Python / Node）：

| 平台 | 分发入口 | 说明 |
| --- | --- | --- |
| Windows | `dist/DetectBodyInVideo/DetectBodyInVideo.exe` | 默认无控制台；SmartScreen 首次运行需点「更多信息 → 仍要运行」 |
| macOS | `dist/DetectBodyInVideo.app` | 标准 App Bundle；未签名首次打开需右键 → 打开 |
| Linux | `dist/DetectBodyInVideo/DetectBodyInVideo` | 需 GTK3 + WebKitGTK，缺依赖时自动降级浏览器 |

---

## 2. 第一次使用：从上传到导出

下面的步骤按界面实际流程编写。界面布局为：顶部状态栏，左侧视频舞台/播放控制/时间轴，右侧卡片式面板。

### 2.1 先确认模型状态

首次打开且模型清单已返回后，如果任一姿态模型未就绪（或硬件推荐的档位缺失），右侧会显示黄色的「首次运行：需要姿态模型」卡片（推荐档位缺失时标题为「推荐档位模型未就绪」），列出三档模型：

| 档位 | 文件名 | 用途 |
| --- | --- | --- |
| 精简 | `pose_landmarker_lite.task` | CPU 或低配机器优先，速度最快 |
| 完整 | `pose_landmarker_full.task` | 推荐，精度与速度折中 |
| 高精度 | `pose_landmarker_heavy.task` | 精度优先，速度较慢 |

点击对应「下载」即可。下载进度每 600 ms 轮询一次；模型写入 `~/.detect-body/models`（源码运行时也可在仓库 `models/`）。**模型卡是条件显示的**：清单未返回时完全不显示；姿态模型全部就绪且推荐档位也存在、且没有下载任务/错误时卡片才消失。看不到它通常代表模型已就绪。模型缺失也不会禁用上传和识别按钮，但分析时会因引擎不可用而失败或自动降级。

网络受限时，在启动服务前设置镜像前缀，或服务启动后在卡片中勾选「使用镜像下载」（只有服务端配置了 `DETECT_BODY_MODEL_MIRROR_URL` 才真正生效）：

```powershell
$env:DETECT_BODY_MODEL_MIRROR_URL = "https://your-mirror.example.com/mediapipe-models"
```

```bash
export DETECT_BODY_MODEL_MIRROR_URL=https://your-mirror.example.com/mediapipe-models
```

### 2.2 拍摄建议弹窗

首次打开会弹出「开拍前先看一眼：怎么拍，深度才可信」遮罩（即「拍摄建议」弹窗），点「知道了」或按 `Esc` 关闭；关闭状态写入浏览器 `localStorage`，之后不再自动弹出。右侧「拍摄建议」卡片始终存在，默认折叠，可随时展开。

### 2.3 导入视频

把视频拖到窗口任意位置，或点击舞台上的「选择视频文件」/按 `U`。一次只处理拖入的第一个文件。

- 支持扩展名：`.mp4`、`.mov`、`.m4v`、`.avi`、`.mkv`（界面提示主要写 MP4 / MOV / AVI / MKV，但 `.m4v` 也在白名单里）。
- 单文件上限 **8 GiB**；超过 16 MiB 自动走分块上传，≤16 MiB 走一次性上传。块大小由服务端会话决定（默认 8 MiB，服务端限制 256 KiB–32 MiB）。
- 上传进度 <0.95 显示「正在上传视频…」，之后显示「正在解析元信息…」。
- 后端按完整文件 SHA-256 去重；同一内容重复上传会复用已有条目，不会重新探测。
- 编码白名单包含 `h264/hevc/h265/avc1/mp4v/mjpeg/mpeg4/prores`，明确拒绝 `av1/vp9/theora`；未知编码最终以 OpenCV 能否解码首帧为准。
- 导入成功后舞台显示视频、文件名、分辨率、帧率、体积；右侧出现「识别设置」「数据质检」「导出」「3D 预览」「Mesh 白膜预览」卡片；未导入时的「运行环境」+「使用流程」被「视频信息」替换。

### 2.4 设置识别参数

右侧「识别设置」的默认值与含义：

| 设置 | 默认 | 说明 |
| --- | --- | --- |
| 推理引擎 | `auto` 自动选择 | 可选 `auto` / `mediapipe` / `onnx`；芯片显示后端实际选择，降级时带「· 降级」 |
| 采样间隔 | 每 1 帧 | 可选 1/2/3/5/10；目标帧率有值时优先于采样间隔 |
| 目标帧率 | 空 | 填正数如 `15` 后按 `round(源fps / 目标fps)` 推导步长；留空提交 `null` |
| 根运动 | `world` 保留位移 | 用 world 点 + 脚部着地锚定恢复下蹲时的髋部竖直位移；水平/前后位移单目不可恢复 |
| 批处理 | 4 | 可选 1/2/4/8；MediaPipe 不支持 batch，会退化为逐帧 |
| 断点续算 | 开启 | 同时设置 `checkpoint=true` 与 `resume=true` |
| 骨长约束拟合 | 开启 | 仅在所选引擎能输出米制 world landmarks 时生效；ONNX 引擎会置灰并关闭 |
| 深度方向修正 | 关闭 | 单目下每根骨段只剩「往前还是往后」一个比特；开启后用整段动作的时序一致性把它定住（前踢腿不再被解成后踢腿）。需要 ≥30 帧，只在符号反了时纠正，不动 2D 与骨长 |
| 深度方向后端 | `temporal` | 勾选「深度方向修正」后可选：`temporal`（零依赖，默认）/ `model`（ONNX 抬升器）。`model` 在 `models/` 里没有对应权重时**置灰不可选**；运行中若权重缺失或自校准判定无效，会退回 `temporal`，并在质检卡片上显示实际后端与原因 |
| 下肢检测阈值 | 空（用服务级 0.5） | `min_detection_confidence` / `min_tracking_confidence`；下肢在画面里只占十几行像素时，默认 0.5 会把腿判成"没看见"，深度只能靠躯干外推 |
| 下肢重跑 | 跟随服务级（开启） | 下肢平均可见度低于 0.5 且坏帧占比 >35% 时，用 `heavy` 档模型重跑整段；摘要里的 `retryReason` 会写明为什么跑了两遍 |

面板会显示「预计 N 帧」。所有参数在任务运行期间禁用。手/脸增强固定开启（界面无开关）；面板底部还有「清空缓存」按钮，删除断点文件让下次识别从第 1 帧重新开始。

3D 骨骼只使用世界坐标（MediaPipe `world_landmarks`，本身已是米制）：不再提供「归一化 + 身高标定」模式，也不再需要输入身高。导出 JSON 里的 `personHeightMeters` 改为由世界坐标自动测量，只作缩放核对用。

### 2.5 开始识别、看进度、随时取消

点击「开始识别」后前端每 320 ms 轮询任务状态；网络异常时改为 1200 ms 重试。按钮变成「取消任务」，参数全部禁用。

进度有两种口径：

- 推理阶段：`frames_done/frames_total 帧`，并显示处理 fps 与 ETA。
- 后处理阶段：改用真实的 `stage_done/stage_total`，显示 `track → densify → gaps → smooth → hands → yaw → ik_fit → angles → quality` 等阶段（`hands` 是手部 21 点时序稳定），避免推理帧数提前满格显示假 100%。

推理开始后只要已有部分帧落盘，`canPreview` 即为 true（支持边算边看），即使 IK 后处理还在跑，Canvas 骨架叠加也会先显示原始推理结果，并标记「预览已可用」。任务完成后自动切换为平滑/拟合/质检后的最终结果。点击「取消任务」会设置取消事件；推理与后处理都会检查该事件，任务进入 `cancelled`，断点文件会保留以便续算。

### 2.6 查看质检结论

「数据质检」卡片显示：可交付 / 需复核 / 建议重录，有效帧比例、平均置信度、平均可见度、最大跳变、深度置信度、深度抖动、骨长波动、重投影误差、问题列表和运行说明。

深度置信度判读：≥0.75 可信，0.5–0.75 一般，<0.5 不可信；有世界坐标且 `quality.enable_depth_gate=true` 时，低于 0.6 会把导出结论压到最多 `review`。构图和机位问题始终只追加 issues、不会降低 verdict；`enable_framing_rules=false` 时连 issues 都不追加。

### 2.7 导出与下载

完成后右侧「导出」卡片可用，顶部状态栏也有一个等效的导出按钮（条件相同）：

1. 选择格式：`姿态 JSON`（默认）或 `动作 JSON`。
2. 姿态 JSON 再选粒度：`compact` 精简 / `keypoints` 仅关键点 / `full` 完整（默认，推荐）。选 `full` 时前端自动附带原始 33 点。注意界面文案把 `compact` 写成「关键帧 + 角度」、`keypoints` 写成「只有关键点与角度」，但实际**只有 `full` 会导出角度**，详见「导出格式」。
3. 点击「导出 JSON」或「导出动作 JSON」会先弹出系统「另存为」对话框：文件名已按命名规则预填，目录默认打开系统桌面，可自行改名或换位置。取消对话框即放弃导出，不写任何文件。
4. 桌面壳（pywebview）负责这个原生对话框；降级到系统浏览器时没有该能力，会退回到服务端默认目录 `~/.detect-body/exports/`，产物块里出现「下载 JSON」按钮（文件落在默认目录时才会有该按钮）。卡片里的 `artifact.path` 永远是文件的真实落盘路径。
5. 若 `warnings` 非空，导出成功提示和产物块会同时显示深度可信度提示，并写入 JSON 文件。

快捷键：`空格` 播放/暂停，`←/→` 逐帧（`Shift` 十帧），`Home/End` 首尾，`U` 打开文件选择，`E` 导出，`T` 主题，`S` 骨架开关。焦点在输入框/按钮/链接内时全局快捷键不触发；时间轴自身聚焦后 `←/→` 是 ±1 秒、`Shift+←/→` 是 ±5 秒，但事件仍会冒泡到全局处理器叠加一次逐帧步进（时间轴底部提示写的「逐帧」属文案笔误）。

---

## 3. 环境与依赖

### 3.1 运行环境

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| Python | `>=3.9` | 只规定最低版本，不设上限；MediaPipe 1.2+ 提供 `py3-none` 通用 wheel，覆盖 3.9–3.14 |
| Node.js / npm | 18+ | 仅在开发模式、构建前端时需要；打包产物不需要 |
| 磁盘 | 至少预留数 GB | 视频、缓存、模型、断点、导出都保存在本机 |
| 浏览器 | 任意现代浏览器 | 开发模式访问 Vite；桌面壳失败时作为降级入口 |
| Windows 桌面 | WebView2 Runtime、VC++ 2015–2022 Redistributable | 缺 WebView2 会自动降级浏览器；缺 VC++ 可能导致 DLL 报错 |
| macOS 桌面 | 系统 WKWebView | 无需额外运行时；分发建议 codesign / notarize |
| Linux 桌面 | GTK3 + WebKitGTK | Debian/Ubuntu：`libwebkit2gtk-4.1-0 libgtk-3-0`；Fedora：`webkit2gtk4.1 gtk3` |

### 3.2 Python 依赖

来自 `pyproject.toml`：

| 类别 | 依赖 |
| --- | --- |
| Web / API | `fastapi>=0.115`、`uvicorn[standard]>=0.30`、`python-multipart>=0.0.9`、`pydantic>=2.7`、`pydantic-settings>=2.3` |
| 视频 / 数值 | `opencv-python>=4.10`、`numpy>=1.26`、`scipy>=1.11` |
| 推理 | `mediapipe>=0.10.14`、`onnxruntime>=1.18` |
| 桌面 | `pywebview>=5.1` |
| 可选 `cuda` | `onnxruntime-gpu>=1.18`、`pynvml>=11.5` |
| 可选 `windows` | `pynvml>=11.5` |
| 可选 `mac` | 空占位；Apple Silicon 使用 onnxruntime 自带 CoreML EP |
| 可选 `dev` | `pytest>=8.0`、`pytest-cov>=5.0`、`httpx>=0.27`、`ruff>=0.5` |

安装入口：

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,cuda]"   # NVIDIA
detect-body                              # 安装后可用的桌面壳命令
```

仓库中没有 `requirements.txt`，依赖唯一来源是 `pyproject.toml`；wheel 构建后端为 `hatchling`。

### 3.3 前端依赖

来自 `app/frontend/package.json`：

| 类别 | 依赖 |
| --- | --- |
| 运行时 | `react@^18.3.1`、`react-dom@^18.3.1`、`three@^0.170.0`、`@types/three@^0.170.0` |
| 构建 | `typescript@^5.6.3`、`vite@^5.4.11`、`@vitejs/plugin-react@^4.3.4` |
| 样式 | `tailwindcss@^3.4.17`、`postcss@^8.4.49`、`autoprefixer@^10.4.20` |
| 类型 | `@types/react@^18.3.12`、`@types/react-dom@^18.3.1` |

前端 npm 脚本：

```bash
npm run dev        # Vite 开发服务器 :5173
npm run build      # tsc --noEmit && vite build → ../backend/static
npm run preview    # 预览构建产物
npm run typecheck  # 仅 TypeScript 类型检查
```

`app/frontend/package-lock.json` 已提交；`build_frontend.*` 在 lock 文件存在时优先 `npm ci`，否则 `npm install`。

### 3.4 bootstrap 脚本做了什么

| 脚本 | 参数 | 行为 |
| --- | --- | --- |
| `scripts/bootstrap.ps1` | `-Cuda`、`-SkipFrontend`、`-SkipModels`、`-Python` | 建 `.venv`、升级 pip、装 `.[dev]` 或 `.[dev,cuda]`、`npm install`、下载 lite/full/heavy 三档姿态模型及 hand/face 增强模型 |
| `scripts/bootstrap.sh` | `--cuda`、`--skip-frontend`、`--skip-models` | 同上，解释器固定 `python3`，模型用 `curl -fL --retry 2` 下载 |

下载失败时脚本只警告、不中断（bash 版会删除残缺文件）；可稍后在界面、`POST /api/models/download` 或手动放置文件解决。bootstrap 直接写目标文件；界面 / API 的下载管理器才写 `.part` 后原子替换，避免半截文件被当成模型。

### 3.5 模型文件与下载

MediaPipe 姿态模型官方前缀：

```text
https://storage.googleapis.com/mediapipe-models
```

代码内置文件：`pose_landmarker_lite.task`、`pose_landmarker_full.task`、`pose_landmarker_heavy.task`，以及 `hand_landmarker.task`、`face_landmarker.task`。三档姿态模型路径含 `/float16/1/`。

- 查找顺序：显式路径 → `settings.models_dir/<filename>` → 打包资源根 `models/<filename>`。
- 源码运行时默认模型目录是仓库根 `models/`；打包后是 `~/.detect-body/models`，避免 Program Files 只读。
- `DETECT_BODY_AUTO_DOWNLOAD_MODELS=1` 可在缺模型时自动下载；默认关闭。
- 镜像只替换官方 URL 的 host 前缀，路径保持不变。
- ONNX 权重不自动下载：候选名包括 `rtmpose-s_256x192.onnx`、`rtmpose-m_256x192.onnx`、`rtmpose-l_384x288.onnx`、`rtmo-s_640x640.onnx`、`rtmo-m_640x640.onnx` 等，按档位在模型目录查找；也可用 `DETECT_BODY_ONNX_POSE_MODEL` 指向绝对路径。

手部 21 点/手 + 面部 20 个语义点（下颌、下巴、鼻、眼睑、虹膜、嘴角、耳）增强**默认开启**（`AnalysisOptions.refinement`）。逐帧检测后，手按腕点距离匹配到最近的人（超过 0.12 归一化距离则丢弃），脸固定挂到当前帧第一个人，用于补全 AppleModel 全关节；缺 `hand_landmarker.task` / `face_landmarker.task` 或缺依赖时自动降级为纯姿态，主链路不受影响。旧版 mediapipe 不支持 `refine_landmarks` 时虹膜点回退为眼中心。

### 3.6 运行时数据目录

源码开发时模型默认在仓库根 `models/`；其余数据默认在 `~/.detect-body/`（Windows 为 `%USERPROFILE%\.detect-body`）：

```text
~/.detect-body/
├─ library/       # 按 SHA-256 去重的原视频
├─ thumbnails/    # 缩略图 JPEG
├─ temp/
│  └─ uploads/    # 分块上传临时文件
├─ exports/       # 默认导出目录
├─ checkpoints/   # *.pose.jsonl 断点续算文件
├─ logs/          # app.log，轮转 5 MiB × 3
├─ runtime/       # 应用内安装的 CUDA / DirectML 独立后端（按需创建）
├─ webview/       # pywebview 本地存储
└─ models/        # 打包后默认模型目录
```

`DETECT_BODY_DATA_DIR` 可整体迁移；`exports_dir`、`checkpoints_dir`、`logs_dir` 等由 `Settings` 自动派生并在首次启动时创建；`runtime/` 是例外，由应用内安装 GPU 运行时时按需创建。

---

## 三种运行方式

| 方式 | 命令 | 适用场景 | 说明 |
| --- | --- | --- | --- |
| 双击入口 | 根目录 `start.bat` / `start.command` / `start.sh` | 不想开终端 | 源码模式；自动 bootstrap + 前端构建 + 桌面壳，见[1.5 双击启动入口](#15-双击启动入口) |
| 开发模式 | `scripts/dev.ps1` / `.sh` | 改代码、调试 UI | 后端 `--reload` + Vite 热更新；访问 `:5173` |
| 源码单端口 / 桌面模式 | `python -m uvicorn app.backend.main:app` 或 `python -m app.desktop.main` | 本机直接使用 | 需先构建前端，使 `app/backend/static/index.html` 存在；只开一个端口 |
| 打包包 | `scripts/package.ps1` / `.sh` | 分发给未安装 Python / Node 的用户 | PyInstaller onedir；桌面壳与浏览器共用同一端口 |

三条路径的端口关系：

- 开发模式是**两个端口**：Vite `:5173` 负责页面，`/api` 代理到后端 `:8000`。`dev.ps1 -BackendPort` / `dev.sh PORT` 只改后端监听端口，不改 Vite 代理目标；改端口需同步修改 `app/frontend/vite.config.ts`。
- 源码单端口与桌面模式是**一个端口**：FastAPI 同时托管 API 和 `app/backend/static` 前端产物，窗口与浏览器都访问 `http://127.0.0.1:<port>`。
- 打包产物仍是单端口，桌面壳自己在后台线程启动 uvicorn，没有 Python 子进程。

### 源码单端口模式

```powershell
# Windows
./scripts/build_frontend.ps1
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000
```

```bash
# macOS / Linux
./scripts/build_frontend.sh
python3 -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000
```

`uvicorn` 直接运行时不会自动分配空闲端口；端口被占用需自己换 `--port`，或改用桌面壳让它自动挑端口。

### 源码桌面模式

```powershell
# Windows：先构建前端，再启动 pywebview 窗口
./scripts/build_frontend.ps1
python -m app.desktop.main
```

```bash
# macOS / Linux
./scripts/build_frontend.sh
python3 -m app.desktop.main
```

`python -m app.desktop.main` 的行为：

1. 解析 `--port` / `--browser` / `--no-window`。
2. 优先使用 `--port`，否则读 `DETECT_BODY_PORT`（默认 8000）；**只有未显式传 `--port` 时**，配置端口被占用才会让系统分配空闲端口。显式指定且端口被占用会导致绑定失败、健康检查超时并退出码 1。
3. 后台线程启动 uvicorn，固定 `loop=asyncio`、`http=h11`、`ws=none`，`log_config=None`，避免打包后动态导入问题。
4. 轮询 `GET /api/health`，最多等 30 秒；超时则退出码 1 并写日志。
5. 成功后打开 pywebview 窗口；pywebview 不可用或 WebView 运行时缺失则弹原生提示并降级为系统浏览器。
6. 窗口关闭时先取消进行中的分析任务，再停止 uvicorn 线程，最多等待 5 秒。
7. Windows 启动时会查 WebView2 注册表/文件系统，缺失时提示下载地址，但不会阻止服务启动。

桌面壳命令行：

```text
python -m app.desktop.main [--port PORT] [--browser] [--no-window]

--port        指定后端端口；默认取配置或自动分配
--browser     强制使用系统浏览器，不创建 pywebview 窗口
--no-window   只启动服务，不打开窗口；适合服务器 / CI
```

如果通过 `pip install -e .` 安装，也可以直接运行 `detect-body`，参数相同。

> **配置注意**：桌面壳的健康检查路径硬编码为 `/api/health`，因此不要在非开发模式下改 `DETECT_BODY_API_PREFIX`；改了会导致桌面壳健康检查超时。前端 Vite 代理同样硬编码 `/api`。

### 无窗口服务 / 局域网部署

```powershell
$env:DETECT_BODY_HOST = "0.0.0.0"
$env:DETECT_BODY_PORT = "8000"
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
# 或源码桌面壳
python -m app.desktop.main --no-window --port 8000
```

```bash
DETECT_BODY_HOST=0.0.0.0 python3 -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

仓库没有内置认证、token、TLS 或反向代理配置；默认 CORS 只放行 `http://127.0.0.1:5173` 和 `http://localhost:5173`。单端口托管时浏览器与 API 同源，不受 CORS 限制。**不要把服务直接暴露到公网**；局域网使用请自行加反向代理、访问控制与 TLS。以 `0.0.0.0` 绑定时，桌面壳本身仍固定用 `127.0.0.1` 启动 uvicorn，需用 `--no-window` + 外部浏览器访问。

---

## 项目结构

```text
DetectBodyInVideo/
├─ app/
│  ├─ backend/
│  │  ├─ main.py                 # FastAPI 工厂 create_app()、全局 app 单例、静态目录挂载
│  │  ├─ pipeline.py             # analyze_video()：端到端主链与 AnalysisOptions
│  │  ├─ jobs.py                 # AnalysisJobManager：后台任务、进度、取消、结果
│  │  ├─ api/                    # HTTP 路由
│  │  │  ├─ analysis.py          # 任务、预览帧、导出
│  │  │  ├─ exports.py           # schema 存档与导出文件下载
│  │  │  ├─ hardware.py          # 硬件报告（60 秒缓存）
│  │  │  ├─ health.py            # /api/health
│  │  │  ├─ models.py            # 模型清单与下载
│  │  │  ├─ runtime.py           # GPU 加速运行时安装 / 切换 / 重启
│  │  │  └─ videos.py            # 视频列表、上传、缩略图、采样计划
│  │  ├─ core/
│  │  │  ├─ settings.py          # Settings、路径派生、.env、DETECT_BODY_ 环境变量
│  │  │  ├─ errors.py            # 统一应用错误
│  │  │  └─ logging.py           # 日志初始化
│  │  ├─ engine/
│  │  │  ├─ base.py              # PoseFrame / Landmark / BoundingBox / PoseEngine 抽象
│  │  │  ├─ registry.py          # 引擎注册、select_engine()、自动降级
│  │  │  ├─ mediapipe_engine.py  # MediaPipe PoseLandmarker
│  │  │  ├─ onnx_engine.py       # HOG 粗检 + ONNX + SimCC/回归解码 + 33 点映射
│  │  │  ├─ acceleration.py      # 硬件报告与 EP 优先级
│  │  │  ├─ hardware.py          # CPU/GPU/内存/ONNX EP 探测
│  │  │  ├─ tracking.py          # IouTracker、轨迹筛选
│  │  │  ├─ consistency.py       # 帧间/引擎间一致性比较
│  │  │  ├─ enhancements.py      # 手部 / 面部增强
│  │  │  ├─ models.py            # 模型 URL、发现、镜像、下载
│  │  │  ├─ model_downloads.py   # 模型下载管理器与进度
│  │  │  └─ accelerator_install.py # 应用内安装 CUDA / DirectML 独立运行时
│  │  ├─ media/
│  │  │  ├─ validation.py        # 扩展名 / 编码 / 文件有效性校验
│  │  │  ├─ probe.py             # 视频探测与缩略图
│  │  │  ├─ reader.py            # 逐帧读取与 FramePacket
│  │  │  ├─ sampling.py          # 采样计划
│  │  │  ├─ store.py             # 视频库、SHA-256 去重
│  │  │  └─ uploads.py           # 分块上传会话
│  │  ├─ perf/
│  │  │  ├─ config.py            # RuntimeConfig：队列、batch、分段、续算
│  │  │  ├─ runner.py            # 生产/消费推理循环
│  │  │  └─ checkpoint.py        # JSONL 断点读写
│  │  ├─ postprocess/
│  │  │  ├─ pipeline.py          # 后处理编排
│  │  │  ├─ one_euro.py          # One Euro 滤波，含 Z 轴独立参数
│  │  │  ├─ smoothing.py         # 移动平均 / Savitzky–Golay
│  │  │  ├─ gaps.py              # 补帧与 densify
│  │  │  ├─ angles.py            # 关节角度
│  │  │  ├─ ik_fit.py            # 固定骨长 + 2D 重投影拟合
│  │  │  ├─ hands.py             # 手部 21 点时序稳定（补帧 / 迟滞 / 平滑）
│  │  │  └─ quality.py           # 质量报告与 depthConfidence
│  │  ├─ skeleton/
│  │  │  ├─ mapping.py           # MediaPipe 33 点 → ARKit 关节
│  │  │  ├─ topology.py          # 骨段拓扑与人体测量学范围
│  │  │  ├─ orientation.py       # yaw 估计
│  │  │  ├─ transform.py         # 坐标系常量与转换
│  │  │  ├─ root_motion.py       # 根运动与平移/旋转拆分
│  │  │  └─ bundle.py            # SkeletonBundle 构建
│  │  ├─ export/
│  │  │  ├─ schema.py            # 姿态 JSON Pydantic 契约，SCHEMA_VERSION = 1.1.0
│  │  │  ├─ builder.py           # AnalysisResult → ExportDocument
│  │  │  ├─ writer.py            # 写盘、回读校验
│  │  │  ├─ motion_clip.py       # 动作 JSON 构建、CLI
│  │  │  └─ samples.py           # 确定性 golden 样例生成器
│  │  └─ static/                 # Vite 构建产物；存在时由 FastAPI 挂载到 /
│  ├─ desktop/
│  │  ├─ main.py                 # pywebview / 浏览器 / 无窗口三种模式
│  │  ├─ checks.py               # WebView2 状态与系统提示
│  │  └─ assets/                 # 图标生成目标目录
│  └─ frontend/
│     ├─ src/main.tsx            # React 挂载入口
│     ├─ src/App.tsx             # 页面组合与全局快捷键
│     ├─ src/api/                # HTTP 客户端与类型
│     ├─ src/hooks/              # 上传、任务轮询、播放、硬件、模型、主题、GPU 安装
│     ├─ src/components/         # 上传区、视频舞台、时间轴、面板、3D/Mesh 预览等
│     ├─ src/lib/                # 格式化与姿态辅助
│     ├─ src/store/              # 全局状态 AppProvider
│     └─ vite.config.ts          # 开发代理 /api → :8000；构建到 backend/static
├─ docs/                         # 打包、基准、JSON Schema、动作 JSON、深度精度说明
├─ models/                       # 本地模型权重；*.task 被 gitignore，本工作区有 lite/full + hand/face，heavy 需下载
├─ packaging/
│  └─ detect-body.spec           # PyInstaller onedir 配置
├─ samples/
│  ├─ expected_pose.json         # golden 导出样例
│  ├─ RealityKitSample.swift     # iOS / macOS 消费示例
│  └─ benchmark/                 # 深度基准素材清单
├─ resources/                    # 本地参考模型/样例（gitignore；本工作区已有 AppleModel.usda 等）
├─ scripts/                      # bootstrap / dev / build / package / bench / icon
├─ start.bat                     # Windows 双击启动入口（源码模式）
├─ start.command                 # macOS 双击启动入口（源码模式）
├─ start.sh                      # Linux 双击启动入口（源码模式）
├─ tests/                        # 23 个测试模块 + conftest + 合成姿态工具
├─ pyproject.toml                # Python 依赖、CLI、pytest 与 ruff 配置
└─ TODO.md                       # 历史任务清单，进度百分比可能滞后于代码
```

不属于第一方源码、不要手工维护的目录 / 文件：

- `app/frontend/node_modules/`：npm 第三方依赖。
- `app/backend/static/`：Vite 构建产物。
- `app/backend/export/schema.json`：由 `python -m app.backend.export.schema` 生成。
- `samples/expected_pose.json`：由 `python -m app.backend.export.samples` 生成。
- `models/*.task`、`samples/*.mp4|mov|m4v`、`resources/`：本地大文件 / 权重，按 `.gitignore` 排除；本工作区的 `resources/` 已有 `AppleModel.usda`、`demo.mp4`、`air-squat-motion.json`，新克隆需自行放置。
- `dist/`、`build/`、`__pycache__/`、`*.pyc`：生成物。

---

## 端到端处理流程

`app/backend/pipeline.py::analyze_video()` 串起全部步骤：

```text
视频上传/校验
    ↓
VideoProbe 探测：fps / 尺寸 / 旋转 / 编码 / VFR
    ↓
SamplingConfig → AnalysisPlan：stride、目标 fps、时间范围、帧数
    ↓
硬件报告 → 加速计划 → select_engine()：MediaPipe 或 ONNX，必要时降级
    ↓
可选手/脸增强：准备 HandLandmarker / FaceLandmarker，逐帧 refine
    ↓
run_inference()：解码线程 → 有界队列 → 推理线程 → PoseFrame[]
    ↓
run_postprocess()：轨迹筛选 → densify → 补帧 → 平滑 → 手部稳定 → yaw → IK → 角度/质检
    ↓
build_skeleton_bundle()：33 点映射 ARKit、坐标转换、root motion
    ↓
AnalysisResult → 预览帧 / 姿态 JSON / 动作 JSON
```

每一步的细节：

1. **上传与入库**  
   `POST /api/videos/upload` 或分块上传接口将文件写入临时目录，计算 SHA-256，校验扩展名和编码，再落入视频库。重复上传同一内容会复用已有条目。

2. **视频探测**  
   `media/probe.py` 读取容器与流信息，生成缩略图。`SamplingConfig` 结合目标 FPS、开始/结束时间和 stride 生成 `AnalysisPlan`，界面可先用 `POST /api/videos/{id}/plan` 预览会分析多少帧。

3. **引擎选择与增强准备**  
   `engine/acceleration.py` 采集 CPU / GPU / 内存 / ONNX EP；`registry.py` 根据 `pose_engine` / `prefer_model_tier` / 运行时计划选择引擎。不可用时记录原因并回退，不会直接中断。`refinement.hands/face` 默认开启，会再准备 `MediaPipeRefiner` 逐帧补手/脸点；缺模型或依赖时静默降级为纯姿态。

4. **推理运行时**  
   `perf/runner.py` 用一个生产者线程解帧并写入有界队列（`runtime_queue_size`），消费者调用 `engine.infer_batch()`。不支持 batching 的引擎自动退化为 1。`runtime_segment_frames > 0` 时按段推进进度。开启续算后，每帧结果写入 `~/.detect-body/checkpoints/*.pose.jsonl`，中断后重跑会跳过已完成帧。取消任务会停止线程并调用 `engine.close()` 释放资源。

5. **后处理**  
   `postprocess/pipeline.py` 的实际顺序：track 筛选 → `densify` → 缺失帧补齐 → 平滑 / 世界点单独平滑 → 手部时序稳定（`hands`）→ yaw → IK 拟合 → 角度与质检。`legacy_postprocess_config()` 可恢复改造前行为，便于 A/B。

6. **骨骼与坐标**  
   `skeleton/mapping.py` 把 MediaPipe 的 33 点映射到 ARKit 关节；`transform.py` 把米制、`y` 向下的世界坐标转换成米制、右手系、`y` 向上、`-z` 向前的导出坐标；`root_motion.py` 生成根平移与四元数旋转。

7. **结果消费**  
   `GET /api/analysis/{id}/frames` 分页返回预览帧给 Canvas 叠加层和 3D 预览；`POST /api/analysis/{id}/export` 生成姿态 JSON 或动作 JSON。

`AnalysisOptions` 是分析入口的核心参数对象：

| 字段 | 作用 |
| --- | --- |
| `sampling` | 采样计划：目标 FPS、stride、时间范围 |
| `postprocess` | 平滑 / 补帧 / 手部稳定 / 角度 / 质检 / IK 配置 |
| `skeleton` | 关节粒度（`include_optional` / `include_all_joints`）、`root_motion_mode`、起点归零、脚部着地锚定；无 `world/calibrated` 坐标模式，也不做镜像 |
| `perf` | 队列长度、batch、分段、checkpoint / resume |
| `refinement` | 手/脸增强：`hands=true`、`face=true`，缺模型自动降级 |
| `ikFit` | 骨长拟合快捷覆盖；为空时取 `postprocess.ik_fit` |
| `engine` | `auto` / `mediapipe` / `onnx` |
| `max_frames` | 调试用，最多分析多少帧 |

---

## 推理引擎、硬件加速与模型

### 硬件与 EP

`GET /api/hardware` 返回 CPU、内存、GPU、ONNX Execution Providers 等信息，结果缓存 60 秒，`?refresh=true` 可强制重新探测。EP 优先级：

```text
TensorRT > CUDA > CoreML > DirectML > CPU
```

`auto` 模式根据探测结果选择：命中前四者视为 GPU 加速，优先尝试 ONNX；否则优先 MediaPipe。档位策略：CPU 强制 `lite`；Apple Silicon 统一内存 ≥16 GB 用 `heavy`、否则 `full`；其它 GPU 显存 ≥8000 MB 用 `heavy`、≥4000 MB 用 `full`、显存未知用 `full`。显式设置 `DETECT_BODY_PREFER_MODEL_TIER=lite|full|heavy` 时该值优先级最高，直接决定档位；只有它为 `auto` 时才按上面的 CPU / Apple Silicon / 显存规则推断。

界面「硬件与加速」卡片始终显示，展开后可看 CPU / 内存 / GPU / EP 和选择理由，并能点「重新探测」。NVIDIA 机器需安装 `.[cuda]` 才能启用 CUDA EP；Apple Silicon 使用 onnxruntime 自带 CoreML EP；Windows 的 WebView2 只影响窗口，不影响推理。

### 引擎选择与降级

- 请求优先级：任务里的 `engine` > `DETECT_BODY_POSE_ENGINE` > `auto`；非法字符串静默改为 `auto`。
- 显式指定 `mediapipe` 或 `onnx` 时，另一个引擎仍会作为备选；因此指定失败会自动降级，界面芯片会显示「· 降级」。
- 候选依次执行 `availability → create → apply_runtime → prepare`；任一失败记录原因并换下一个。全部失败才抛 `EngineUnavailableError`，错误详情含 `requested/candidates/reasons/hint`。
- `canPreview` 与 `canExport` 是两个不同能力：推理开始后只要有部分帧即可预览（边算边看），后处理/质检完成才可导出。

### MediaPipe 引擎

`MediaPipePoseEngine` 使用 MediaPipe Tasks 的 `PoseLandmarker`，统一输出 33 点并自带米制 `world_landmarks`：

| 档位 | 文件 | 说明 |
| --- | --- | --- |
| lite | `pose_landmarker_lite.task` | 速度优先，CPU 自动档位 |
| full | `pose_landmarker_full.task` | 默认折中，界面推荐 |
| heavy | `pose_landmarker_heavy.task` | 精度优先、速度较慢 |

本工作区当前可见 lite / full 姿态模型及 `hand_landmarker.task` / `face_landmarker.task` 增强模型；`.task` 文件被 `.gitignore` 排除，不会随仓库分发。新克隆后由 bootstrap、模型卡、`POST /api/models/download` 或手动放置获取，heavy 需自行下载。

重要细节：

- `smooth_landmarks=true` 时进入 `VIDEO` 模式，否则 `IMAGE`；MediaPipe 引擎本身不读取其 `tracking_ids`，但 `perf/runner.py` 会给**所有引擎**统一分配 `track_id`（单人恒为 1，多人用 IoU 跟踪），因此后处理轨迹筛选照常生效。
- `delegate` 默认 CPU；只有 `DETECT_BODY_MEDIAPIPE_GPU_DELEGATE=true` 才请求 GPU，创建失败会记录 note 并回退 CPU。
- 不支持 batching；`runtime.batch_size` 会退化为逐帧。
- 手部/面部增强默认开启：手输出 21 点、脸只保留 20 个语义点（非 468 点）；手按腕点距离匹配到最近的人（超阈值丢弃），脸挂到当前帧第一个人。

### ONNX 引擎

`OnnxPoseEngine` 的处理链：

```text
HOG 人体粗检 → 裁剪 / 仿射变换 → ONNX 推理 → SimCC / 回归解码
    → COCO17 映射到 MediaPipe 33 点 → IoU 跟踪
```

- HOG 阈值 0.35，宽度 >512 先缩到 512，`winStride=(8,8)`、`padding=(8,8)`、`scale=1.05`；未检出时退回整帧框。
- 输入默认 `(192,256)`（W,H），BGR→RGB、ImageNet mean/std、NCHW float32；支持 SimCC 双输出和单输出回归。
- 只接受 COCO 17 点模型，其余 16 个 MediaPipe 索引是占位 0；真正的脊柱/头部等关节在骨骼映射层推导。
- `world_landmarks=[]`、`z=0`：**ONNX 引擎没有深度和米制世界坐标**，因此不产出 3D 关节（3D 预览为空）；需要 3D 骨骼请用 MediaPipe。
- 多人能力由模型文件名是否含 `rtmo` 决定，不是看检测结果；RTMPose 单人支持 batching，RTMO 多人逐帧检测 + IoU 跟踪。
- Session 创建失败会自动回退 `CPUExecutionProvider`；`fp16=auto` 只在 CUDA / TensorRT / CoreML 上尝试 `*.fp16.onnx` 或 `*_fp16.onnx`，DirectML 不算，找不到则保持 FP32。
- 候选权重按档位查找：`rtmpose-s_256x192.onnx` / `rtmpose_t_256x192.onnx` / `rtmpose-s.onnx`，`rtmpose-m_256x192.onnx` / `rtmo-s_640x640.onnx` 等；可设 `DETECT_BODY_ONNX_POSE_MODEL` 指向绝对路径，但该路径不存在时直接报错，不会继续搜索。

### 模型下载与镜像

- 官方 base URL：`https://storage.googleapis.com/mediapipe-models`，镜像只替换 host，路径保持一致。
- 自动下载：`DETECT_BODY_AUTO_DOWNLOAD_MODELS=1`，默认关闭。
- 界面 / API：模型卡或 `POST /api/models/download`，请求体支持 `kind`（`pose`/`hand`/`face`）、`tier`（`lite`/`full`/`heavy`）、`use_mirror`。勾选 `use_mirror=false` 会强制官方直链，即使服务端配置了镜像。
- 下载先写 `.part`，256 KiB 分块、60 秒超时，完成后原子替换；同一文件同时只跑一个下载任务（同名任务复用），不同文件可并发。
- `GET /api/models` 返回模型清单、就绪状态、`mirrorConfigured` 和最近 6 条下载任务；`poseReady` 只要求任意一档姿态模型存在。
- 源码运行时模型目录默认是仓库根 `models/`；打包后默认是 `~/.detect-body/models`。

### 应用内 GPU 运行时安装

除了用 `bootstrap.ps1 -Cuda` / `bootstrap.sh --cuda` 在开发环境预装 CUDA 版 onnxruntime，应用内还提供一套**免管理员权限**的 GPU 运行时安装流程（后端 `engine/accelerator_install.py`，API `/api/runtime/*`，前端嵌在「硬件与加速」卡片里）：

- 检测到 GPU 但当前没有可用加速 EP 时，卡片会列出可安装变体：`cuda`（`onnxruntime-gpu>=1.22` + cuDNN/cuBLAS 12，约 1.3 GB，NVIDIA 优先）与 `directml`（`onnxruntime-directml>=1.20`，约 350 MB，Windows 上 AMD/Intel 也可用）。不满足条件（如 CUDA 版本低于 12）的变体会显示阻塞原因并禁用安装。
- 安装用 `pip --target` 写入 `~/.detect-body/runtime/<variant>` 独立目录，**不改动当前 Python 环境**；已有 CPU 版始终保留作兜底，删除目录即可回滚。前端 700 ms 轮询安装进度，显示百分比、日志尾部，可取消安装；单次 pip 尝试 10 分钟超时，最多 3 次。
- 安装完成后提示「重启应用后生效」：onnxruntime 的动态库已被当前进程加载，无法热切换，需点「重启应用」（`POST /api/runtime/restart`，不支持时显示原因）或手动重启。已安装但未激活的运行时可「切换」/「移除」；移除会回滚到 CPU 版。
- `GET /api/runtime/accelerators` 返回可选项、已安装项、当前激活变体、当前 EP、`gpuAccelerated`、`restartRequired` 与安装任务状态；相关 API 见「HTTP API」。

---

## 界面使用说明

第 2 节已按时间顺序走完一次完整流程；本节按界面元素查阅。

### 侧栏卡片的显示条件

| 卡片 | 显示条件与内容 |
| --- | --- |
| 模型卡 | 清单已返回但姿态模型缺失 / 推荐档位缺失（标题变「推荐档位模型未就绪」）/ 下载中 / 出错时显示；只列 lite/full/heavy 三档姿态模型，清单未返回时反而不显示 |
| 拍摄建议 | 始终显示，默认折叠；首次打开另有一次性弹窗 |
| 识别设置 + 数据质检 | 导入视频后出现；两个 section 合并在同一卡片组内；底部有「清空缓存」按钮 |
| 导出 | 导入视频后出现；按钮只有任务 `done` 时可用；顶部状态栏也有等效导出按钮 |
| 3D 预览 | 导入视频后出现；用 ARKit 关节渲染，可拖拽旋转、滚轮缩放，显示地面网格与坐标轴 |
| Mesh 白膜预览 | 导入视频后出现；Three.js Marching Cubes 把关键点展开为带胸腔/四肢的人形隐式曲面，头颅与五指单独建模，可拖拽旋转、滚轮缩放；白膜是姿态驱动的近似体型，不代表真实外观 |
| 视频信息 / 运行环境 | 未导入视频时显示运行环境（版本、Python、系统、服务状态）与使用流程；导入后换成视频信息，使用流程消失 |
| 硬件与加速 | 始终显示；可点「重新探测」；GPU 未启用加速时内嵌「GPU 加速」安装区（见「应用内 GPU 运行时安装」） |
| 使用流程 | 仅在未导入视频时显示，引导上传 → 识别 → 导出 |

### 视频舞台与骨架叠加

- 舞台按源视频宽高比自适应；Canvas 叠加 MediaPipe 33 点和标准连线，与视频像素坐标对齐。
- 默认：骨架开、关键点编号关、检测框开、连线实线。`visibility < 0.05` 的点不画；按可见度分三档颜色。
- 鼠标靠近关键点 14 px 内显示中文点位名、归一化 x/y/z 与置信度；检测框为青色虚线。
- 关闭「骨架」后检测框也不会绘制；「编号」开关在窄屏隐藏但状态仍有效。
- 骨架取「最后一个 `t <= currentTime`」的帧，不是绝对时间最近的帧；`valid=false` 时左下角提示「本帧无有效姿态（valid=false）」。
- 3D 预览使用米制 ARKit 关节，跟随主题调整骨骼颜色，显示地面网格与坐标轴。

### 播放与时间轴

- 真正驱动 `<video>` 播放，支持 `0.25× / 0.5× / 1× / 1.5× / 2×`，默认 `1×`。
- 逐帧按钮先暂停，再按 `1 / video.fps` 秒步进。
- 时间轴用 Canvas 绘制：青色进度、白色播放头、青色 A/B 循环区间、红色刻度表示缺失帧；支持拖拽 seek 与悬停时间提示。
- A/B 循环在当前时间标记起点/终点，播放头越过 B 时回到 A；有循环时才出现「清除循环」。换视频会清空循环并跳回 0 秒。

### 快捷键

| 按键 | 功能 |
| --- | --- |
| `空格` | 有视频时播放 / 暂停 |
| `←` / `→` | 有视频时后退 / 前进 1 帧 |
| `Shift + ←/→` | 后退 / 前进 10 帧 |
| `Home` / `End` | 跳到首帧 / 末帧 |
| `U` | 打开文件选择 |
| `E` | 按当前设置导出；未完成分析时弹提示 |
| `T` | 切换主题 |
| `S` | 显示 / 隐藏骨架 |

`Ctrl/Cmd/Alt` 组合键不触发全局快捷键；焦点在 `input/textarea/select/button/a/contenteditable` 内也不触发。时间轴自身聚焦后 `←/→` 是 ±1 秒、`Shift+←/→` 是 ±5 秒（事件仍冒泡到全局处理器，会再叠加一次逐帧步进）。右侧栏宽度可拖拽调整、双击分隔条复位，宽度写入 `localStorage['detect-body-side-panel-width']`。主题默认深色，切换写入 `localStorage['detect-body-theme']`，不跟随系统主题。拍摄建议弹窗关闭状态写入 `localStorage['detect-body:shooting-guide-seen']`。

---

## 配置参考

配置分两层：

1. **服务级配置**：`app/backend/core/settings.py::Settings`，通过 `DETECT_BODY_` 前缀环境变量或仓库根 `.env` 设置，影响服务、路径、日志、默认引擎参数。
2. **单次分析配置**：`POST /api/analysis` 的 `options`，由界面「识别设置」或 API 请求体传入，只影响该任务。

### 服务级配置（环境变量 / .env）

所有字段都可通过 `DETECT_BODY_` + 大写字段名覆盖，例如：

```text
DETECT_BODY_PORT=8000
DETECT_BODY_RUNTIME_BATCH_SIZE=8
DETECT_BODY_IK_FIT_ENABLED=0
```

也可在仓库根创建 `.env`（UTF-8，未被提交，被 `.gitignore` 排除）：

```dotenv
DETECT_BODY_HOST=127.0.0.1
DETECT_BODY_PORT=8000
DETECT_BODY_LOG_LEVEL=DEBUG
DETECT_BODY_DATA_DIR=D:/detect-body-data
DETECT_BODY_MODEL_MIRROR_URL=https://mirror.example.com/mediapipe-models
DETECT_BODY_AUTO_DOWNLOAD_MODELS=1
DETECT_BODY_ONNX_POSE_MODEL=D:/models/rtmpose-m_256x192.onnx
```

`Settings` 使用 `lru_cache(maxsize=1)`，首次调用时创建所有派生目录；改完 `.env` 需重启服务。未知配置项会被忽略。

#### 服务与路径

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DETECT_BODY_APP_NAME` | `DetectBodyInVideo` | 应用名，出现在健康检查和窗口标题 |
| `DETECT_BODY_VERSION` | `0.1.0` | 版本号 |
| `DETECT_BODY_HOST` | `127.0.0.1` | `uvicorn` 监听地址 |
| `DETECT_BODY_PORT` | `8000` | 服务端口；桌面壳占用时自动换 |
| `DETECT_BODY_API_PREFIX` | `/api` | API 路由前缀；**不要改**，桌面壳和 Vite 都硬编码 `/api` |
| `DETECT_BODY_CORS_ORIGINS` | `http://127.0.0.1:5173, http://localhost:5173` | CORS 白名单；单端口模式无需 CORS |
| `DETECT_BODY_DATA_DIR` | `~/.detect-body` | 上传、导出、断点、日志根目录 |
| `DETECT_BODY_MODELS_DIR` | 源码：`models/`；打包：`~/.detect-body/models` | 模型目录 |
| `DETECT_BODY_STATIC_DIR` | `app/backend/static` | 前端构建产物目录 |
| `DETECT_BODY_MODEL_MIRROR_URL` | 空 | MediaPipe 模型镜像前缀 |
| `DETECT_BODY_AUTO_DOWNLOAD_MODELS` | `false` | 缺模型时是否自动下载 |
| `DETECT_BODY_ONNX_POSE_MODEL` | 空 | ONNX 权重绝对路径 |

#### 日志、上传与视频

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DETECT_BODY_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`；日志同时写控制台与 `~/.detect-body/logs/app.log`（5 MiB × 3） |
| `DETECT_BODY_LOG_MAX_BYTES` | `5242880` | 单个日志文件上限 |
| `DETECT_BODY_LOG_BACKUP_COUNT` | `3` | 日志轮转保留数 |
| `DETECT_BODY_MAX_UPLOAD_BYTES` | `8589934592` | 单文件上传上限（8 GiB） |
| `DETECT_BODY_CHUNK_SIZE` | `8388608` | 分块上传目标块大小（8 MiB），服务端钳制 256 KiB–32 MiB |
| `DETECT_BODY_ALLOWED_EXTENSIONS` | `.mp4,.mov,.m4v,.avi,.mkv` | 扩展名白名单 |
| `DETECT_BODY_ALLOWED_CODECS` | `h264,hevc,h265,avc1,mp4v,mjpeg,mpeg4,prores` | 记录的编码白名单；当前校验仅用 `reject_codecs` 明确拒绝 |
| `DETECT_BODY_REJECT_CODECS` | `av1,vp9,theora` | 明确拒绝的编码 |
| `DETECT_BODY_THUMBNAIL_MAX_WIDTH` | `480` | 缩略图最大宽度 |
| `DETECT_BODY_UPLOAD_TTL_HOURS` | `24` | 未完成的上传会话保留时长 |

#### 推理与后处理默认值

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DETECT_BODY_MAX_INFERENCE_THREADS` | `8` | 硬件计划中的推理线程上限 |
| `DETECT_BODY_PREFER_MODEL_TIER` | `auto` | 优先档位：`auto` / `lite` / `full` / `heavy` |
| `DETECT_BODY_POSE_ENGINE` | `auto` | 默认引擎：`auto` / `mediapipe` / `onnx` |
| `DETECT_BODY_MAX_POSES` | `1` | 最多跟踪人数，1–10 |
| `DETECT_BODY_MIN_DETECTION_CONFIDENCE` | `0.5` | 检测置信度阈值 |
| `DETECT_BODY_MIN_TRACKING_CONFIDENCE` | `0.5` | 跟踪置信度阈值 |
| `DETECT_BODY_SMOOTH_LANDMARKS` | `true` | MediaPipe 内置平滑（VIDEO 模式） |
| `DETECT_BODY_MEDIAPIPE_GPU_DELEGATE` | `false` | 请求 MediaPipe GPU delegate，失败回退 CPU |
| `DETECT_BODY_TRACK_IOU_THRESHOLD` | `0.3` | IoU 跟踪阈值 |
| `DETECT_BODY_TRACK_MAX_AGE` | `15` | 轨迹丢失后保留帧数 |
| `DETECT_BODY_RUNTIME_QUEUE_SIZE` | `64` | 有界队列长度，1–4096 |
| `DETECT_BODY_RUNTIME_BATCH_SIZE` | `4` | 批大小，1–32 |
| `DETECT_BODY_RUNTIME_SEGMENT_FRAMES` | `0` | 分段帧数，0 = 不分段 |
| `DETECT_BODY_IK_FIT_ENABLED` | `true` | 骨长约束拟合总开关 |
| `DETECT_BODY_IK_FIT_MAX_NFEV` | `60` | 每帧求解函数求值上限，1–500 |
| `DETECT_BODY_IK_FIT_FOV_DEGREES` | `60.0` | 相机水平视场角，10–150 度 |
| `DETECT_BODY_Z_SMOOTHING_SCALE` | `0.3` | Z 轴 min_cutoff 相对 x/y 的系数 |
| `DETECT_BODY_DEPTH_SIGN_ENABLED` | `false` | 置 `1` 时全局启用深度方向修正（即使界面开关未勾选） |
| `DETECT_BODY_DEPTH_SIGN_MODE` | `temporal` | 后端：`temporal` 用世界坐标 z 做时序一致性（零新增依赖）/ `model` 用 ONNX 抬升器判方向（权重缺失自动退回 `temporal`，`summary.depthSign.mode` 会如实回传实际后端） |
| `DETECT_BODY_LIFTER` | `videopose3d` | `model` 后端用哪个抬升器：`videopose3d` / `motionbert` / `agformer`，决定搜 `models/videopose3d-h36m.onnx` 还是 `motionbert-h36m.onnx` 等；三者契约相同（`[T,17,2] → [T,17,3]`），只用 Δz 符号 |
| `DETECT_BODY_LOWER_BODY_REFINE` | `false` | MediaPipe 专用：下肢可见度低时把下半身裁出来再推理一次（ROI 重解），坏帧会多付一次推理 |
| `DETECT_BODY_LOWER_LIMB_RETRY` | `true` | 下肢可见度不足时是否用更高档位模型重跑整段（只重跑一遍，不循环） |
| `DETECT_BODY_LOWER_LIMB_RETRY_VISIBILITY` | `0.5` | 髋/膝/踝/脚跟/脚尖的平均可见度低于该值算「下肢不可靠」 |
| `DETECT_BODY_LOWER_LIMB_RETRY_RATIO` | `0.35` | 不可靠帧占比超过该值才重跑（零星坏帧重跑不划算） |

> 另有两个**不由 `Settings` 解析**的环境变量：`DETECT_BODY_WEBVIEW_DEBUG`（设置任意值打开 pywebview debug，由 `app/desktop/main.py` 直接读取）与 `DETECT_BODY_CONSOLE`（值为 `1` 时打包保留控制台，由 `scripts/package.*` 在打包阶段读取）。它们不会出现在 `Settings` 字段中。

### 单次分析配置（AnalysisOptions）

`AnalysisOptions` 是 `POST /api/analysis` 的 `options` 对象，字段名与 JSON 别名都可用。顶层默认值：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `sampling` | `SamplingConfig()` | `every_n_frames=1`、`target_fps=null`、`start_s=null`、`end_s=null`、`trim_empty=false` |
| `postprocess` | `PostProcessConfig()` | One Euro、补帧、手部稳定、角度、质检、IK、yaw 全套默认配置 |
| `skeleton` | `SkeletonConfig()` | 不含可选关节、root motion `world`、`anchor_feet_to_ground=true`（脚部着地锚定）、`include_all_joints=false`；无坐标模式 / 镜像字段 |
| `perf` | `RuntimeConfig()` | 队列 64、batch 4、worker 0、fp16 `auto`、分段 0、checkpoint/resume `false` |
| `refinement` | `RefinementConfig()` | 手/脸增强：`hands=true`、`face=true`、`max_hands=2`、`num_faces=1`、`min_detection_confidence=0.5`、`hand_detection_confidence=0.4`、`hand_presence_confidence=0.35`、`refine_landmarks=true`、`hand_model=null`、`face_model=null`；缺模型自动降级 |
| `ikFit` / `ik_fit` | `null` | 骨长拟合快捷覆盖；HTTP 别名是 `ikFit` |
| `engine` | `auto` | `auto` / `mediapipe` / `onnx` |
| `max_frames` | `null` | 调试用最大帧数 |

关键嵌套默认值：

- `postprocess`：`estimate_orientation=true`、`compute_angles=true`、`fps=30.0`、`stride=1`、`target_track_id=null`。
- 平滑：`method=one-euro`、`window=5`、`polyorder=2`、`extra_world_pass=true`；归一化点 x/y 的 One Euro 为 `1.0/0.5/1.0`，z 为 `0.3/0.15/1.0`；世界点另有独立一组 `1.0/1.0/1.0`，z 为 `0.3/0.3/1.0`（米制量级不同，不能共用参数）。
- 补帧：可见度阈值 0.5、至少 6 个点、`score>=0.4`、最多补 5 帧内部缺口；片头片尾与长缺口保持 `valid=false`。
- 手部稳定（`hand_stability`）：手部 21 点默认开启掉帧补全、姿态迟滞与相对姿态平滑；可在 A/B 时整体关闭。
- 角度：9 个名称（左右肘/肩/膝/髋 + `torso_lean`），优先用 world 点，缺 33 点时回退归一化点，退化角由 `compute_angles()` 直接省略；导出的 `attach_angles()` 在短时遮挡时会沿用上一有效角度，默认最多保持 15 帧（`hold_occluded_frames`）。
- IK：`enabled=true`、`max_nfev=60`、`hard_bones=true`、`refine_passes=2`、`temporal_axes="z"`、`max_yaw_deviation=25°`、`loss="soft_l1"`、`depth_flat_weight=1.5`、`depth_flat_gate_power=4`、`rigidify_failed_frames=true`、`max_reproj_increase=0.01`、`root_prior_weight=0.2`、`depth_prior_weight=0.5`、`aspect_ratio` 缺省 `9/16`、`refine_tolerance=0.01`、`bone_sample_min_projected_ratio=0.85`、`symmetrize_bones=true`、`max_left_right_asymmetry=0.08`，遮挡期还有 `occlusion_weight=1.5`、`occlusion_max_frames=30`、`occlusion_jump_degrees=75`、`max_reproj_error=0.05`；只改 world 点，不改 2D 关键点。
  - `depth_flat_weight` 是**深度压平先验**：按「骨段投影长度 / 标定骨长」的比例（`^gate_power` 门控）把不可观测的视线方向分量压向 0。正面站直的视频不会再被解成"侧视屈膝"；骨段真的朝向镜头（前伸的胳膊）时先验自动失效。设为 `0` 可回到旧行为。
  - `rigidify_failed_frames` 让 IK 被跳过的有效帧（可见度不足 / 尺度估计失败 / 2D 不自洽）也按「观测方向 + 标定骨长」重建，保证整段视频骨长恒定；单骨段位移超过 `rigidify_max_joint_shift`（默认 0.12 m）时才原样透传。
- 质检：有效帧 ready/review 阈值 `0.95/0.8`，z 抖动 `0.01/0.03 m`，骨长 CV `0.10/0.30`，深度置信度门禁 0.6。

界面提交的值与后端裸默认值不同：界面强制 `skeleton.include_optional=true`、`include_all_joints=true`、`root_motion_mode` 默认 `world`、`resume/checkpoint=true`、`batch_size=4`、`refinement.hands/face=true`（无开关）、`ikFit.enabled` 随引擎能力开关。API 直接调用时不会自动套用这些界面值。

> `DETECT_BODY_RUNTIME_*` 和 `DETECT_BODY_IK_FIT_*` 只提供服务端默认值；`pipeline.py` 默认构造 `RuntimeConfig()`，并不会自动读取 `RuntimeConfig.from_settings()`。要让环境变量覆盖单次任务，请在请求体中显式传对应字段。IK 的环境变量覆盖只在未提供 `ikFit` 时生效。

---

## HTTP API

后端启动后，Swagger UI 位于 <http://127.0.0.1:8000/api/docs>，OpenAPI JSON 位于 `/api/openapi.json`。所有路由统一挂在 `/api` 下；开发模式请访问 Vite 地址，由代理转发。

API 约定：

- 业务错误统一返回 `{"error": {"code": "...", "message": "...", "details": {...}}}`。常见 code：`validation_failed` (422)、`unsupported_media` (415)、`payload_too_large` (413)、`not_found` (404)、`conflict` (409)、`analysis_cancelled` (409)、`service_unavailable` (503)、`export_failed` / `export_validation_failed` (500)。
- 任务状态和结果只存内存，服务重启后任务消失；断点文件独立保存在 `~/.detect-body/checkpoints/`。`AnalysisJobManager` 默认最多保留 8 个任务；超出时只从**已结束**任务中淘汰最老的，若 8 个任务都在运行则会暂时超过该数。
- 一次性上传上限 8 GiB；超过 16 MiB 的前端会自动改走分块接口，块大小由 `POST /videos/uploads` 返回的会话决定。
- `GET /analysis/{id}/frames` 只在 `canPreview` 后可读；推理期间返回原始预览帧，完成后切换到最终后处理帧。`limit` 默认 2000、最大 8000；前端按 4000 一页、最多 100000 帧分页加载。
- 导出只有 `canExport`（任务 `done`）后才允许；`downloadUrl` 仅在输出目录等于默认 `exports_dir` 时返回。

### 路由总览

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查，返回版本、Python 与平台 |
| `GET` | `/api/hardware?refresh=false` | 硬件与 EP 报告，60 秒缓存 |
| `GET` | `/api/videos` | 已入库视频列表 |
| `POST` | `/api/videos/upload` | 一次性上传小文件 |
| `POST` | `/api/videos/uploads` | 创建分块上传会话 |
| `GET` | `/api/videos/uploads/{upload_id}` | 查询上传进度 |
| `PUT` | `/api/videos/uploads/{upload_id}/chunks/{index}` | 上传单个分块，二进制 body |
| `POST` | `/api/videos/uploads/{upload_id}/complete` | 合并分块并入库 |
| `DELETE` | `/api/videos/uploads/{upload_id}` | 取消上传 |
| `GET` | `/api/videos/{video_id}` | 视频元信息 |
| `GET` | `/api/videos/{video_id}/thumbnail` | 缩略图 JPEG |
| `GET` | `/api/videos/{video_id}/file` | 原视频内联播放 |
| `POST` | `/api/videos/{video_id}/plan` | 预览采样计划 |
| `POST` | `/api/analysis` | 启动分析任务，返回 `JobStatus` |
| `GET` | `/api/analysis` | 任务列表 |
| `GET` | `/api/analysis/{job_id}` | 查询任务状态与进度 |
| `POST` | `/api/analysis/{job_id}/cancel` | 取消任务 |
| `DELETE` | `/api/analysis/cache` | 清理断点续算缓存，下次识别从第 1 帧开始 |
| `GET` | `/api/analysis/{job_id}/frames` | 分页读取预览帧，`offset` / `limit`，默认 2000、最大 8000 |
| `POST` | `/api/analysis/{job_id}/export` | 导出姿态或动作 JSON |
| `GET` | `/api/exports/schema` | 下载姿态 JSON Schema |
| `GET` | `/api/exports` | 导出文件列表 |
| `GET` | `/api/exports/{filename}` | 下载导出文件 |
| `GET` | `/api/models` | 模型就绪状态、镜像配置与最近 6 条下载任务 |
| `POST` | `/api/models/download` | 启动模型下载 |
| `GET` | `/api/runtime/accelerators` | GPU 运行时选项、已安装/激活状态与安装任务 |
| `POST` | `/api/runtime/accelerators/install` | 启动 CUDA / DirectML 运行时安装（body: `{"variant":"cuda"\|"directml"}`） |
| `POST` | `/api/runtime/accelerators/install/cancel` | 取消当前安装任务 |
| `POST` | `/api/runtime/accelerators/{variant}/activate` | 切换激活的运行时（重启后生效） |
| `DELETE` | `/api/runtime/accelerators/{variant}` | 移除已安装运行时并回滚到 CPU |
| `POST` | `/api/runtime/restart` | 重启应用以加载新运行时 |

### 典型请求

启动分析：

```http
POST /api/analysis
Content-Type: application/json

{
  "video_id": "<videoId>",
  "options": {
    "engine": "auto",
    "max_frames": 300,
    "sampling": { "target_fps": 30 },
    "skeleton": { "root_motion_mode": "in_place" },
    "perf": { "batch_size": 4, "resume": false }
  }
}
```

导出姿态 JSON：

```http
POST /api/analysis/{jobId}/export
Content-Type: application/json

{
  "format": "pose",
  "granularity": "full",
  "include_raw_landmarks": false,
  "include_root_motion": true
}
```

导出动作 JSON：

```http
POST /api/analysis/{jobId}/export
Content-Type: application/json

{
  "format": "motion",
  "motion": {
    "sample_interval_ms": 100,
    "loop": false,
    "name": "Air Squat",
    "name_zh": "深蹲"
  }
}
```

导出响应包含 `artifact`、`outputDir`、`downloadUrl`（仅当输出到默认目录时）、`schemaVersion`、`format`、`depthConfidence` 与 `warnings`。前端应把 `warnings` 展示给用户。

补充约定：

- `format="pose"` 默认 `granularity="full"`、`include_raw_landmarks=false`、`include_root_motion=true`；`output_dir` 留空写 `~/.detect-body/exports`。
- `format="motion"` 固定内部用 `keypoints` 文档转换，忽略 pose 的 `granularity` 与 `include_raw_landmarks`；`motion.sample_interval_ms` 默认 100（10–1000 ms），可选 `loop`、`exercise_id`、`name`、`name_zh`。
- pose 的 `schemaVersion` 是字符串 `"1.1.0"`；motion 的 `schemaVersion` 是字符串 `"1"`，文件内是整数 `1`，不要混写。
- 文件名格式：pose 为 `{原名}_{YYYYmmdd-HHMMSS}_pose{granularity}.json`，motion 为 `{原名}_{YYYYmmdd-HHMMSS}_motion.json`。显式 `filename` 不以 `.json` 结尾会自动补后缀。
- 写出先落 `<filename>.part`，再 Pydantic 回读校验帧数与版本，最后原子替换；回读是一次性 `read_text()`，超大文件校验时会占用内存。

---

## 命令行与脚本

### Python 模块入口

| 命令 | 作用 |
| --- | --- |
| `python -m uvicorn app.backend.main:app --reload` | 仅启动 FastAPI 服务 |
| `python -m app.desktop.main` | 桌面壳 / 浏览器 / 无窗口模式 |
| `python -m app.backend.export.schema` | 重新生成 `app/backend/export/schema.json` |
| `python -m app.backend.export.samples` | 重新生成 `samples/expected_pose.json` |
| `python -m app.backend.export.motion_clip SOURCE [-o OUTPUT]` | 把已有姿态 JSON 离线转成动作 JSON |

姿态转动作的完整参数：

```text
python -m app.backend.export.motion_clip SOURCE [-o OUTPUT]
    [--interval MS] [--loop] [--name NAME] [--name-zh NAME_ZH]

--interval    采样间隔，默认 100 ms
--loop        是否循环
--name        动作英文名
--name-zh     动作中文名
```

示例：

```powershell
python -m app.backend.export.motion_clip clip_posefull.json -o clip_motion.json --name "Air Squat" --name-zh "深蹲"
```

### 双击入口

| 文件 | 平台 | 作用 |
| --- | --- | --- |
| `start.bat` | Windows | 双击启动源码桌面壳；缺环境时自动 bootstrap + 构建前端 |
| `start.command` | macOS | 同上；首次需 `chmod +x start.command` |
| `start.sh` | Linux | 同上；首次需 `chmod +x start.sh` |

入口会把参数透传给 `python -m app.desktop.main`（在终端执行时生效，例如 `./start.sh --browser`）。分发用的原生产物入口见[1.5 双击启动入口](#15-双击启动入口)与[打包与部署](#打包与部署)。

### 生命周期脚本

| 脚本 | 参数 | 作用 |
| --- | --- | --- |
| `scripts/bootstrap.ps1` | `-Cuda`、`-SkipFrontend`、`-SkipModels`、`-Python` | Windows 首次准备 |
| `scripts/bootstrap.sh` | `--cuda`、`--skip-frontend`、`--skip-models` | macOS / Linux 首次准备 |
| `scripts/dev.ps1` | `-BackendPort`、`-SkipFrontend` | Windows 开发模式 |
| `scripts/dev.sh` | 环境变量 `PORT` | macOS / Linux 开发模式 |
| `scripts/build_frontend.ps1` | `-Clean`、`-SkipInstall` | Windows 构建前端到 `app/backend/static` |
| `scripts/build_frontend.sh` | `--clean`、`--skip-install` | macOS / Linux 构建前端 |
| `scripts/package.ps1` | `-SkipFrontend`、`-Console`、`-Clean`、`-Python` | Windows PyInstaller 打包 |
| `scripts/package.sh` | `--skip-frontend`、`--console`、`--clean`、`PYTHON` 环境变量 | macOS / Linux PyInstaller 打包 |

脚本使用注意：

- `bootstrap`/`dev` 会优先使用仓库 `.venv`；`package.ps1` 默认用系统 `python`，建议显式传 `-Python .venv\Scripts\python.exe`，`package.sh` 用 `PYTHON=.venv/bin/python ./scripts/package.sh`，避免打包进错环境。
- `build_frontend.*` 在有 `package-lock.json` 时用 `npm ci`，否则 `npm install`；找不到 npm 会直接报错并要求 Node.js 18+。
- `dev.ps1 -BackendPort` / `PORT=... dev.sh` 只改变后端监听端口，Vite 代理仍指向 8000；见第 1 节说明。
- `package.*` 会在缺少 PyInstaller 时自动安装 `pyinstaller>=6.6`，并先构建前端、生成图标；`-Console` / `--console` 设置 `DETECT_BODY_CONSOLE=1` 保留控制台便于排错。
- 修改 `export/schema.py` 或后处理/质检逻辑后，需要同步重生成 schema 与 golden 样例：

```bash
python -m app.backend.export.schema
python -m app.backend.export.samples
pytest tests/test_export.py -q
```

### 工具脚本

| 脚本 | 常用命令 | 作用 |
| --- | --- | --- |
| `scripts/bench.py` | `python scripts/bench.py --engine stub --synthetic --batch 1,4` | 推理链路吞吐基准；`--engine` 可选 `stub` / `auto` / `mediapipe` / `onnx` |
| `scripts/bench_depth.py` | `python scripts/bench_depth.py --synthetic --json out/depth.json` | 深度 / 几何质量基准：`z_jitter`、`bone_length_cv`、`reproj_error_px`、`angle_drift_deg`、`depth_sign_flips`、`sagittal_motion_share`；踢腿专项加 `--labels` 后输出 `sign_accuracy`、`kick_sagittal_error_*`、`kick_phase_max_frame_error` |
| `scripts/bench_z_smoothing.py` | `python scripts/bench_z_smoothing.py --json out/z_params.json` | Z 轴 One Euro 参数 A/B |
| `scripts/export_videopose3d_onnx.py` | `python scripts/export_videopose3d_onnx.py --verify models/videopose3d-h36m.onnx` | 把 VideoPose3D 离线导出为 ONNX（供 `DETECT_BODY_DEPTH_SIGN_MODE=model`），或校验已有权重 |
| `scripts/export_motionbert_onnx.py` | `python scripts/export_motionbert_onnx.py --verify models/motionbert-h36m.onnx` | 同上，导出 MotionBERT（DSTFormer），支持 `--mode dynamic/fixed` 两种 T 策略 |
| `scripts/fuse_views.py` | `python scripts/fuse_views.py --front BM-05.mp4 --side BM-06.mp4 --out out/fused.json` | 双机位角度级融合（不三角化）：矢状面角度取侧视、额状面取正视，按动作相位对齐时间轴 |
| `scripts/check_depth_sign_chain.py` | `python scripts/check_depth_sign_chain.py --mode model` | 深度方向修正链路自检（无视频/无权重也能跑）：阶段进度、`mode`、翻转次数、质检 notes 四项判定 |
| `scripts/make_icon.py` | `python scripts/make_icon.py` | 无第三方依赖生成 512 PNG 与 256 ICO 图标 |

`scripts/bench.py` 默认引擎是 `stub`，只验证基准工具与队列 / batch 链路，**不代表真实模型性能**。真实性能必须显式传 `--engine auto|mediapipe|onnx`。

`scripts/bench_depth.py` 当前虽然解析 `--engine`，但 `_analyze()` 没有把引擎写入 `AnalysisOptions`，实际引擎由默认选择决定；做引擎对比时请以界面或 API 显式设置 `engine` 的结果为准。

---

## 导出格式

### 姿态 JSON（`*_pose*.json`）

契约以 `app/backend/export/schema.py` 为准，当前 `schemaVersion = "1.1.0"`。顶层字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `version` | string | `schemaVersion`，当前 `1.1.0` |
| `generator` | string | 生成器标识 |
| `source` | object | 源视频：`file` / `fps` / `width` / `height` / `frames` / `durationSeconds` |
| `coordinateSystem` | string | 恒为 `arkit-right-handed-y-up` |
| `units` | string | 恒为 `meters` |
| `depthIsEstimated` | bool | 恒为 `true` |
| `depthConfidence` | number | 0–1，整段深度可信度 |
| `warnings` | string[] | 深度 / 构图提示，低于阈值才非空 |
| `personHeightMeters` | number | 由世界坐标自动测量的参考高度（头中心→脚底），用于核对模型缩放，不参与坐标计算 |
| `jointNames` | string[] | 本文件使用的 ARKit 关节名 |
| `frames` | object[] | 逐帧数据 |
| `quality` | object | 质检报告 |
| `rootMotion` | object \| null | 根运动 |
| `notes` | string[] | 运行说明 |

`frames[]` 每项：

| 字段 | 说明 |
| --- | --- |
| `index` | 帧号，严格递增 |
| `t` | 相对视频起点的秒数 |
| `valid` | `false` 表示没有可信姿态 |
| `interpolated` | 是否由插值补齐 |
| `joints` | `{关节名: {x, y, z, visibility}}`，相对 root，米制 |
| `angles` | `{角度名: 0~180}`，仅 `full` 粒度 |
| `rawLandmarks` | 原始 33 点，仅显式开启时导出 |

三种粒度：

| 粒度 | 内容 |
| --- | --- |
| `keypoints` | 18 个 ARKit 核心关节 + 可选关节；界面提交 `include_all_joints=true` 时为 91 个，无角度 |
| `full` | 关键点 + 关节角度，默认；界面会自动附带原始 33 点，并输出 91 个关节 |
| `compact` | 只保留 18 个 ARKit 核心关节，无角度；UI 不带原始 33 点，但 API 显式传 `include_raw_landmarks=true` 仍会附带（此时写出器按 `keypoints` 回执/命名） |

界面识别设置默认提交 `skeleton.include_all_joints=true`，导出 `jointNames` 与
`resources/AppleModel.usda` 的 91 个关节**逐项同名同序**（顺序以 `app/backend/skeleton/mapping.py::APPLE_ALL_JOINTS` 为准；`resources/` 被 gitignore，本工作区已有该文件，新克隆需自行放置；含双手各 24 个指关节、
脊柱 1–7、颈 1–4、下颌/鼻/眼/眼睑/虹膜、脚尖末端）。API 直接调用默认仍是旧 18/22
关节，保持向后兼容；显式传 `include_all_joints=true` 即可开启全关节。增强关节的
坐标是估计值：`world` 模式以对应姿态腕点/鼻点为锚、按局部尺度换算，无尺度时记入
`missing` 而不伪造米制坐标；`compact` 粒度仍只保留 18 个核心关节。

> 界面按钮的说明文案与代码有出入：卡片把 `compact` 写成「关键帧 + 角度」、把 `keypoints` 写成「只有关键点与角度」，但实际只有 `full` 导出 `angles`。以本表和 `app/backend/export/builder.py` 为准。另外，`root` 与 `hips_joint` 都取自两髋中点，不是两个独立关节；`joints` 是相对 root 的位移。

质量报告 `quality` 的关键字段：`framesTotal`、`framesValid`、`validFrameRatio`、`interpolatedFrames`、`meanVisibility`、`meanScore`、`maxJump`、`maxJumpFrame`、`zJitterRms`、`boneLengthCv`、`meanReprojError`、`depthConfidence`、`verdict`（`ready` / `review` / `rerecord`）、`verdictLabel`、`issues`、`smoothing`、`notes`。拍摄机位/距离指标的中间报告 `framing` 只存在于内部 `QualityReport`，导出时折算进 `issues` / `warnings`，不作为 `quality` 字段写出。

`depthConfidence` 的计算：

```text
无 world landmarks（如 ONNX 引擎）        → confidence = 0.0
否则 confidence = 1 - (0.5 * min(1, z抖动 / 0.03)
                     + 0.3 * min(1, 骨长CV / 0.30)
                     + 0.2 * min(1, 重投影误差 / 0.02))
```

判读：≥0.75 可信；0.5–0.75 一般；<0.5 不可信。只有**有世界坐标且** `quality.enable_depth_gate=true` 时，`confidence < 0.6` 才会把 `ready` 压到 `review`；无 world 时不降级。内部 `framing` 指标只追加 issues，不改变 verdict。

版本规则：同一 major（`1.x`）向后兼容，新增字段必须可选且带默认值；改名、改单位、改必填必须升 major。

### 动作 JSON（`*_motion.json`）

面向 iOS App 的 `MotionFrame` 解码器：

| 字段 | 说明 |
| --- | --- |
| `schemaVersion` | 当前为 `1` |
| `exerciseID` / `name` / `nameZh` | 动作标识与显示名 |
| `coordinateSpace` | 恒为 `hip-local-meters-yup` |
| `bodyOrientation` | 恒为 `upright` |
| `referenceModel` / `mirroredReference` | 参考骨架标识与镜像约定（默认 `true`） |
| `jointVisibilityNote` | 可见度缺省规则说明（缺省视为 1） |
| `sampleIntervalMs` | 默认 `100`，即 10 Hz |
| `durationMs` | 最后一帧的 `t` |
| `loop` / `demonstrationLoops` | 循环设置 |
| `skeleton` / `keyframesMs` / `playbackHolds` | App 播放元数据 |
| `frames[].t` | 毫秒时间戳，严格递增 |
| `frames[].joints` | 17 个身体关节的米制坐标，`pelvis` / `root` 恒为原点 |
| `frames[].jointVisibility` | 只列 `visibility < 1` 的关节，缺省视为 1 |
| `frames[].pelvisOffsetMeters` | 骨盆相对站立参考帧的升降 |
| `frames[].confidence` | 该帧 17 个关节的平均可见度 |
| `viewpoint` | 朝向与 forward 轴说明 |

关节映射：

| 动作 JSON | 本项目 ARKit 关节 |
| --- | --- |
| `root` / `pelvis` | `root` / `hips_joint` |
| `spine` | `spine_3_joint` |
| `centerShoulder` | 左右肩关节中点（`spine_7_joint` 只是映射表占位，写出时被中点覆盖） |
| `neck` | `neck_1_joint` |
| `left/rightShoulder` | `left/right_shoulder_1_joint` |
| `left/rightElbow` | `left/right_arm_joint` |
| `left/rightWrist` | `left/right_hand_joint` |
| `left/rightHip` | `left/right_upLeg_joint` |
| `left/rightKnee` | `left/right_leg_joint` |
| `left/rightAnkle` | `left/right_foot_joint` |

重采样规则：两帧有效姿态之间线性插值；`valid=false` 的帧不参与插值；间隔超过 400 ms 时保持前一帧，不伪造动作；坐标保留 4 位小数并消除 `-0.0`。

### iOS / macOS 消费检查清单

1. `coordinateSystem` 必须是 `arkit-right-handed-y-up`；RealityKit 是米制、右手系、`y` 向上、相机朝 `-z`。
2. `units` 必须是 `meters`；用 `personHeightMeters` 核对模型缩放。
3. USDZ 骨骼节点名需与 `jointNames` 一致；命名不同则在 App 侧做重映射。
4. `left_*` 只来自人物左侧，不要按屏幕方向镜像。
5. `valid == false` 的帧保持上一姿态，不要插值。
6. `joints` 是相对 root 的位移；世界位移看 `rootMotion.frames`；`in_place` 模式位移恒为 0。
7. `depthIsEstimated` 恒为 `true`。不要拿 `z` 判断人物站位或走了多远；只取关节方向 + 模型自身骨长。
8. `warnings` 非空或 `depthConfidence < 0.6` 时，应把提示展示给用户。

参考实现：`samples/RealityKitSample.swift`（解码、时间轴插值、root motion、坐标系校验）。

---

## 坐标、深度与精度边界

三种坐标的可信度不同，读数据前先确认自己在看哪一种：

| 坐标 | 含义 | 可信度 |
| --- | --- | --- |
| `landmarks`（归一化） | `x,y ∈ [0,1]`，`y` 向下；`z` 相对髋中心 | `x/y` 直接来自 2D 检测，亚像素级可信；`z` 是模型估计 |
| `world_landmarks` | 米制，原点约两髋中点，`y` 向下 | 已是米制，`x/y` 可信；`z` 仅为估计，不能当绝对深度 |
| 导出 `joints` | 米、右手系、`y` 向上、`-z` 向前 | 由上面两者转换，`x/y` 沿用 2D 可信度，`z` 同为估计 |

导出统一转换：**先把每个关节减去髋中心（或 root）归零**，再做 `x' = x`、`y' = -y`、`z' = -z`；根平移单独写入 `rootMotion`。

### 根运动与世界坐标

- 3D 骨骼直接使用 MediaPipe world landmarks，其原点约在两髋中点，水平位移被抹掉；`root.y` 由**脚部着地锚定**按髋到脚的跨度恢复，使下蹲时髋部下沉、脚底保持在地面（参考腿长取整段较高分位数，抗误检）。
- 没有 world landmarks 的引擎（ONNX/RTMPose）不产出米制关节：不再用"身高 ÷ 归一化像素跨度"反解尺度（该尺度随姿态和机位漂移，实测效果明显更差）。
- `in_place` 根运动模式把平移全部置 0，只保留 yaw 旋转；`world` 模式允许平移。
- `rootMotion.totalTranslation` 是**末帧相对首帧的直线距离**，不是累计行走路程；其前后（z）分量仍是单目估计值，不可用于测量走了多远。
- IK 拟合的 yaw 先验用肩/髋投影宽度比估计，符号取自左右肩 world z 差；正对（宽度比 >0.92）时置信度上限 0.45，宽度比 <0.25 退化为 ±90° 且置信度 0.15。
- 导出的 `rootMotion.rotation` 是另一路：默认取左右髋连线的世界坐标 `atan2(-dz, dx)`（`yaw_source="hips"`），与旧的 `yaw_from_world_z()` 同型；旧函数本身仍只作对比、不在该路径调用。

### 关于 Z / 深度

- 单目 RGB **无法测绝对深度**。`z` 是"基于人体先验的估计"，绝对深度与前后位移不可信。
- **正面视角看不到矢状面，就无法判断"侧面该不该弯"**：同一张正面图像对应一族深 3D 姿态，屈膝与否在数学上是多解的。工程上唯一正确的处理是"没有证据就不掰"——`depth_flat_weight` 把不可观测的深度分量压向 0，宁可少还原一点前后偏移，也不凭空造出屈膝。真实深度只有在 2D 已经强烈要求时（骨段投影明显短于骨长，例如前伸的胳膊）才会被保留。
- 可用的是**关节相对方向与角度**：膝角、肘角、躯干前倾、哪只手更靠前、转身方向。
- 项目用三道手段压住 Z 误差：固定骨长 + 2D 重投影拟合（`ikFit`）、Z 轴独立强平滑、肩宽/髋宽投影宽度比估计 yaw。
- 这些手段能保证"骨长恒定、与画面 2D 一致、朝向不抖"，但**不能**让绝对深度变准。
- 导出 JSON 用 `depthIsEstimated: true`、`depthConfidence` 和 `warnings` 明确标注；低置信度时给出"建议从侧面重拍"等原因。
- 驱动 USDZ 时只取关节方向，骨长用模型自身骨架；这样即使 `z` 有误差，也不会放大成体型变化。
- 需要真实走位 / 绝对深度时，请使用多机位、深度相机或其他外部数据。

---

## 拍摄建议

拍摄质量决定精度上限。后处理只能修正抖动，**修不出没拍到的信息**。

### 机位与构图

- **正对或正侧拍摄**：斜 45° 是两者最差的组合——2D 两个方向都被压缩，深度又最靠猜。
- **机位与胸口齐平**：俯拍 / 仰拍会让上下半身尺度不一致，透视越强深度越不可信。
- **退到 3–5 米、用标准焦段**：贴脸广角会拉长四肢并放大深度误差。
- **人物占画面高度 60%–90%**：太小则人脸 / 手指点位精度不够。
- **避免快速朝镜头走动**：前后位移是单目最不可靠的量。
- 画面稳定、光线均匀、不逆光，全身尽量不被裁切。

### 按动作类型选机位

| 动作类型 | 建议机位 | 原因 |
| --- | --- | --- |
| 下蹲 / 弓步 / 前弓步 / 硬拉 | 正侧面 | 膝角、髋角在图像平面内，前缩误差最小 |
| 前踢腿 / 直腿抬高 / 摆腿 | **正侧面**（若只能正面：务必配合双机位或接受"方向对、幅度不可信"） | 踢腿的位移几乎全在前后方向（Z 轴），正面机位下 2D 只看到腿长缩短，深度完全靠猜：方向可能对，但抬腿幅度不可验证 |
| 转体 / 挥臂 / 上肢伸展 / 开合跳 | 正对 | 需要看清左右对称与水平方向角度 |
| 走路 / 跑步 | 正侧（横向走） | 前后走会把位移交给最不可信的深度 |
| 混合动作 | 正侧优先 | 对深度依赖最小，容错最高 |

质检面板会在"机位过高/偏低、透视过强、离得太远、快速走近"时给出提示；导出 JSON 的 `warnings` 也会写原因。

---

## 数据目录

源码开发时，模型默认在仓库根目录 `models/`；其余数据默认在 `~/.detect-body/`（Windows 为 `%USERPROFILE%\.detect-body`）：

```text
~/.detect-body/
├─ library/       # 按 SHA-256 去重的原视频
├─ thumbnails/    # 缩略图 JPEG
├─ temp/
│  └─ uploads/    # 分块上传临时文件
├─ exports/       # 默认导出目录
├─ checkpoints/   # *.pose.jsonl 断点续算文件
├─ logs/          # 桌面壳与后端日志
├─ runtime/       # 应用内安装的 CUDA / DirectML 独立后端（按需创建）
├─ webview/       # pywebview 本地存储
└─ models/        # 打包后模型目录
```

可修改 `DETECT_BODY_DATA_DIR` 整体迁移；`exports_dir`、`checkpoints_dir`、`logs_dir` 等由 `Settings` 自动派生并在启动时创建；`runtime/` 由安装动作按需创建。

---

## 测试与基准

### 测试

项目包含 23 个测试模块，加上 `conftest.py` 和合成姿态工具 `synth_pose.py`，当前共有 237 个顶层 `test_` 函数（参数化后实际用例更多）。

```bash
# 全部测试
python -m pytest

# 单项
pytest tests/test_export.py -q
pytest tests/test_yaw.py -q
pytest tests/test_motion_clip.py -q

# 静态检查
ruff check app scripts tests
```

主要覆盖：

| 模块 | 覆盖范围 |
| --- | --- |
| `test_acceleration.py` | TensorRT / CUDA / CoreML / DirectML / CPU EP 选择、线程上限、VRAM 边界、fallback |
| `test_analysis_api.py` | 任务 → 预览帧 → 导出、motion 导出、下载 URL、schema、错误状态 |
| `test_apple_joints.py` | 91 关节与 `resources/AppleModel.usda` 逐项对齐、手/脸增强映射到 ARKit、左右手分配、world 锚定与无尺度跳过 |
| `test_depth_fit.py` | 合成真值骨架、骨长保持、重投影改善、yaw prior、禁用与降级 |
| `test_desktop.py` | 空闲端口、健康检查、WebView2 状态、图标、标题、退出清理 |
| `test_engine.py` | `PoseFrame` 序列化、bbox、引擎降级、crop、SimCC 解码、COCO17 映射、多人跟踪 |
| `test_export.py` | 三种粒度、raw landmarks、四位小数、文件名、流式写出与回读、schema 同步、golden 样例 |
| `test_ground_contact.py` | 脚部着地锚定：踝固定/髋下沉、可关闭、无脚帧跳过、深蹲 bundle 回归 |
| `test_hand_stability.py` | 手部 21 点时序稳定：缺帧复用、长缺口不伪造、单帧误检拒绝、迟滞/确认帧、左右手按腕点归属、pipeline 阶段上报 |
| `test_hardware_collect.py` | 系统信息采集鲁棒性与报告结构 |
| `test_models_api.py` | 模型清单、下载进度、重复任务复用 |
| `test_motion_clip.py` | 动作 JSON 字段、17 关节、10 Hz 重采样、可见性、骨盆升降、CLI |
| `test_occlusion_hold.py` | 肢体互相遮挡时关节角保持与释放、关闭时的劣化对照 |
| `test_perf.py` | batch、checkpoint resume、坏行容错、计划 key |
| `test_pipeline.py` | 端到端分析、取消、`max_frames` |
| `test_postprocess.py` | One Euro、移动平均 / SAVGOL、补帧、角度、质检 verdict、深度警告 |
| `test_resume_preview.py` | 取消 → 断点续算时预览必须包含此前已完成帧 |
| `test_runtime_api.py` | GPU 加速决策（NVIDIA/AMD/Apple/旧驱动/frozen）、一键安装任务、重试/超时/重复/回滚、`/api/runtime` 接口 |
| `test_skeleton.py` | ARKit 映射、左右不交换、镜像、2D/3D roundtrip、root motion、yaw、无 world 回退 |
| `test_upload_api.py` | 上传去重、缩略图、坏文件拒绝、plan、分块上传、缺块 |
| `test_video_media.py` | 视频探测、缩略图、逐帧读取、stride、时间范围、随机访问、采样计划 |
| `test_yaw.py` | ±20° / ±45° / ±60° yaw 恢复、±90° 退化、遮挡回退 |
| `test_z_smoothing.py` | Z 参数默认值、独立 z 滤波、向后兼容、z 抖动下降且 xy 不变 |

跳过条件：

- OpenCV 缺少 `mp4v` 编码器时，合成视频 fixture 会跳过。
- `skeleton.orientation` 导入失败时，yaw 测试整体跳过。
- `resources/air-squat-motion.json` 未放置时，动作样例子集测试跳过。
- `test_apple_joints.py` 直接读取工作区的 `resources/AppleModel.usda`，没有 skip 保护；新克隆的仓库需自行放置该参考文件，否则该模块会失败。

### 基准

```powershell
# 推理链路吞吐（stub 自检）
python scripts/bench.py --engine stub --synthetic --batch 1,4 --stride 2

# 真实视频 + 自动选型
python scripts/bench.py samples/dance.mp4 --engine auto --batch 1,4,8 --tiers lite,full

# 深度 / 几何质量
python scripts/bench_depth.py samples/BM-01-front-full.mp4 --json out/BM-01.json --label BM-01

# 与上次结果对比
python scripts/bench_depth.py samples/BM-01-front-full.mp4 --baseline out/BM-01.json

# 深度方向修正 A/B（对比 depth_sign_flips 与 sagittal_motion_share）
python scripts/bench_depth.py samples/kick.mp4 --depth-sign --json out/kick-on.json
python scripts/bench_depth.py samples/kick.mp4 --json out/kick-off.json --baseline out/kick-on.json

# Z 轴滤波器调参
python scripts/bench_z_smoothing.py --json out/z_params.json
```

`docs/benchmark.md` 记录方法与历史结果；其中 `BENCH:START` / `BENCH:END` 之间由 `scripts/bench.py` 自动替换，不要手改。仓库**不包含真实基准视频**，只有 `samples/benchmark/README.md` 中的素材清单（BM-01 至 BM-07，其中 BM-05~07 是踢腿素材，标注放 `samples/benchmark/kick-labels.json`）。

---

## 打包与部署

打包采用 PyInstaller **onedir**，配置在 `packaging/detect-body.spec`；窗口用 pywebview，前端静态资源由后端托管，窗口和浏览器都只访问同一个端口。

```powershell
# Windows
./scripts/bootstrap.ps1
./scripts/build_frontend.ps1
./scripts/package.ps1
# 产物：dist/DetectBodyInVideo/DetectBodyInVideo.exe
```

```bash
# macOS / Linux
./scripts/bootstrap.sh
./scripts/build_frontend.sh
./scripts/package.sh
# macOS: dist/DetectBodyInVideo.app
# Linux: dist/DetectBodyInVideo/DetectBodyInVideo
```

也可直接用项目环境里的 PyInstaller：

```bash
python -m PyInstaller packaging/detect-body.spec --noconfirm
```

`package.ps1` 默认调用系统 `python`；如果使用 `.venv`，显式传 `-Python .venv\Scripts\python.exe`。`package.sh` 用 `PYTHON=.venv/bin/python ./scripts/package.sh`，否则可能打包进错误的解释器。脚本会在缺少 PyInstaller 时安装 `pyinstaller>=6.6`。

打包 spec 会自动收集 `app/backend/static`、构建时已存在的 `models/*.task`、`webview` 的平台 DLL / JS，并显式加入 `webview.platforms.*`、`clr_loader`、`pythonnet`、`uvicorn/h11/anyio` 等 hidden imports。如果构建时 `models/` 为空，产物不会内嵌权重，首次启动仍需在界面下载到 `~/.detect-body/models`。Windows 使用 `.ico`，macOS 生成 `.app`（bundle id `ai.detectbody.desktop`，最低 macOS 11.0）。`DETECT_BODY_CONSOLE=1` 时保留控制台便于排错。

仓库没有 Dockerfile、Compose 或 Kubernetes 清单；服务端部署通常是源码 `uvicorn` 或 onedir 产物 + 自建反向代理。分发前验收清单：

1. 目标机无 Python / Node 时能从干净目录双击启动并完成一次导出。
2. 无 NVIDIA 的机器走 CPU 路径，硬件卡显示 CPU 模式并标注降级理由。
3. Windows 装 WebView2；Linux 装 WebKitGTK；缺依赖时能降级系统浏览器。
4. 首启缺模型时能在界面下载并立即开始分析；离线机器需预置 `~/.detect-body/models`。
5. 单端口页面、API、导出文件下载都正常；日志写在 `~/.detect-body/logs/app.log`。

平台要点：

| 平台 | 运行时依赖 | 说明 |
| --- | --- | --- |
| Windows 10/11 | WebView2 Runtime、VC++ 2015–2022 Redistributable | 缺 WebView2 会自动降级浏览器；未签名 exe 可能被 SmartScreen 拦截 |
| macOS | 系统 WKWebView | 需 codesign / notarize 才能无提示分发；未签名可右键打开或移除 quarantine |
| Linux | GTK3 + WebKitGTK | Debian/Ubuntu：`libwebkit2gtk-4.1-0 libgtk-3-0`；Fedora：`webkit2gtk4.1 gtk3` |

如果只是想在本机双击打开而不做分发，根目录的 `start.bat` / `start.command` / `start.sh` 已覆盖源码模式（见[1.5 双击启动入口](#15-双击启动入口)）；分发时请直接使用上表中的原生产物。

无窗口服务模式：

```bash
DetectBodyInVideo --no-window --port 8000
# 或源码
python -m app.desktop.main --no-window --port 8000
```

macOS 签名与公证命令、AppImage / .deb 思路见 `docs/packaging.md`。平台验收清单在该文档中列出，目标机上的完整验收仍需在真实环境完成。

---

## 故障排查

| 现象 | 处理 |
| --- | --- |
| 低版本 Python 下安装失败 | 使用 Python 3.9 及以上；项目声明 `>=3.9`，无上限 |
| 找不到 npm | 安装 Node.js 18+ 后重新 bootstrap |
| 模型下载失败 / 超时 | 设置 `DETECT_BODY_MODEL_MIRROR_URL`，或手动把 `.task` 放到 `models/` / `~/.detect-body/models` |
| 打包后白屏 / 404 | 确认 `app/backend/static/index.html` 存在；先跑 `build_frontend` |
| 双击 exe 没窗口 | 用 `-Console` / `--console` 重跑；Windows 装 WebView2，Linux 装 WebKitGTK |
| 双击 `start.bat` 后一闪而过 | 在终端执行 `.\start.bat` 看报错；通常是 PATH 里缺 Python 3.9+ |
| 双击 `start.command` / `start.sh` 提示无权限 | 终端执行一次 `chmod +x start.command`（或 `start.sh`） |
| 双击入口卡在前端构建 | 确认 Node.js 18+ 在 PATH 中；或先手动跑 `build_frontend` |
| 缺 `msvcp140.dll` / `vcruntime140.dll` | 安装 VC++ 2015–2022 Redistributable |
| Windows SmartScreen / 杀毒误报 | PyInstaller 产物常见；代码签名或加白名单 |
| macOS Gatekeeper 拦截 | 右键打开，或 `xattr -dr com.apple.quarantine /Applications/DetectBodyInVideo.app` |
| 端口占用 | 未显式传 `--port` 时桌面壳自动选空闲端口；显式指定占用端口会健康检查超时退出，需换端口 |
| 健康检查超时 | 查 `~/.detect-body/logs`；桌面壳 30 秒未就绪会返回退出码 1 |
| 测试无法生成视频 | 当前 OpenCV 缺 `mp4v` 编码器时 fixture 会跳过 |
| 基准结果看起来很好但不真实 | `stub` 只是链路自检；真实性能需用 `--engine auto|mediapipe|onnx` 重跑 |
| 深度结果不可信 | 检查 `depthConfidence` 与 `warnings`；按侧面机位重拍；不要把 `z` 当绝对距离 |
| 深蹲时脚离地、髋不下沉 | 已启用脚部着地锚定：3D 预览按脚尖贴地重定位髋，`world` 模式 `root.y` 也恢复下沉；若脚踝/脚尖长期被遮挡，锚定会跳过，可先用 `skeleton.anchor_feet_to_ground=false` 做 A/B 对比 |
| 动作 JSON 无法还原腾空 | 单目 + 髋局部坐标只能靠着地假设还原下蹲下沉，腾空高度不可测 |
| 左右与真实人物相反 | 源视频可能是自拍镜像；转码翻转画面或在 App 侧处理 `mirroredReference` |

---

## 已知限制与维护说明

- 单目视频的绝对深度和前后位移不可恢复；导出中的 `z` 只能作相对关系使用。
- 动作 JSON（iOS `MotionFrame`）不含手指和面部关键点，仍是 17 关节；全关节只在姿态 JSON 的 `full`/`keypoints` 粒度中提供。
- `world` 模式下手指/面部关节无米制 world 观测，以姿态腕点/鼻点为锚按局部尺度估计；单目深度不可靠，指关节的前后方向仅作相对参考。
- `mirroredReference` 固定为 `true`（与 App 样例一致）；自拍镜像需额外处理。
- `pelvisOffsetMeters` 靠着地锚定只能还原下蹲 / 屈膝下沉，无法还原跳跃腾空。
- ONNX 引擎只接受 COCO 17 点模型，不输出 `z` / world landmarks；多人能力只由文件名中的 `rtmo` 判断。
- MediaPipe 引擎不直接使用其 VIDEO 模式的 `tracking_ids`；`track_id` 由 `perf/runner.py` 统一分配（单人恒为 1，多人 IoU 跟踪），后处理轨迹筛选照常生效；显式指定引擎失败也会自动降级到另一引擎。
- 补帧只处理长度 ≤5 的内部缺口；片头片尾和长缺口保持 `valid=false`。`densify(stride>1)` 在末帧不对齐步长时可能漏掉末帧。
- IK 只修改 world 点、不改 2D 关键点；根平移只优化 x/y，z 沿用引擎估计值。无 SciPy 时仅做骨长钳制，不做最小二乘优化。
- 深度压平先验是**用精度换合理性**：它会让重投影误差从约 5 px 升到约 9 px（1080p，0.5% 画面高度以内），换来的是不再凭空出现十几厘米的屈膝。需要极限 2D 一致性时把 `depth_flat_weight` 调低到 0.5 或 0。
- 单目 RGB 下每根骨段的深度**只剩一个比特**（骨长与 2D 投影长度已把 `|Δz|` 定死，`ΔZ = ±√(L² − ΔX² − ΔY²)`）。「向前踢腿」在正面机位就是典型的两个等价解，`depth_sign` 只是用运动先验替观众做选择，**不是把不可观测量变成可观测**，所以导出仍标记 `depthIsEstimated`；要物理上可靠请改用正侧机位。`temporal` 后端只能消除帧间翻转、无法纠正整体方向，要判断「朝前还是朝后」必须用 `model` 后端并自备 ONNX 权重。
- 补帧现在同时插值 `world_landmarks`；若有效帧本身没有世界坐标（例如 ONNX 引擎），导出 3D 关节为空并在 `notes` 提示改用能输出 world landmarks 的引擎，不会回退到身高标定反解。
- 任务结果与预览只在内存，服务重启后任务丢失；断点文件独立保存在 `~/.detect-body/checkpoints/`，其 key 由版本号、文件名/大小/mtime、探测到的 fps/帧数和采样计划共同派生（SHA-1 前 16 位），不是内容 hash。
- 导出回读校验是一次性 `read_text()`，超大文件会占用内存；写出本身是流式的。
- 仓库（Git 分发）不含真实视频基准素材，也不含 `resources/AppleModel.usda`、`demo.mp4`、`air-squat-motion.json`，这些文件按 `.gitignore` 排除；**本工作区已存在这三个参考文件**，新克隆需自行放置。
- `.task` / `.onnx` 权重按 `.gitignore` 排除；新克隆需要下载或手动放置。本工作区当前有 lite/full 姿态模型与 hand/face 增强模型，heavy 缺失。
- `scripts/bench.py` 默认 `stub` 只做链路自检；`scripts/bench_depth.py --engine` 未传入 `AnalysisOptions`，不能用于引擎对比。
- 界面导出粒度的说明文案与代码不一致，只有 `full` 会输出 `angles`；见「导出格式」。
- `DETECT_BODY_API_PREFIX` 和 Vite `/api` 代理目标都不要随意改：桌面壳健康检查硬编码 `/api/health`。
- 服务没有内置认证/TLS，默认 CORS 只放行本地 Vite；不要直接暴露到公网。
- `TODO.md` 中的百分比与完成状态是历史记录，部分条目已落后于源码；当前行为以源码和本文档为准。
- 旧版 README 提到的 `docs/acceleration.md` 当前不存在；硬件加速矩阵已合并到本文档。
- `docs/benchmark.md` 中的 Python 3.13.3 基准环境仍然受支持（`>=3.9` 无上限）。
- 项目元数据声明 MIT，但仓库根目录当前没有 `LICENSE` 文本；对外分发前建议补齐。

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/packaging.md` | 三平台打包、签名、运行时依赖、故障排查与验收清单 |
| `docs/json-schema.md` | 姿态 JSON 逐字段说明、`depthConfidence` 算法、版本兼容规则 |
| `docs/motion-clip.md` | 动作 JSON 字段、关节映射、重采样规则与限制 |
| `docs/benchmark.md` | 吞吐基准方法与历史结果、深度 / Z 平滑 / yaw 基准记录 |
| `docs/todo-depth-z-accuracy.md` | 深度精度改进计划与阶段性记录 |
| `samples/benchmark/README.md` | BM-01 至 BM-04 基准视频素材清单 |
| `samples/RealityKitSample.swift` | iOS / macOS 端解码与播放参考实现 |
| `TODO.md` | 历史任务清单，进度可能滞后 |

---

## 许可证

`pyproject.toml` 中声明项目许可证为 MIT。当前仓库根目录未包含 `LICENSE` 文件，若需要对外分发，请补充完整的 MIT 许可证文本及版权声明。
