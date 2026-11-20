# DetectBodyInVideo — 开发任务清单

> 目标：一个**跨平台（Windows / macOS / Linux）的可视化桌面应用**。用户拖入一段含人物的视频，应用逐帧识别人体关键点、记录关键点位置与关节角度，最终导出**可直接喂给 iOS / macOS RealityKit 的 JSON**，用于驱动 USDZ 模型播放动作。
>
> 使用方式：勾选 `- [ ]` 为 `- [x]` 表示完成。任务按依赖顺序排列，未完成前置任务时不要跳到后续阶段。

**进度：38 / 50 个主任务**（子任务见缩进；完成一项后同步更新）

---

## 0. 项目速览（先读，再动手）

### 0.1 技术选型（已决策，实现时不要随意更换）

| 层 | 选择 | 理由 |
| --- | --- | --- |
| 运行时 | Python >=3.9（无上限） | MediaPipe 1.2+ 提供 py3 通用 wheel，覆盖 3.9–3.14 |
| 本地引擎服务 | FastAPI + Uvicorn（仅监听 127.0.0.1，随机端口 + 一次性 token） | 前后端解耦，UI 与算法可独立迭代 |
| 姿态推理（兜底） | MediaPipe Tasks `PoseLandmarker`（33 点 + world landmarks） | 零配置、跨三平台稳定、CPU 即可跑 |
| 姿态推理（加速） | ONNX Runtime + RTMPose / RTMO | 支持 CUDA、CoreML、DirectML，满足"自动加速"需求 |
| 硬件探测 | `nvidia-smi` / `pynvml`、`platform`、`onnxruntime.get_available_providers()` | 判定可用执行提供器（EP） |
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS | 现代、易做出好看界面 |
| 2D 骨架叠加 | Canvas 2D | 逐帧绘制 33 点 + 连线，性能足够 |
| 3D 骨骼预览 | three.js | 让用户直观确认 3D 数据可用 |
| 桌面外壳 | pywebview（Win: WebView2 / mac: WKWebView / Linux: WebKitGTK） | 单进程、无需 Node 运行时、打包体积小 |
| 打包 | PyInstaller（Win/Linux）、PyInstaller + codesign/notarize（macOS） | 一套脚本三平台产物 |

> 备选：若 pywebview 在某平台 WebView 兼容性踩坑，降级为"启动后自动打开系统浏览器 + 本地服务"模式（功能等价，只是不是原生窗口）。

### 0.2 坐标与单位约定（贯穿全项目，写进代码常量）

- MediaPipe 归一化坐标：`x, y ∈ [0,1]`（相对图像宽高，`y` 向下），`z` 为相对髋中心的深度（量级≈x）。
- MediaPipe world landmarks：**米**，原点≈两髋中点，`y` 向下，`z` 向相机后方为正。
- RealityKit：**米**，右手坐标系，`y` 向上，相机朝 `-z` 看。
- 因此导出时统一做：`x' = x`、`y' = -y`、`z' = -z`，并在 JSON 里显式写 `"coordinateSystem": "arkit-right-handed-y-up"`。

### 0.3 目录结构（初始化时就按此建空目录）

```
DetectBodyInVideo/
├─ app/
│  ├─ backend/            # FastAPI 服务
│  │  ├─ main.py
│  │  ├─ api/             # 路由：upload / analyze / export / hardware
│  │  ├─ core/            # 配置、日志、坐标转换
│  │  ├─ engine/          # 硬件探测 + 推理引擎抽象与实现
│  │  ├─ postprocess/     # 平滑、补帧、角度
│  │  └─ export/          # JSON 导出与 schema
│  ├─ frontend/           # Vite + React + Tailwind
│  └─ desktop/            # pywebview 启动器
├─ models/                # 本地模型权重（gitignore）
├─ samples/               # 测试视频
├─ tests/
├─ scripts/               # dev / build / package 脚本
└─ TODO.md
```

---

## 1. 工程初始化

- [x] 1.1 建立仓库骨架
  - [x] 创建上述目录结构，保留 `.gitkeep`
  - [x] `git init`，配置 `.gitignore`（`models/`、`dist/`、`build/`、`*.mp4`、`.venv/`、`node_modules/`）
  - [x] 写 `README.md`（定位、截图位、三平台运行方式占位）
- [x] 1.2 Python 侧依赖与配置
  - [x] `pyproject.toml`：`fastapi`、`uvicorn[standard]`、`python-multipart`、`opencv-python`、`numpy`、`mediapipe`、`onnxruntime`、`pywebview`、`pydantic-settings`
  - [x] 分平台 extras：`[cuda]` 装 `onnxruntime-gpu`；`[mac]` 装 `onnxruntime-silicon`（或统一走 `onnxruntime` 的 CoreML EP）
  - [x] `app/backend/core/settings.py`：数据目录、临时目录、模型目录、日志级别
  - [x] 统一日志：控制台 + 文件（`~/.detect-body/logs/`）
- [x] 1.3 前端脚手架
  - [x] `npm create vite@latest frontend -- --template react-ts`
  - [x] 接入 Tailwind CSS，配置主题色板与字体
  - [x] 配置 Vite dev proxy 指向 `http://127.0.0.1:8000`
  - [x] 建立 `src/api/`（fetch 封装）、`src/store/`（全局状态）、`src/components/`
- [x] 1.4 一键开发脚本
  - [x] `scripts/dev.ps1` / `scripts/dev.sh`：并行起 uvicorn + vite
  - [x] `scripts/bootstrap.ps1` / `bootstrap.sh`：建 venv、装依赖、下载模型

## 2. 硬件与算力自检（"自动检测电脑版本"）

- [x] 2.1 系统信息采集
  - [x] 平台与架构：`platform.system()`、`machine`、macOS 版本、Windows 版本、glibc
  - [x] CPU：型号、物理核数、AVX2/AVX512 支持
  - [x] 内存总量与可用值
  - [x] 输出结构化 `SystemInfo`（Pydantic 模型）
- [x] 2.2 GPU / 加速器探测
  - [x] NVIDIA：`pynvml` 取型号、显存、驱动版本；`nvidia-smi` 兜底取 CUDA 版本
  - [x] Apple Silicon：`platform.processor() == "arm"` + `sysctl` 判断 M 系列，读 `chip` 型号与统一内存
  - [x] Windows 通用 GPU：`onnxruntime` DirectML 可用性
  - [x] 枚举 `onnxruntime.get_available_providers()`，与硬件交叉校验
- [x] 2.3 加速能力决策表
  - [x] 定义优先级：`TensorRT > CUDA > CoreML > DirectML > CPU(多线程)`
  - [x] 输出 `AccelerationPlan`：首选 EP、线程数、推荐模型档位（lite / full / heavy）
  - [x] 写单元测试：mock 各种硬件组合，断言决策结果正确
- [x] 2.4 对外暴露
  - [x] `GET /api/hardware` 返回完整能力报告
  - [x] 前端启动时先请求该接口，顶部显示"加速状态"徽章（绿色=GPU 加速 / 黄色=CPU / 红=不可用原因）

## 3. 视频接入与解码

- [x] 3.1 上传与校验
  - [x] 拖拽上传 + 文件选择两种入口
  - [x] 校验容器/编码（mp4/mov/avi/mkv，H.264/H.265/HEVC），拒绝时给出明确原因
  - [x] 大文件分块上传 + 进度条；落盘到临时目录并计算 hash 去重
- [x] 3.2 元信息解析
  - [x] 用 OpenCV 读取帧率、总帧数、分辨率、时长、旋转元数据（手机竖拍需纠正）
  - [x] 抽 1 帧生成缩略图返回前端
- [x] 3.3 帧读取器
  - [x] 顺序读取 + 可指定区间（起止时间码）
  - [x] 处理可变帧率（VFR）：统一按时间戳重采样到固定 fps
  - [x] 内存上限保护：不整段读入内存，流式逐帧
- [x] 3.4 采样策略
  - [x] 默认全帧分析；提供"每 N 帧"降采样开关
  - [x] 可选"自动裁掉无人物区间"（用轻量检测先粗筛）

## 4. 推理引擎与关键点识别

- [x] 4.1 引擎抽象层
  - [x] 定义 `PoseEngine` 接口：`prepare(device)` / `infer(frame) -> PoseFrame` / `close()`
  - [x] `PoseFrame` 数据结构：`index`、`timestamp_ms`、`landmarks[33]`（x,y,z,visibility）、`world_landmarks[33]`、`bbox`、`score`
  - [x] 引擎注册表 + 按 `AccelerationPlan` 自动选型，失败自动降级并记录原因
- [x] 4.2 MediaPipe 引擎（兜底，必做）
  - [x] 集成 `mediapipe.tasks.python.vision.PoseLandmarker`
  - [x] 支持 `model_complexity` 三档（lite / full / heavy）
  - [x] 开启 `smooth_landmarks`、`min_detection_confidence` / `min_tracking_confidence` 可配
  - [x] 输出 33 点归一化坐标 + world landmarks
- [x] 4.3 ONNX 加速引擎（可选路径，做成就自动启用）
  - [x] 集成 RTMPose（单人）/ RTMO（多人）ONNX 模型
  - [x] 按 EP 创建 `InferenceSession`（CUDA / CoreML / DirectML / CPU），线程数由 `AccelerationPlan` 决定
  - [x] IoU 跟踪器把跨帧检测串成 track，支持多人物
  - [x] 与 MediaPipe 结果做一次一致性对比（同视频、同帧，误差阈值内视为等价）
- [x] 4.4 多人处理
  - [x] 多人 track 管理（id、存活帧数、丢失重建）
  - [ ] UI 中可切换"跟随目标人物"，默认选出现时间最长的 track
- [x] 4.5 手指 / 面部（可选增强，二期）
  - [x] MediaPipe Hand Landmarker（每手 21 点）
  - [x] Face Landmarker（468 点，仅导出精简子集）

## 5. 后处理（让数据能真正驱动模型）

- [x] 5.1 平滑去抖
  - [x] One-Euro Filter（默认，参数可调）
  - [x] 可选 Savitzky-Golay 或移动平均作为对比
  - [x] 只平滑位置，不平滑 visibility；导出时记录所用滤波器与参数
- [x] 5.2 缺失帧处理
  - [x] 置信度低于阈值或检测失败的帧标记 `valid=false`
  - [x] 短缺口（≤N 帧）线性插值补齐，长缺口保持缺失
  - [x] 不伪造长时间缺失数据，避免 iOS 端模型乱抖
- [x] 5.3 关节角度计算
  - [x] 实现三点夹角工具（肩-肘-腕、髋-膝-踝等）
  - [x] 输出主要关节角度序列：左右肘、左右肩、左右膝、左右髋、躯干前倾
  - [x] 处理退化解（三点共线 / 零长度向量）不产生 NaN
- [x] 5.4 数据质检
  - [x] 统计有效帧比例、平均置信度、最大跳变（单帧位移异常）
  - [x] 生成"质量报告"对象，随 JSON 一起导出，UI 里给出"可直接使用 / 建议重录"结论

## 6. 骨骼映射与 RealityKit 契约

- [x] 6.1 建立映射表
  - [x] MediaPipe 33 点 → ARKit `ARSkeleton.JointName` 18 关节映射表（代码常量 + 文档表格）
  - [x] 缺失关节（如 spine 中段）用髋-颈插值推导
  - [x] 左右手/脚不做镜像混淆，写测试断言 left/right 不串
- [x] 6.2 坐标转换模块
  - [x] 归一化坐标 → 世界米制（按画面宽高比与人物身高标定反解）
  - [x] 统一转为 `y-up、右手系、-z 向前`
  - [x] 提供 `personHeightMeters` 参数（可在 UI 中设定，默认 1.70 m）
- [x] 6.3 层级与根运动
  - [x] 计算 root（髋中心）位移与旋转，单独输出 `rootMotion`
  - [x] 提供"原地播放（固定 root）"与"带位移播放"两种导出模式

## 7. JSON 导出（最终交付物）

- [x] 7.1 定义并冻结 Schema
  - [x] 写 `app/backend/export/schema.py`（Pydantic v2），生成 `schema.json` 存档
  - [x] 顶层字段：`version`、`generator`、`source`（文件/fps/分辨率/帧数）、`coordinateSystem`、`units`、`personHeightMeters`、`jointNames`、`frames[]`、`quality`
  - [x] 每帧字段：`index`、`t`（秒）、`valid`、`joints{ name: {x,y,z,visibility} }`、`angles{}`
  - [x] 提供 `schemaVersion = "1.0.0"`，后续不兼容变更必须升大版本
- [x] 7.2 导出实现
  - [x] 三种粒度：仅关键点 / 关键点+角度 / 精简（只留 18 个 ARKit 关节）
  - [x] 浮点压缩到 4 位小数，控制文件体积
  - [x] 大文件走流式写出，避免内存峰值
  - [x] 文件名规则：`{原名}_{时间戳}_pose{mode}.json`，输出到用户选择的目录
  - [x] 额外导出 iOS App 动作 JSON（`hip-local-meters-yup`、10Hz、`pelvisOffsetMeters`），格式对齐 `resources/air-squat-motion.json`
- [x] 7.3 自校验
  - [x] 导出后立刻回读并用 Pydantic 校验，格式错误要立刻报错而非静默产出
  - [x] 样例数据入库 `samples/expected_pose.json`，作为回归基准
- [x] 7.4 配套示例
  - [x] `samples/RealityKitSample.swift`：演示如何解码 JSON、按时间轴在 `Scene.update` 里对 `ModelEntity` 骨骼做姿态插值
  - [x] 在 README 中说明 iOS 侧必须的坐标/单位一致性检查
  - [x] `docs/motion-clip.md`：动作 JSON 字段、关节映射、骨盆升降来源与已知限制

## 8. 可视化界面（要求"样式好看"）

- [x] 8.1 视觉规范
  - [x] 深色为主基调 + 单一高饱和强调色（如青绿 `#22D3EE`）
  - [x] 圆角 12px、卡片式布局、柔和阴影、1px 描边代替重阴影
  - [x] 字号层级固定（12/14/16/20/28），间距使用 4 的倍数
  - [x] 所有按钮/输入框有 hover、focus-visible、disabled 三态
- [x] 8.2 布局
  - [x] 左侧：视频舞台（16:9，黑底居中，骨架叠加层）
  - [x] 右侧：分析面板（硬件状态、参数、进度、质量报告）
  - [x] 底部：时间轴（可拖拽 seek，标记缺失帧）
  - [x] 顶部：文件信息 + 导出按钮 + 主题切换
- [x] 8.3 关键交互
  - [x] 拖拽上传，拖入时全屏虚化高亮
  - [x] 分析进度用真实帧数推进（不是假动画），显示 fps 与预计剩余时间
  - [x] 播放/暂停/逐帧、倍速、循环区间
  - [x] 骨架叠加开关、显示关键点编号开关、连线样式切换
  - [x] 悬停关键点时浮出该点坐标与置信度
- [x] 8.4 空态与反馈
  - [x] 空态插画/提示："拖入一段含人物的视频开始"
  - [x] 加载骨架屏，禁止白屏
  - [x] 所有错误用 toast + 面板内详情，不弹 `alert`
- [x] 8.5 3D 预览（可选但推荐）
  - [x] three.js 场景：简易胶囊/球棍骨架按 world landmarks 渲染
  - [x] 与 2D 视频同步播放
  - [x] 显示地面网格与坐标轴，方便判断 `y-up` 是否正确
- [x] 8.6 响应式与可访问性
  - [x] 1280×800 起可用，窗口缩到 1024 宽不塌
  - [x] 键盘可完成上传/播放/导出主流程
  - [x] 对比度满足 WCAG AA

## 9. 性能与加速落地

- [x] 9.1 推理链路优化
  - [x] 帧读取与推理分离为生产者/消费者线程（有界队列，背压保护）
  - [x] 批处理（batch=4/8）在加速 EP 上启用
  - [x] 半精度（FP16）在 CUDA / CoreML 上尝试，失败回退 FP32
- [x] 9.2 基准测试
  - [x] `scripts/bench.py`：固定样例视频，输出各档模型的 fps 与内存峰值
  - [x] 结果写入 `docs/benchmark.md`，作为"加速是否生效"的证据
- [x] 9.3 资源控制
  - [x] 可配置并发度，避免占满 CPU 导致 UI 卡顿
  - [x] 任务可取消（取消后立刻释放线程与显存）
  - [x] 长视频分段处理，支持断点续算

## 10. 桌面外壳与三平台打包

- [x] 10.1 桌面壳
  - [x] `app/desktop/main.py`：启动 uvicorn（子线程/子进程）→ 等健康检查 → 打开 pywebview 窗口
  - [x] 前端构建产物由后端静态目录托管，单端口访问
  - [x] 窗口尺寸/最小尺寸、标题、图标、关闭时优雅退出（杀掉 Python 子进程）
- [ ] 10.2 构建脚本
  - [ ] `scripts/build_frontend.*`：`npm ci && npm run build` → 拷贝到 `app/backend/static`
  - [ ] `scripts/package.*`：PyInstaller `--onedir`（比 `--onefile` 启动快、更稳）
  - [ ] 处理 MediaPipe / onnxruntime 的动态库与模型文件打包（hidden-imports、datas）
- [ ] 10.3 平台验证
  - [ ] Windows 10/11：WebView2 依赖检测与提示
  - [ ] macOS：`--windowed` + icon，Universal（arm64 + x86_64）或仅 arm64
  - [ ] macOS：codesign + notarize 流程（无签名时文档说明如何绕过 Gatekeeper）
  - [ ] Linux：AppImage（需 WebKitGTK 运行时说明）或提供 `.deb`
- [ ] 10.4 首次运行引导
  - [ ] 检查模型文件是否存在，缺失则内嵌下载器（带进度与镜像地址）
  - [ ] 首启展示硬件报告页，让用户确认加速状态

## 11. 测试与验收

- [ ] 11.1 单元测试
  - [ ] 硬件探测决策（mock）
  - [ ] 平滑滤波器数值正确性（已知输入 → 已知输出）
  - [ ] 角度计算（直角、共线、零向量）
  - [ ] 坐标转换往返一致性（转过去再转回来误差 < 1e-6）
  - [ ] JSON schema 校验与向后兼容
- [ ] 11.2 集成测试
  - [ ] 3 段样例视频（单人全身、多人、快速运动）跑通全流程
  - [ ] 断言有效帧比例、导出文件大小、字段完整性
- [ ] 11.3 端到端验收（对着真实 iOS App 验）
  - [ ] 导出的 JSON 能被 iOS 端 RealityKit 正确解析
  - [ ] 模型动作与视频肉眼一致（无明显镜像、无左右反、无上下颠倒）
  - [ ] 播放无抖动、无穿模级跳变
- [ ] 11.4 三平台冒烟
  - [ ] Windows（含无 NVIDIA 的机器）跑通
  - [ ] macOS Apple Silicon 跑通并确认 CoreML/MPS 生效
  - [ ] Linux 跑通

## 12. 文档

- [ ] 12.1 README：功能截图、安装、三种运行方式（源码/安装包/开发模式）
- [ ] 12.2 `docs/json-schema.md`：字段说明 + ARKit 18 关节映射表 + 完整样例
- [ ] 12.3 `docs/acceleration.md`：三平台加速矩阵、常见失败原因与降级行为
- [ ] 12.4 `docs/troubleshooting.md`：模型下载失败、WebView 缺失、显存不足、编码不支持

---

## 附录 A：导出 JSON 草案（实现以 Pydantic Schema 为准）

```json
{
  "version": "1.0.0",
  "generator": "DetectBodyInVideo 0.1.0",
  "source": { "file": "dance.mp4", "fps": 30, "width": 1080, "height": 1920, "frames": 900 },
  "coordinateSystem": "arkit-right-handed-y-up",
  "units": "meters",
  "personHeightMeters": 1.7,
  "jointNames": ["root", "hips_joint", "spine", "neck_1_joint", "head", "left_shoulder", "..."],
  "frames": [
    {
      "index": 0,
      "t": 0.0,
      "valid": true,
      "joints": {
        "root": { "x": 0.0, "y": 0.92, "z": 0.0, "visibility": 0.98 },
        "left_shoulder": { "x": 0.18, "y": 1.38, "z": 0.02, "visibility": 0.96 }
      },
      "angles": { "left_elbow": 172.4, "right_elbow": 168.1, "left_knee": 175.0 }
    }
  ],
  "quality": { "validFrameRatio": 0.996, "meanVisibility": 0.94, "smoothing": "one-euro", "verdict": "ready" }
}
```

## 附录 B：MediaPipe 33 点 → ARKit 关节映射（初稿）

| MediaPipe | ARKit JointName |
| --- | --- |
| 23/24 hip | `hips_joint`（root） |
| 11/12 shoulder | `left_shoulder_1_joint` / `right_shoulder_1_joint` |
| 13/14 elbow | `left_arm_joint` / `right_arm_joint` |
| 15/16 wrist | `left_hand_joint` / `right_hand_joint` |
| 25/26 knee | `left_leg_joint` / `right_leg_joint` |
| 27/28 ankle | `left_foot_joint` / `right_foot_joint` |
| 31/32 foot_index | `left_toes_joint` / `right_toes_joint` |
| 0 nose | `head_joint`（头部朝向需结合耳/眼推算） |
| 11-12 中点 | `spine_3_joint` |
| 23-24 中点 → 颈 | `spine_7_joint` |

## 附录 C：风险与未决项

- [ ] MediaPipe Python 侧 GPU delegate 支持不稳定 → 依赖 ONNX 加速引擎兜底，需在 4.3 完成后实测确认
- [ ] pywebview 在部分 Linux 发行版缺 WebKitGTK → 提供浏览器模式降级
- [ ] iOS 端骨骼命名需以实际 USDZ 模型为准，映射表可能需二次映射层（导出时保留原始 33 点以便重映射）
- [ ] 手指与面部（4.5）不定为一期目标，避免拖慢主线
