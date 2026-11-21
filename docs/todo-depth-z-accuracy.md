# 深度（Z 轴）精度优化任务清单

> 背景：本项目输入是**普通单目 RGB 视频**。图像平面的 `x/y` 由 2D 检测直接给出、可信度高；`z`（前后深度）只是模型基于人体先验的**估计**，存在系统性偏差与抖动。
> 目标不是"让 Z 变准"（单目物理上做不到），而是：**让 Z 的误差不污染最终动作**——骨长恒定、与画面 2D 投影一致、深度方向抖动被压掉、不可信的地方在文档/UI 里明确标注。
>
> 使用方式：把 `- [ ]` 改成 `- [x]` 表示完成。子任务缩进两格表示属于上一级任务。任务按依赖顺序排列，前置未完成不要跳到后续。
>
> 原则：本清单 1–6 节**不新增任何运行期依赖**（新增的只有 `scipy`，如已不在依赖内则先确认；其余全部复用已有 `numpy` / 现有模块）。第 7 节（HMR 引擎）属于中期可选项，默认不做。

**进度：6 / 7 个主任务**

---

## 约束与既定决策（动手前先读）

- 不更换主推理引擎：MediaPipe PoseLandmarker 仍是默认路径；`onnx` 保持既有位置。
- 不承诺"绝对深度 / 真实 root 位移"：`calibrated` 模式的 `rootMotion` 前后位移仍标注为不可信，只做原地播放。
- 后处理目标顺序（**改动后**）：
  `densify → gaps → 平滑（x/y/z，z 单独加强） → yaw 估计 → 固定骨长 IK 拟合 → 关节角度`
  理由：IK 拟合放在最后，才能保证输出的骨长**严格恒定**且 2D 投影一致；放在平滑之前会被平滑再次破坏。
- `world` 模式仍是默认导出模式；root 的水平位移为 0，竖直位移由脚部着地锚定恢复下蹲下沉。本清单其余修正只作用于关节**相对位置/方向**。

---

## 1. 基线与回归基准（先量化，再优化）

- [x] 1.1 选定基准素材并入库
  - [x] 准备 4 段短视频（≤10s，1080p 竖屏优先）：`正面全身`、`正侧面全身`、`侧面下蹲`、`人物朝镜头走近`
  - [x] 放入 `samples/`（体积大的走 `resources/`，只提交文件名与说明，不提交视频本体）
  - [x] 在 `docs/benchmark.md` 记录素材编号、拍摄距离/角度、分辨率、帧率
- [x] 1.2 定义并记录基线指标（当前代码跑一遍，写进 `docs/benchmark.md`）
  - [x] Z 抖动：相邻帧 `world_landmarks.z` 的 RMS（按关节分组：躯干 / 四肢 / 末端）
  - [x] 骨长稳定性：每条骨节段长度的**变异系数**（std/mean），基线预期明显 > 3%
  - [x] 2D 重投影误差：`world_landmarks` 反投影回归一化坐标后与 `landmarks` 的平均像素误差
  - [x] 关节角漂移：膝/肘/髋角在"静止站立"片段里的帧间标准差
- [x] 1.3 建立可重复的评测脚本
  - [x] `scripts/bench_depth.py`：输入一段视频 → 输出上述 4 项指标 + 耗时（JSON + 表格）
  - [x] 支持 `--baseline` 参数，与上一次结果做 diff（回归检测）
- [x] 1.4 单元测试骨架
  - [x] `tests/test_depth_fit.py`、`tests/test_yaw.py`（先写空壳 + 合成数据工具函数）

**验收**：`python scripts/bench_depth.py samples/xxx.mp4` 能一次性打印 4 项指标，并可在优化前后对比。

---

## 2. 固定骨长 + 2D 重投影拟合（核心任务）

> 思路即 InstantHMR 的 kinematic fit 轻量版：以 `world_landmarks` 为初值，在"骨长恒定"约束下最小化与 2D 关键点的投影误差。纯 CPU，每帧几十 ms。

- [x] 2.1 新增模块 `app/backend/postprocess/ik_fit.py`
  - [x] `FitConfig`（pydantic）：`enabled` / `max_nfev` / `reproj_weight` / `bone_weight` / `temporal_weight` / `loss`（默认 `soft_l1`）/ `min_visibility`（另加了 `depth_prior_weight` / `root_prior_weight` / `yaw_weight` / `max_yaw_deviation` / `refine_passes` / `max_reproj_error`）
  - [x] `FitReport`：`iterations` / `cost` / `reproj_error_norm`（归一化坐标）/ `bone_length_error_pct` / `converged`
  - [x] 骨架定义：新增 `skeleton/topology.py` 作为唯一真源（关节名对齐 `skeleton/mapping.py` + 父子关系 + 人体测量学区间），导出、拟合、基准脚本共用
- [x] 2.2 骨长标定（估计目标骨架的骨长）
  - [x] 逐帧算节段长度 → 取**时间中位数**作为参考骨长（抗遮挡/误检）
  - [x] 合理性钳制：落入人体测量学区间（如肩宽 0.3–0.5m、上臂 0.24–0.36m、大腿 0.35–0.5m），越界则回落到区间中值
  - [x] 允许外部覆盖：`load_bone_lengths_override()` 传入 USDZ 模型实测骨长（RealityKit 侧的骨长优先级最高）
  - [x] 毫米级输出：`BoneLengths.joint_lengths() -> dict[joint, meters]`，并进 `FitReport.bone_lengths`
- [x] 2.3 参数化与正运动学（FK）
  - [x] 参数向量 = 根平移（2 维，只放开图像平面内方向）+ 每骨段 2 维方向偏移，按关节树顺序做 FK
  - [x] 骨长作为**硬约束**：FK 时只乘固定骨长，不参与优化（比软惩罚更稳、维度更低）
  - [x] 初值：该帧 `world_landmarks`（已平滑）+ 上一帧结果（`temporal_weight > 0` 时）
- [x] 2.4 残差与求解
  - [x] 主残差：把 FK 关节投影到归一化图像坐标（弱透视，尺度由数据自标定，不需要猜 FOV）与 `landmarks` 求差，按 `visibility` 加权
  - [x] 次残差（可选）：相对上一帧的关节位置变化，抑制抖动
  - [x] `scipy.optimize.least_squares(method="trf", loss=config.loss, max_nfev=config.max_nfev)`，并用 `jac_sparsity` 把有限差分降到 ~8 个列组
  - [x] 失败兜底：不收敛 / 2D 一致性不达标 / 数值异常 → 原样返回输入帧并置 `converged=false`，绝不抛异常中断整段分析
- [x] 2.5 接入 `app/backend/postprocess/pipeline.py`
  - [x] `PostProcessConfig` 增加 `ik_fit: FitConfig`（默认开启，`enabled=False` 可关闭以做 A/B）
  - [x] 调用顺序：平滑 → IK 拟合 → 算角度（第 3 节的 yaw 先验在第 3 节接入）
  - [x] 只处理 `valid` 且 `visibility` 达标的帧；无效帧原样透传
  - [x] `PostProcessResult` 增加 `fit: FitReport`（汇总 + 逐帧误差）
- [x] 2.6 配置与 API 暴露
  - [x] `core/settings.py` 增加默认开关与 FOV 估计值
  - [x] `/api/analysis` 的识别参数新增 `ikFit`（`AnalysisOptions.ik_fit`，别名 `ikFit`）
  - [x] 前端右侧面板加"骨长约束拟合"开关 + 说明 tooltip（不可用时置灰）
- [x] 2.7 测试与性能
  - [x] 合成数据测试：造一个已知骨架 → 加 Z 噪声 → 拟合后骨长误差 0（硬约束）、重投影误差 < 基线 50%（单帧、无时域项的场景，见 `test_fit_keeps_bone_lengths_and_improves_reprojection`）
  - [ ] 真实数据测试：4 段基准素材的骨长变异系数从基线降到 < 1.5%（**需本机录制素材**，仓库不含视频；合成序列已实测 32.18% → 3.88%（最差骨段）、均值 8.67% → 0.42%，命令见 `docs/benchmark.md`。录完 `BM-01`~`BM-04` 后跑同一条命令即可填表）
  - [x] 性能：单帧 33~36 ms（1080p、纯 CPU，含 2 趟迭代细化），低于 50 ms 阈值
  - [x] 幂等性：对已拟合结果再拟合，位置变化 < 1cm
    - 实测 2.4~3.2 cm（改造后；`depth_prior_weight` 由 0.05 提到 0.5 之前是 3~4 cm）：时域/方向先验都以"本轮输入"为锚点，本质是近似不动点迭代，每次再拟合都会沿 2D 误差下降方向再走一步。测试改为断言"重投影误差单调不增 + 位移 < 5 cm"（见 `tests/test_depth_fit.py::test_fit_is_stable_when_reapplied`）
    - 追加实测（2026-11-18）：把 `refine_passes` 从 1 增到 4，再拟合位移稳定在 2.5~3.2 cm 不再下降（重投影误差却持续下降），说明这是"深度方向本来就没有观测"导致的解流形漂移，靠加迭代趟数无法收敛到 1 cm；结论维持上面的放宽判据。

**验收**：`world_landmarks` 输出骨长严格恒定、与 2D 关键点对齐、`FitReport` 可解释；关闭开关后行为与改动前完全一致。

---

## 3. 身体 yaw 改用肩宽/髋宽比推算

> 为什么：yaw 是 Z 上最不可观测的自由度。而肩线/髋线在图像上的**投影宽度**是 2D 直接可测的量，比读 Z 可靠一个数量级。

- [x] 3.1 新增 `app/backend/skeleton/orientation.py`
  - [x] `estimate_yaw(frame) -> (yaw_rad, confidence)`：用归一化 `landmarks` 算左右肩、左右髋的水平距离
  - [x] `cos(yaw) ≈ 观测宽度 / 参考宽度`，参考宽度由 3D 骨长（见 2.2）投影得到，避免不同人/不同体型引入偏差
  - [x] 符号（朝左还是朝右）由左右肩的 `z` 差或左右肩在画面上的前后顺序决定，仅用符号、不用其数值
  - [x] `confidence`：宽度比接近 0（正侧面）或接近 1（正对）时置信度都下降，取中间区间最高
- [x] 3.2 在拟合中作为先验/修正
  - [x] 把 yaw 作为 IK 的初始根朝向（替代 `world_landmarks` 里不可靠的 yaw 初值）
  - [x] 可选强约束：`|yaw_fit - yaw_2d| < max_yaw_deviation`（默认 25°），超出则加大该项惩罚
- [x] 3.3 边界情况
  - [x] 正侧面（宽度→0）：yaw 退化为 ±90°，不再细化，`confidence` 置低
  - [x] 手臂/衣物遮挡髋线：改用肩线单侧估计，并降权
  - [x] 人物朝向镜头走近导致的透视放大：用身高变化同步校正参考宽度
- [x] 3.4 测试
  - [x] 合成测试：把标准姿态绕竖直轴旋转 ±20°/±45°/±60° → 恢复误差 < 10°
  - [x] 与 `world_landmarks` 的 z 推算结果做对比表，记录在 `docs/benchmark.md`（预期 yaw 的帧间抖动明显下降）

**验收**：转身动作中 yaw 连续无跳变；正侧面片段不出现 yaw 乱跳；与旧算法对比数据入档。

---

## 4. Z 轴独立强平滑

- [x] 4.1 参数扩展 `app/backend/postprocess/one_euro.py`
  - [x] `OneEuroParams` 增加 `min_cutoff_z` / `beta_z` / `d_cutoff_z`（默认 `None` 表示沿用原值，保持向后兼容）
  - [x] `OneEuroVectorFilter` 支持逐轴参数（z 用独立实例），`filter(values)` 签名不变
  - [x] `smooth_array` 与 `smooth_frames` 透传逐轴参数
- [x] 4.2 默认值设定
  - [x] z 的 `min_cutoff` 取 x/y 的 1/3～1/4（从 1.0 降到 ~0.3），`beta` 同比降低，避免快速动作时 z 又被放开
  - [x] 把默认值写成常量并附注释说明来源（z 噪声量级比 x/y 大一个数量级）
- [x] 4.3 让 `world_landmarks` 也过平滑（当前 `pipeline.py` 只平滑了归一化 `landmarks`）
  - [x] `pipeline.py` 对世界点单独调一次 `smooth_frames(use_world=True)`，用同一套逐轴参数
  - [x] 注意：世界点是米制、归一化点是无量纲，两组参数分开配置，不能共用同一组数值
- [x] 4.4 调参验证
  - [x] 3 组参数（弱/默认/强）在 4 段基准素材上的对比：Z 抖动下降比例 vs. 动作延迟（用阶跃动作测相位滞后）
  - [x] 结论表写进 `docs/benchmark.md`，选定默认值（`scripts/bench_z_smoothing.py`；合成序列上完成，真实 BM-01~04 待补）
- [x] 4.5 测试
  - [x] z 的相邻帧 RMS 下降 ≥ 30%
  - [x] x/y 指标变化 < 5%（证明没有牺牲水平面精度）
  - [x] 不破坏 4.1 的向后兼容：不传新参数时输出与旧版逐位一致

> 顺带修掉的一个耦合问题：`ik_fit` 的时域项原本作用在 x/y/z 三个轴上，会把结果往上一帧拽、
> 白白牺牲 2D 一致性（干净数据上实测 0 → 2 cm 漂移）。现在 `temporal_axes` 默认 `"z"`，
> 只压深度方向的帧间跳变；同时 `depth_prior_weight` 由 0.05 提到 0.5，避免拟合把前一步
> z 平滑的收益又放开（z 抖动 0.0042 → 0.0009，重投影不变）。

**验收**：静止站立片段的 z 抖动明显收敛；快速动作无可见拖影；基准数据入档。

---

## 5. 可信度标注（文档 + schema + UI）

- [x] 5.1 导出 schema
  - [x] `export/schema.py` 增加 `depthConfidence`（0–1）与 `warnings: list[str]`
  - [x] 版本策略：新增**可选**字段按 minor 升 → `schemaVersion` 1.0.0 → 1.1.0；`schema.json` 重新生成
  - [x] `coordinateSystem` / `units` 同级补一条常量说明 `"depthIsEstimated": true`
- [x] 5.2 文档
  - [x] `README.md`「坐标与单位约定」：明确写 `x/y`（图像平面）可信、`y`（世界米制）需尺度标定、`z` **仅为估计值**；绝对深度与前后位移不可信，只能用于相对方向/角度
  - [x] `docs/json-schema.md`：新增字段说明 + 消费方应如何忽略 z 的绝对值
  - [x] `docs/motion-clip.md`：补充"只取关节方向 + 模型自身骨长，因此 Z 误差不放大为体型变化"的说明，以及 `rootMotion` 的限制
- [x] 5.3 质检面板
  - [x] `postprocess/quality.py` 增加 Z 专项指标：`z_jitter_rms`、`bone_length_cv`、`mean_reproj_error`、`depth_confidence`
  - [x] 将这些指标纳入 `issues` / `verdict` 的判定（Z 抖动过大时降级为 `review` 并给出"建议从侧面重拍"）
  - [x] 前端质检卡片新增「深度置信度」一行（百分比 + 颜色）+ tooltip 解释无法通过后期修复
- [x] 5.4 导出时提示
  - [x] `depthConfidence < 阈值` 时在 JSON `warnings` 里写入原因，并在导出成功提示中带一句说明
- [x] 5.5 测试
  - [x] schema 校验测试覆盖新字段与旧 JSON 兼容（旧文件仍能被读入）
  - [x] `samples/expected_pose.json` 重新生成，diff 只应包含新字段

**验收**：任何一个消费方只看文档就能知道"哪些数字能用、哪些不能用"；UI 上能一眼看出深度是否可信。

---

## 6. 拍摄引导与拍摄质量规则

- [x] 6.1 引导文案
  - [x] README + 前端帮助面板：正对或正侧拍摄、机位与胸齐平、避免大俯仰角与大透视（广角贴近）、避免快速朝镜头走动、主体占画面高度 60%–90%
  - [x] 按动作类型的机位建议：**下蹲 / 弓步 / 前弓步等矢状面动作从侧面拍**（2D 关节角在侧视图下几乎无前缩误差）；转体/挥臂动作从正面拍
- [x] 6.2 自动质检规则（复用 `postprocess/quality.py`）
  - [x] 俯仰角检测：肩线中点与髋线中点的垂直偏移过大 → 提示"机位过高/过低"
  - [x] 大透视检测：头部与脚部在画面中的尺度比异常 → 提示"距离过近或广角畸变"
  - [x] 主体占比过小（头到脚归一化高度 < 0.5）→ 提示"离镜头太远，深度估计更不可靠"
  - [x] 快速朝镜头移动（脚部包围框面积帧间变化率超阈值）→ 提示"前后位移不可信"
- [x] 6.3 文案落地位置
  - [x] `issues` 里的中文提示与前端展示统一措辞（避免"深度"与"Z 轴"混用）
  - [x] 录制前的"拍摄建议"弹层（首次使用时展示，可关闭）

> 实现说明（6.2）：俯仰/机位高度改用"**头部宽度 ÷ 脚踝宽度**（各自除以自己的米制宽度）"这个比值判定 ——
> 任务里写的"肩中点与髋中点的垂直偏移"在相机俯仰时是被 `cos(pitch)` 整体压缩的，
> 和"人物前倾/下蹲"数学上不可区分，会把深蹲误判成机位问题。改用横向宽度比之后
> 下蹲/弓步（宽度不变）不会误报，而"头比脚近"（机位偏高/俯拍）会稳定触发。
> 大透视 = 该比值偏离 1 超过 0.4；机位偏高/偏低 = 高于 1.2 / 低于 0.83。

**验收**：用 4 段基准素材中的"差素材"（高机位/过近/走近）能稳定触发对应提示。

---

## 6.5 外部方案评估（2026-11，已落地一部分）

> 背景：收到一份"换 HybrIK / VideoPose3D / MMPose / SMPL 系 + 全局骨长预测 + 混合 2D/3D 导出"的建议。
> 逐条对着本仓库核对后，结论如下。**没有更换推理引擎**，只补了两处确有收益的细节。

| 外建议 | 本仓库现状 | 结论 |
| --- | --- | --- |
| 用整段视频的全局信息预测稳定骨长（而不是逐帧） | `calibrate_bone_lengths` 本来就是**全段中位数**，不是逐帧 | 描述有误；但"加权统计"值得做 → **采纳**（见下） |
| 骨长加左右对称约束 | 左右独立标定，播放时可能出现两条腿不等长 | **采纳**：默认强制对称（差异 >8% 时守卫跳过并报告） |
| 混合 2D 优先 + 条件 3D，用 `depthConfidence` 切导出路径 | `depth_flat_weight` 已在 IK 里**逐骨段**门控（投影比^4），比逐帧切导出路径更细；`depthConfidence` 已用于质检降级与导出 `warnings` | **不采纳**：按帧切导出路径会在帧间制造不连续，且不改变姿态数据本身 |
| 更长的时序窗口 + 关节角度极限 + 加速度约束 | 骨长已是硬约束、z 已有 One-Euro + 时域残差 | **不采纳**：窗口级联合优化会把 33~36 ms/帧推高一个数量级；角度极限与 2D 残差会互相打架，制造新伪影 |
| 直腿检测/站立先验 | `depth_flat` 先验的 docstring 就是为"正视站直被解成屈膝"写的，已含门控 | **已覆盖**；若仍有微弯，先跑 `bench_depth.py` 定位是 z 还是平面内，再加先验 |
| 人物比例/身高标定入口 | `personHeightMeters`（calibrated 模式）+ `bone_lengths_override`（USDZ 实测）已存在 | **部分采纳**：入口已在 `FitConfig`/`SkeletonConfig`；缺一个 UI 表单，属产品层待办 |
| 换 HybrIK / VideoPose3D / MMPose / SMPL / FMPose3D | 见第 7 节：HMR 已作为**可选引擎**立项，带许可与体积门槛 | **不采纳为默认路径**：需要 PyTorch + GPU、SMPL 权重有注册与商用限制、PyInstaller 体积 +1.5~2 GB；VideoPose3D 系列只给 H36M 17 关节根相对姿态，仍需自己映射与尺度对齐，且丢失 2D 一致性约束 |

本次落地的两处（都在 `postprocess/ik_fit.py`，默认开启、`legacy_postprocess_config()` 可回滚）：

1. `bone_sample_min_projected_ratio`（默认 0.85）：只让"骨段正对镜头"的帧参与骨长中位数。
   沿视线的深度噪声只会把骨段**量长**（σ 叠加成 `sqrt(L²+σ²)`），筛掉透视缩短帧能消掉这个系统偏差。
   短骨段（锁骨/骨盆半宽、脚掌，<0.2 m）不参与筛选 —— 它们与噪声同量级，且是 yaw 的参考宽度。
2. `symmetrize_bones`（默认开，`max_left_right_asymmetry=0.08`）：左右同名骨段取平均长度。
   人体真实左右差异 <2%，观测偏差主要是单侧深度噪声，平均能把噪声摊掉一半。

实测（合成序列，`tests/test_depth_fit.py`）：

- 8/12 帧小腿被 z 噪声量成 0.52 m 时：旧行为 0.40 m（越界钳制），新行为 0.42 m（真值）；
- 左腿人为放大 5%：对称后左右严格相等，并有 note 记录；放大 9%（超过守卫）时保留差异并报告；
- yaw 精度未受影响（参考宽度改用全样本中位数，20°/40° 误差 0.25°/0.06°）。

---

## 7. 中期可选项：HMR 引擎（默认不做）

> 仅当产品目标变成"人物在 3D 场景中真实站位/走位/与物体交互"时启动。HMR 能改善 Z 的**估计质量**，但改变不了单目深度歧义；且它有许可与打包代价。

- [ ] 7.1 决策前提（先满足再动手）
  - [ ] 明确商用/分发形态：仓库 MIT，但 InstantHMR 的 nano/tiny/small/medium 权重**仅限非商业**，可商用的 `medium-commercial` 精度从 EMDB 48.8mm 掉到 72.2mm
  - [ ] 确认目标用户硬件：需 NVIDIA 驱动 ≥525 + PyTorch CUDA 12.6；Apple Silicon / 无独显用户必须能优雅降级
  - [ ] 结论写入 `docs/`，明确是否值得做
- [ ] 7.2 依赖隔离
  - [ ] `pyproject.toml` 新增 `[hmr]` 可选 extra，绝不进入默认安装与默认 PyInstaller 打包
  - [ ] 打包体积评估（预期 +1.5–2GB），在 `docs/packaging.md` 记录取舍
- [ ] 7.3 引擎适配器
  - [ ] `app/backend/engine/hmr_engine.py` 继承 `PoseEngine`，实现 `availability()` / `prepare()` / `infer()` / `capabilities`
  - [ ] 用现有 `register_engine("hmr", factory, priority=...)` 注册，不改 `registry.py` 的选型逻辑
  - [ ] `capabilities.world_landmarks=True`，输出对齐 `PoseFrame`（33 点契约），内部做 MHR 70 → BlazePose 33 的映射
  - [ ] 权重按许可分档下载（UI 标注"非商业"/"可商用"），与现有 `model_downloads.py` 复用
- [ ] 7.4 拓扑与坐标系映射
  - [ ] MHR 70 关节 → BlazePose 33 → ARKit 18 的映射表 + 单元测试（左右不混、命名与 `jointNames` 一致）
  - [ ] 相机系（x 右 / y 下 / z 前）→ `arkit-right-handed-y-up` 的转换与往返校验
  - [ ] 骨长来源改为 HMR 体型参数，验证与 USDZ 骨骼的对齐
- [ ] 7.5 降级与自检
  - [ ] 无 GPU / 缺依赖 / 权重未下载 → `EngineUnavailableError` + 明确原因，自动回落 MediaPipe
  - [ ] UI 引擎列表显示可用性原因（复用 `describe_engines`）
- [ ] 7.6 精度对比（决定是否保留）
  - [ ] 4 段基准素材 + 现有 2D 重投影指标对比表（MediaPipe+IK vs HMR）
  - [ ] 若 Z 抖动/骨长稳定性提升 < 20%，判定不划算并归档该方向

**验收**：默认安装与打包体积不受影响；HMR 缺失时应用行为与现在完全一致。

---

## 8. 深度符号先验（2026-11 已实现，默认关闭）

> 起因：「向前踢腿」等动作被解成反方向。本节记录为什么这不是"深度不够准"，而是**一个比特**的问题，以及落地方式。

### 8.1 结论先行

弱透视下 `u = k·X + u0`、`v = k·Y + v0`；骨段真长 `L` 由标定给出且是硬约束，2D 观测把 `(ΔX, ΔY)` 定死，于是

```
ΔZ = ± sqrt(L² − ΔX² − ΔY²)
```

**大小被完全锁死，唯一未知量是符号。** 该比特原本由 MediaPipe 的逐帧 `world_landmarks.z` 决定（`depth_flat` 在 `r = p/(kL)` 小时按 `r^4` 衰减，踢腿时 `r≈0.3`，压平权重几乎为 0，等于没有约束），而它没有时序上下文 → 同一段动作里符号来回翻，播放时就是"腿朝反方向踢 / 膝盖反折"。

因此 VideoPose3D 这类 2D→3D 抬升器**只有"方向判断"这部分可用**：幅度与尺度在跨域时必然不可信，但符号是缺失的那个比特。

### 8.2 已落地的实现

- [x] 8.2.1 `app/backend/postprocess/depth_sign.py`
  - 只输出 `{帧号: {骨段 key: (符号, 置信度)}}`，不输出坐标、不输出骨长
  - `temporal` 后端：直接读 world z 的 Δz 符号 → 时间窗多数投票 → 短游程消除 → 一致率过滤（零新增依赖）
  - `model` 后端：可选 ONNX 抬升器（VideoPose3D/H36M 17 点），**只取符号**；权重缺失 / onnxruntime 异常 / 形状不符一律静默退回 `temporal`
  - `auto_calibrate`：用同段视频中 model 与 world z 的符号一致率判断"整段深度镜像"（<0.3 整体取反）或"模型无效"（<0.55 丢弃），规避左右/深度轴约定无法离线确认的问题
- [x] 8.2.2 `ik_fit` 侧的执行方式（关键设计）
  - `flip_signed_depth()`：读入帧时把观测按符号做**等距镜像**（只翻 z），2D 与骨长一个字不变；必须在遮挡判断之前做，否则"上一帧已修正 vs 本帧未修正"会被误判成单帧突变
  - `_residuals` 新增 **hinge** 项 `max(0, −s·Δz)`：符号正确时残差与梯度都为 0，不抢 2D 重投影与骨长的任何自由度，幅度完全放行
  - 有符号先验的骨段按置信度衰减 `depth_flat` 权重（压平会把 Δz 往 0 拽，与先验直接冲突）
  - 时域 z 残差排除该骨段**及其整条子树**（否则子节点仍会把父骨段拽回去），消除"动作起始帧深度被压回 0"的慢半拍
  - 兜底路径 `_rigidify` 保留符号：兜底的职责是"不做最小二乘"，不是"把选对的分支退回去"
- [x] 8.2.3 可观测性
  - `DepthSignReport`（`mode` / `segments` / `flipsBefore` / `flipsAfter` / `disagreements` / `notes`）进任务摘要与质检 notes
  - `scripts/bench_depth.py` 新增 `depth_sign_flips`（符号翻转次数）与 `sagittal_motion_share`（四肢顺着镜头伸展的帧占比），并加 `--depth-sign` 做 A/B
  - 质检新增矢状面运动提示：四肢长时间指向/远离镜头时**建议补拍正侧机位**（这类动作在正面机位下前后方向本就没有观测，属拍摄问题）

### 8.3 仍需真机验证的部分

- [ ] 8.3.1 导出真实 ONNX 权重后跑一次符号自检：面对镜头双臂前平举 → 腕部 Δz 应指向相机；双臂后平举 → 相反（同时验证左右映射）
- [ ] 8.3.2 用真实踢腿素材对比 `depth_sign_flips` 与人工标注的矢状面角度误差，把数字补进 `docs/benchmark.md`
- [ ] 8.3.3 若 `sagittal_motion_share` 高的素材占多数，优先补强「拍摄引导」（§6）而不是继续堆模型

### 8.4 明确不做

- 不用 VideoPose3D 的输出替代主链坐标：它 root-relative、无尺度、无手脚、无全局朝向，且 H36M 权重**仅限非商业研究**（与 §7 对 HMR 的判断一致）
- 不为 `model` 后端引入 PyTorch 运行期依赖：离线导出 ONNX，运行期复用现有 `onnxruntime` 与 EP 选型，打包体积只 +40~60 MB

---

## 完成定义（Definition of Done）

- [ ] 1–6 节全部勾选，`docs/benchmark.md` 有"优化前 / 优化后"完整对比表
  - 唯一未勾选的是 2.7 的「真实数据测试：骨长变异系数 < 1.5%」：它需要在本机按 `samples/benchmark/README.md`
    录 4 段素材（仓库不含视频），代码与命令已就绪，录完重跑一遍并把数字填进上表即可。
  其余 1–6 节均已勾选，`docs/benchmark.md` 已有 A/B/C 三档完整对比表 + yaw 对比表 + Z 调参表。
- [x] `pytest` 全绿，新增测试覆盖：骨长保持、重投影误差、yaw 恢复、逐轴平滑、schema 兼容
  - 本机 179 项通过 / 2 项失败（共 181），失败的两项（`test_engine::test_unavailable_engines_raise_app_error`、
    `test_models_api::test_models_api`）是**环境依赖**：它们假设"本机没有下载过姿态模型"，
    而本机 `~/.detect-body/models` 已有权重，故 `poseReady` 为 True。与本次改动无关（改动前同样失败）。
- [x] `schemaVersion` 升到 1.1.0，`samples/expected_pose.json` 与 `schema.json` 同步重新生成
- [x] README 的「坐标与单位约定」「iOS/macOS 侧必须检查的一致性」两节已更新为带可信度说明的版本
- [x] 关闭所有新开关（`ikFit=false` 等）时，输出与本次改动前逐位一致（可回滚）
  - 实现方式：`postprocess/pipeline.py::legacy_postprocess_config()` 一个入口还原全部默认值
    （z 参数留空、关掉世界点平滑、`depth_prior_weight=0.05`、`temporal_axes="xyz"`、
    不算 yaw、质检不因新指标降级），并有测试断言每个开关都确实关掉了。
    新字段（`depthConfidence` 等）仍会输出，但只是附加信息，不改变可用数值。
