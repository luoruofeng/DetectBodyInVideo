# 「前踢腿」深度方向与幅度优化任务清单

> 背景：正面机位拍摄的**向前踢腿**动作，导出后被解成"腿朝反方向踢 / 膝盖反折 / 伸展幅度明显不够"。
>
> 根因不是"深度不够准"，而是**正面机位下矢状面不可观测**：弱透视下 `u = k·X + u0`、`v = k·Y + v0`，骨段真长 `L` 由标定给出且是硬约束，2D 观测把 `(ΔX, ΔY)` 定死，于是
>
> ```
> ΔZ = ± sqrt(L² − ΔX² − ΔY²)
> ```
>
> **大小被完全锁死，唯一未知量是一个比特（符号）。** 该比特原本由 MediaPipe 逐帧 `world_landmarks.z` 决定，没有时序上下文，同一段动作里来回翻 → 播放时就是"腿翻面"。而**幅度**（踢得多高、伸得多直）则由 `depth_flat` 压平先验和 2D 噪声决定，同样没有证据支撑。
>
> 因此本清单的目标分两档：
>
> - **必达**：单段正面视频不再出现"腿翻面"，踢腿方向与人工观察一致；
> - **力争**：在正面机位下给出**可信程度明确的**伸展幅度，并在幅度不可信时如实标注，而不是伪造。

**进度：0 / 31 个主任务**（§6.1 / §8.1 中标注为 `[x]` 的子项已在现有代码中落地，见附录 A，只需确认；实际待做 30 项主任务 + 2 项确认）

> 使用方式：
>
> - 把 `- [ ]` 改成 `- [x]` 表示完成。子任务缩进两格表示属于上一级任务。
> - **章节编号固定不变**（历史讨论、日志、提交信息都在引用它），实际执行顺序看 §0.2；每节开头的「执行位置 / 依赖」说明了先后，前置未完成不要跳到后续。
> - 动手前务必先读 **附录 A：代码现状核定（2026-11-21）**。它把本文档与代码逐条对照过一遍，并列出 5 处已改正的偏差；本文档与代码冲突时以附录 A 为准，并回填本文档。

---

## 0. 一页结论

### 0.1 收益分三层，只有后两层碰"幅度"

| 层次 | 解决的问题 | 对应章节 | 外部依赖 |
| --- | --- | --- | --- |
| 符号层（必达） | 腿翻面、膝盖反折 | §2（§5.1 的 2D 精度可辅助） | `temporal` 无依赖；`model` 需 H36M 权重 |
| 幅度层 | 踢得多高、伸得多直 | §4（软方向先验）、§5.1（2D 精度） | §3 权重 + §1 人工标注 |
| 物理根治 | 幅度可信 | §6 | 补拍正侧视频 |

> 关键判断：**"幅度"不是自由量**。骨长硬约束下 `|Δz| = sqrt(L² − ΔX² − ΔY²)` 已被 2D 定死，
> 想改变幅度只有两条路——① 提高 2D 精度（§5.1，不牺牲 2D 一致性）；② 让关节偏离观测像素（§4，用重投影换）。
> **不要指望"换一个更强的 3D 模型"就免费拿到幅度。**

### 0.2 推荐实施顺序（按投入产出，不是按编号）

| 波次 | 章节 | 内容 | 为什么排这里 | 预计投入 |
| --- | --- | --- | --- | --- |
| 第 1 波 | §2.1、§2.4 | 打开已写好的符号链路；前端可选后端 + 缺权重置灰 | 零新依赖、可完全回滚，直接消"腿翻面" | 1–2 小时 |
| 第 2 波 | §8.2、§8.3 | UI 显示实际后端 / 降级提示 / 自校准比例 | 让第 1 波的效果可被看见，避免把估计值当真值 | 半天 |
| 第 3 波 | §5.1 | 2D 精度：阈值暴露 + 下肢 crop 二次推理 | 唯一"不牺牲 2D 一致性就能改善幅度"的路径，无许可风险 | 1–2 天 |
| 第 4 波 | §1 → §3 → §4 | 建基线 → 换 lifter → 方向软先验 | 需录素材 + 逐帧标注 + 非商业权重；§4 的收益是拿重投影换的 | 1–2 周 |
| 可选 | §6.2、§7 | 双机位 / 米制引擎 | 产品决策，不属于调优 | — |

> 如果第 4 波不做（不录素材、不做人工标注），§3 与 §4 的"幅度改善 −30%"**永远无法证伪**，
> 建议直接归档这两节，把资源留给 §5.1 与 §6.1。

---

## 约束与既定决策（动手前先读）

- **不更换主推理引擎**：MediaPipe PoseLandmarker 仍是默认路径，ONNX/RTMPose 保持既有位置。
- **不承诺绝对深度**：抬升模型的米制幅度在跨域时必然不可信；本清单只让它提供**方向**与**尺度无关的比例**。
- **不引入运行期 PyTorch**：第三方模型一律离线导出 ONNX，运行期复用现有 `onnxruntime`。EP 选型的**唯一真源**是 `engine/acceleration.py`；当前 `depth_sign.py::_model_signals()` 里另有一份写死的 provider 候选列表，属待清理项（见 §3.1），新代码不要再复制第三份。
- **默认行为不变**：所有新开关默认关闭，关闭时输出与当前版本逐位一致（可回滚）；`legacy_postprocess_config()` 必须能还原。
- **任何新后端失败都必须静默降级**：权重缺失 / 推理异常 / 形状不符 / 自校准判定无效 → 退回 `temporal`，绝不中断分析。
- **权重许可**：Human3.6M 训练的权重普遍**仅限非商业研究**。产物不提交仓库、不随商用发行版分发；商用前需在 §7 做许可决策。
- **不重复实现已落地项**：附录 A 中标为「已落地」的条目（符号先验全链路、自校准、`ik_fit` 执行层、质检矢状面提示、README 拍摄建议与已知限制）直接在已有实现上确认，不要另写一份。
- **§4 的收益有定价**：方向先验改变幅度必须牺牲重投影一致性（见 §4 的「数学前提」）。验收按"重投影预算"口径，不按"免费改善"口径。
- **素材编号统一**：踢腿素材直接并入 `samples/benchmark/README.md` 的 BM 系列（`BM-05`~`BM-07`），不再新建 `KICK-01`~`KICK-03` 编号——两套编号会让基准分裂。
- 既有实现与本文档的关系：符号先验已落地于 `app/backend/postprocess/depth_sign.py`（详见 `docs/todo-depth-z-accuracy.md` §8），本清单**在其上扩展**，不重写。

---

## 1. 基线与验收口径（先量化，再改）

> **执行位置**：第 4 波的起点。**依赖**：无，但只在你打算做 §3 / §4 时才值得投——不打算做就没有验收对象。
>
> 没有可重复的数字，后面所有"感觉变好了"都无法判断真假。

- [ ] 1.1 建立踢腿基准素材（并入现有素材清单，不新建编号体系）
  - [ ] 录制 3 段 ≤10s 素材：`BM-05 正面全身前踢`、`BM-06 正侧全身前踢`、`BM-07 正面斜 30° 前踢`
  - [ ] 直接**追加**到 `samples/benchmark/README.md` 现有 `BM-01`~`BM-04` 表末尾（文件名示例 `BM-05-front-kick.mp4`），不要另建 `KICK-01`~`KICK-03`
  - [ ] 视频本体不进仓库（`.gitignore` 已忽略 `samples/*.mp4|mov|m4v`），只提交说明与文件名
  - [ ] 在 `docs/benchmark.md` 记录：编号、机位角、距离、分辨率、帧率、动作次数
- [ ] 1.2 人工标注真值（后续所有幅度对比的唯一依据）
  - [ ] `BM-06 正侧`逐帧标注摆动腿的**大腿抬起角**与**小腿伸展角**（可用半自动：先跑一遍侧视结果再人工修正）
  - [ ] 标注**踢腿相位**：起始帧 / 最大伸展帧 / 回落帧，供 §5.3 的速度剖面先验使用
  - [ ] 真值文件存 `samples/benchmark/kick-labels.json`，字段与 `bench_depth.py` 输出对齐，并在 `samples/benchmark/README.md` 注明该文件
- [ ] 1.3 扩展评测脚本
  - [ ] **前置（当前不具备，必须先补）**：`scripts/bench_depth.py::build_postprocess_config()` 现在只用 `--depth-sign` 把 `enabled` 置 True，**没法选 `model` 档** → 先加 `--depth-sign-mode {temporal,model}`，否则 §1.4 的三档基线采不全
  - [ ] 新增 `kick_sagittal_error_deg`：与人工标注比对大腿/小腿矢状面角，输出 mean / p95
  - [ ] 新增 `sign_accuracy`：符号与真值一致的帧占比（区分大腿 / 小腿 / 摆动腿 / 支撑腿）
  - [ ] 复用现有 `depth_sign_flips`、`sagittal_motion_share`、`reproj_error_px`，不重复造指标
  - [ ] 支持 `--labels samples/benchmark/kick-labels.json` 与 `--baseline` 对比
- [ ] 1.4 记录改动前基线
  - [ ] 三条命令各跑一次并存 JSON：`深度方向修正=关`、`=temporal`、`=model`（第三条需 §2.2 的权重）
  - [ ] 数字写进 `docs/benchmark.md` 的「踢腿专项」小节（该小节**当前不存在**，需新建）

**验收**：`python scripts/bench_depth.py samples/benchmark/BM-05-front-kick.mp4 --depth-sign --depth-sign-mode temporal --labels samples/benchmark/kick-labels.json --json out/bm05.json` 一次性输出符号准确率与矢状面角误差，且可与基线 diff。

---

## 2. P0：把已有的符号先验链路真正跑起来（零新依赖）

> **执行位置：第 1 波，第一件要做的事。** **依赖**：无。**收益**：必达档（消"腿翻面"），不改善幅度。
>
> 现状（已核实，见附录 A）：`depth_sign` 全链路已实现且默认关闭；前端硬编码 `mode: 'temporal'`；`models/` 里没有 `videopose3d-h36m.onnx`，所以 `model` 后端从未真正生效。

- [ ] 2.1 打开开关并确认链路生效（今天就能做）
  - [ ] 界面勾选「深度方向修正」，或在 `.env` 设 `DETECT_BODY_DEPTH_SIGN_ENABLED=1`
  - [ ] 确认后处理阶段出现 `lift → 修正前后方向` 进度（`postprocess/pipeline.py`）
  - [ ] 确认任务摘要 `depthSign.mode` 不再是 `off`，`flipsBefore → flipsAfter` 有下降
  - [ ] 确认质检 notes 里能看到一致率与丢弃原因（`quality.py` 已把 `DepthSignReport.notes` 汇入）
- [ ] 2.2 导出并校验 VideoPose3D ONNX 权重
  - [ ] `python scripts/export_videopose3d_onnx.py --repo ../VideoPose3D --checkpoint ... --output models/videopose3d-h36m.onnx`（脚本已存在，`--verify` 不依赖 torch）
  - [ ] `python scripts/export_videopose3d_onnx.py --verify models/videopose3d-h36m.onnx` 通过
  - [ ] 记录产物体积与 EP（预期 +40~60 MB）。注意推理端 EP 候选目前**写死在** `depth_sign.py::_model_signals()` 里（`CUDA`/`CoreML`/`Dml`/`CPU`），并没有复用 `engine/acceleration.py`
- [ ] 2.3 跑 `model` 后端并读自校准结论
  - [ ] `DETECT_BODY_DEPTH_SIGN_MODE=model` 后重跑 `BM-05`
  - [ ] 读 `DepthSignReport.notes`：一致率 >55% = 自校准通过；0.3~0.55 = 判定无效（说明该权重在此素材上无用，停止调参）；<0.3 = 判定深度轴镜像并已整体取反（阈值已在 `_calibrate_model_signs()` 实现，无需重新设计）
  - [ ] 若判定无效，把结论写进 `docs/benchmark.md`，**不要**继续在这条路上堆参数，并直接归档 §3
- [ ] 2.4 前端不再硬编码后端
  - [ ] `app/frontend/src/components/AnalysisPanel.tsx` 的 `mode: 'temporal'` 改为可选项（`temporal` / `model`），并在权重缺失时置灰
  - [ ] `app/frontend/src/api/analysis.ts::DepthSignOptions` 补 `model_path`、`auto_calibrate`、`swap_left_right`（后端 `DepthSignConfig` 已有这三个字段，纯 DTO 补齐）
  - [ ] 界面显示**当前实际使用的**后端：必须读后端回传的 `depthSign.mode`，**不要在前端推断**——`app/backend/pipeline.py` 中有合并规则：前端传的 `mode` 若等于默认值 `temporal`，会被 `settings.depth_sign_mode` 覆盖，前端自己算会显示错
  - [ ] 降级时显示原因（如"权重缺失，已退回 temporal"），文案直接取 `depthSign.notes`，不要在前端另写一套判断

**验收**：`model` 后端可被界面选择并在权重就绪时生效；`depth_sign_flips` 相比基线下降 ≥50%；权重缺失时自动退回 `temporal` 且界面明确提示。

---

## 3. P1：抬升器后端抽象 + 换 MotionBERT（推荐主力，但先过闸门）

> **执行位置**：第 4 波，§1 之后。**依赖**：§2.3 的 `model` 后端自校准通过 + §1 的基线数字。
>
> VideoPose3D 是 2019 方案，H36M + Detectron 2D 域，正对镜头的素材存在系统性偏差。MotionBERT 用 H36M + MPI-INF-3DHP + COCO 联合训练，视角分布更广，且输入契约同为 `[T,17,2] → [T,17,3]`，可几乎无缝替换。

- [ ] 3.0 立项闸门（先判断，别先动手）
  - [ ] 记录 §2.3 的一致率；若落在 0.3~0.55（判定无效），**归档本节**，不要换 MotionBERT 重试——那是同一个信息缺失问题
  - [ ] 确认是否接受非商业权重：VideoPose3D 用 H36M，MotionBERT 用 H36M + MPI-INF-3DHP，均限非商业研究
- [ ] 3.1 抽象抬升器后端（不改算法语义）
  - [ ] `depth_sign.py` 新增 `LifterBackend` 协议：`availability()` / `load()` / `infer(seq) -> np.ndarray[T,17,3]` / `name`
  - [ ] `resolve_model_path()` 改为按 `DETECT_BODY_LIFTER`（`videopose3d|motionbert|agformer`）返回候选路径（**该环境变量目前不存在**，需在 `settings.py` 新增），保留现有 `models_dir` → 打包资源根的查找顺序
  - [ ] `_model_signals()` 只保留"跑推理 + 抽符号 + 记 note"，推理细节下沉到各 backend
  - [ ] EP 选型只保留**一处**：现在 `_model_signals()` 自己写死了 provider 候选列表，重构时要么上移到 `engine/acceleration.py` 暴露公共函数，要么留在 lifter backend 内。**不要两边各写一份**（本节原稿"复用 acceleration 的 EP 选型"与现状不符）
  - [ ] `_h36m_input()`（1.25× 人体框归一化）保持不变，作为所有 lifter 的公共输入
  - [ ] `_calibrate_model_signs()`、`_vote()`、`_despike()` 完全复用，不重复实现
- [ ] 3.2 接入 MotionBERT
  - [ ] 新增 `scripts/export_motionbert_onnx.py`（参照 `export_videopose3d_onnx.py` 的风格：`--verify` 不依赖 torch）
  - [ ] 处理注意力分支的动态 `T`（`dynamic_axes` + 固定 `max_T` 两套方案都验证一遍）
  - [ ] 输入归一化与官方 `read_input` 对齐：bbox 外扩 1.25、`normalize_screen_coordinates` 到 `[-1,1]`
  - [ ] 感受野填充策略参数化（VideoPose3D 243 帧、MotionBERT 视窗口而定），不足时边缘复制
  - [ ] 权重缺失 / 形状不符 / EP 异常 → 写 note 并退回 `temporal`
- [ ] 3.3 与 VideoPose3D 做 A/B
  - [ ] 同一素材跑 `videopose3d` / `motionbert` 两档，输出 `sign_accuracy`、`kick_sagittal_error_deg`、耗时
  - [ ] 结论表写进 `docs/benchmark.md`；若 MotionBERT 提升 <10%，判定不划算并归档（避免无谓体积）
- [ ] 3.4 测试
  - [ ] 合成序列单测：给定已知 3D 骨架 → 投影成 2D → 走 backend → 符号还原率 100%
  - [ ] 降级路径单测：模型缺失 / 输出含 NaN / 形状错误三种情况都不抛异常且 `report.mode == "temporal"`
  - [ ] 左右镜像单测：人为镜像模型输出 → `auto_calibrate` 判出并整体取反

**验收**：`DETECT_BODY_LIFTER=motionbert` 可切换生效；三条降级路径都有测试覆盖；`pytest tests/test_depth_fit.py -q` 全绿。

---

## 4. P2：从"只有符号"升级到"方向向量软先验"（用重投影换幅度）

> **执行位置**：第 4 波末尾，§3 之后。**依赖**：lifter 能给出方向（否则 `n_target` 无处可取）+ §1 人工标注（否则无法验收）。
>
> **数学前提（决定本节能否达标，先读）**：`hard_bones=True`（默认）时骨长 `L` 是硬约束，
> 2D 观测定死 `(ΔX, ΔY)`，于是 `|Δz| = sqrt(L² − ΔX² − ΔY²)` **不是自由量**。
> 方向先验想改变幅度只有一条通道：让关节**偏离观测像素**，用重投影误差换深度方向。
> 所以本节验收本质是「**每帧 1 px 重投影预算换矢状面角误差**」的交易，不是免费收益；
> 若目标是"不动 2D 一致性就改善幅度"，本节做不到 → 去 §5.1。
>
> 当前 hinge 只惩罚 `max(0, −s·Δz)`，符号对了就完全放行，幅度仍由压平先验和 2D 噪声决定。改用抬升器输出的**单位骨段方向** `n = (Δx, Δy, Δz) / L`：尺度无关、跨域比米制坐标鲁棒一个数量级，且天然同时编码符号与幅度。

- [ ] 4.1 数据结构扩展（注意向后兼容）
  - [ ] `SignMap` 增加可选的方向通道：`{帧: {骨段: (符号, 置信度, n_target|None)}}`；`n_target is None` 时行为与现在逐位一致
  - [ ] **兼容约束**：`tests/test_depth_sign.py` 里有用例直接构造 `{帧: {骨段: (符号, 置信度)}}` 二元组，`_sign_context()` 也是"四数组"契约。扩展要么按"可选第三元素"做解包兼容，要么一次性改全部调用点与用例，**不要两种形态混跑**
  - [ ] 新增 `DirectionPriorConfig`（或 `DepthSignConfig` 增字段）：`direction_weight`、`direction_min_agreement`、`direction_floor_deg`（方向偏差小于该角度不惩罚）
- [ ] 4.2 ik_fit 侧执行
  - [ ] `_sign_context()`（`app/backend/postprocess/ik_fit.py`）扩展为同时产出方向数组，保持"按节点下标对齐"的现有约定
  - [ ] `_residuals()` 在 hinge 项旁新增软残差：只惩罚 `n_pred` 相对 `n_target` 的**垂直分量**，允许沿骨段方向的自由度
  - [ ] 与 `depth_flat` 的冲突处理沿用现有 `depth_sign_flat_suppression` 逻辑，按置信度让位
  - [ ] `flip_signed_depth()`、`_rigidify()`、时域 z 残差的子树排除规则（`_make_context()` 里的 `temporal_rows`）同步支持方向通道；`_build_sparsity()` 的稀疏结构也要跟着加块，否则雅可比会退化成稠密
  - [ ] 方向先验被丢弃（一致率不足）时退回纯 hinge，保证可回滚
- [ ] 4.3 权重与量纲
  - [ ] 权重按 `sqrt(direction_weight) * k` 归一，与重投影残差同量纲（沿用 `_flat_scale` / `_sign_context` 的既有做法）
  - [ ] 扫 3 档权重（弱/默认/强），记录 `reproj_error_px` 与 `kick_sagittal_error_deg` 的**权衡曲线**（不是单纯求最小）
  - [ ] 选定默认值并写注释说明来源，不接受"凭空调参"
- [ ] 4.4 测试
  - [ ] 合成测试：真值骨架 + 已知踢腿幅度 → 打开方向先验后矢状面角误差下降 ≥30%，重投影恶化不超过预算（1 px，见验收口径）
  - [ ] 幂等性：对已拟合结果再拟合，位移变化 <5 cm（与现有 `test_fit_is_stable_when_reapplied` 判据一致）
  - [ ] 回滚测试：`direction_weight=0` 时输出与改动前逐位一致

**验收**：正面素材上踢腿伸展幅度有明显改善，且 `depthConfidence` / `warnings` 如实反映可信度；关闭开关时行为完全回滚。**重投影恶化 ≤1 px 是本节主动支付的成本，不是副作用**；若某素材必须超过 1 px 才能换到 −30%，说明该素材不适合本节，应转 §5.1 或 §6。

---

## 5. P3：2D 质量与摆动腿时序先验（投入产出最高的一档）

> **执行位置：第 3 波**，可在 §2 之后、§3/§4 之前做。**依赖**：无（不需要任何抬升模型或权重许可）。
>
> 正面踢腿时小腿几乎正对镜头，投影长度极小，脚踝 1 px 误差 ≈ 十几厘米深度误差。上游 2D 差 1 个像素，下游任何模型都救不回来。

- [ ] 5.1 提高下肢 2D 精度（本节是全清单性价比最高的一条）
  - [ ] **不要**把默认档位改成 `heavy`：现有设计是 `prefer_model_tier="auto"`，由 `engine/acceleration.py` 按显存 / 统一内存自动选档，"默认 heavy"会打掉自动降级能力，低配机器直接跑不动
  - [ ] 改为**条件策略**：下肢可见度低于阈值（或 `sagittal_motion_share` 超阈值）时，对该段视频用 `heavy` 重跑并择优——只有真正需要的素材才付 heavy 的代价
  - [ ] `min_detection_confidence` / `min_tracking_confidence` 暴露到界面（后端 `settings` 与 `mediapipe_engine` 已支持，缺的只是 `AnalysisOptions` 透传），踢腿场景建议 0.35
  - [ ] 对下肢做 hierarchical crop 二次推理（复用 `engine/onnx_engine.py` 的 crop 逻辑，MediaPipe 侧新增轻量实现）
  - [ ] 对比实验：crop 前/后 `sign_accuracy` 与 `kick_sagittal_error_deg`（需 §1.3 的指标）
  - [ ] 记录 2D 精度提升后 `|Δz|` 的抖动是否下降——这是"不牺牲重投影就改善幅度"的直接证据
- [ ] 5.2 摆动腿识别
  - [ ] 用髋–膝–踝的 2D 运动速度能量区分摆动腿与支撑腿，输出 `swing_leg` 到任务摘要
  - [ ] 只对摆动腿施加更强方向先验，支撑腿保持现有压平先验（避免把站立腿掰弯）
- [ ] 5.3 踢腿相位与速度剖面先验
  - [ ] 从 2D 踝部轨迹检测相位：`起脚 → 加速 → 最大伸展 → 回落`
  - [ ] 在"最大伸展"帧附近放宽方向先验权重、在静止段收紧（利用动作本身的可预测性）
  - [ ] 相位检测失败时不影响主链，只记 note
- [ ] 5.4 测试
  - [ ] 合成梯形/正弦速度剖面 → 相位检测误差 <3 帧
  - [ ] 摆动腿误判（双腿交替摆动）时退回对称处理，不制造单侧伪影

**验收**：条件 `heavy` + crop 后 `sign_accuracy` 相对 P0 再提升；摆动腿标注正确率 ≥90%；相位检测不引入新抖动；且 `reproj_error_px` 不劣化（本节不看"用重投影换幅度"，只看纯增益）。

---

## 6. P4：机位与多视角（物理上唯一根治）

> **执行位置**：§6.1 随时可做（几乎零成本，且大部分已完成）；§6.2 属产品决策。
>
> 正面单目下矢状面没有观测，任何模型都只能"猜得合理"。真正可靠的做法是让矢状面进入图像平面。

- [ ] 6.1 质检与拍摄引导补强
  - [x] `postprocess/quality.py::build_quality_report()` 已复用 `sagittal_motion_share`，四肢长时间指向镜头时给出「建议补拍一个正侧机位」issue（开关 `enable_sagittal_hint`）
  - [ ] 唯一缺口：「拍摄建议」机位表补一行「**前踢腿 → 正侧优先**」。`README.md`「按动作类型选机位」表已有下蹲/弓步/前弓步/硬拉、转体、走路/跑步、混合，**缺踢腿**
  - [ ] 导出 `warnings` 在矢状面不可观测时写明原因（现有 issue 已做到，确认导出侧同步即可，避免下游把估计值当真值）
- [ ] 6.2 双机位融合（可选，产品化时再做）
  - [ ] 定义输入契约：同一动作的 `正面 + 正侧` 两条视频，时间轴需人工对齐或按音频/动作起止对齐
  - [ ] 侧面结果提供矢状面角度，正面结果提供左右对称与水平方向角度，按可见度加权融合
  - [ ] 融合只在 2D 层或关节角层做，不引入新的 3D 优化器，避免翻倍耗时
  - [ ] 无第二机位时完全退化为现有单机位路径
- [ ] 6.3 验收
  - [ ] 双机位输出与 `BM-06` 人工标注对比，矢状面角误差 <8°

**验收**：单机位正面素材会在质检与 `warnings` 里明确告知幅度不可信；双机位路径（若实现）显著优于任何单目方案。

---

## 7. P5：可选米制方案与许可决策（默认不做）

> **执行位置**：不排期。仅当产品目标变成"要真实站位/真实踢出多少厘米"时启动。
> 与 `docs/todo-depth-z-accuracy.md` §7 是**同一立项**，不要在两份文档里各评估一遍。

- [ ] 7.1 决策前提
  - [ ] 明确分发形态：仓库 MIT，但 H36M 权重与 SMPL-X 系权重均为**非商业/注册制**
  - [ ] 评估 MeTRAbs（单图直接出米制相机系 3D、截断鲁棒）与 Multi-HMR / CameraHMR 的精度与体积
  - [ ] 结论写入 `docs/`，明确是否值得做
- [ ] 7.2 依赖隔离
  - [ ] `pyproject.toml` 新增可选 extra（如 `[metric]`），**绝不**进入默认安装与默认 PyInstaller 打包
  - [ ] 打包体积评估（预期 +1.5~2 GB）写入 `docs/packaging.md`
- [ ] 7.3 引擎适配（若立项）
  - [ ] 新增 `app/backend/engine/metric_engine.py` 实现 `PoseEngine`，`capabilities.world_landmarks=True`
  - [ ] 输出对齐 `PoseFrame` 33 点契约，内部做模型关节 → BlazePose 33 的映射
  - [ ] 无 GPU / 缺依赖 / 权重未下载 → `EngineUnavailableError` + 原因，自动回落 MediaPipe
- [ ] 7.4 精度对比
  - [ ] 在 `BM-05`~`BM-07` 上对比 MediaPipe+IK vs 米制引擎的 `kick_sagittal_error_deg`
  - [ ] 提升 <20% 判定不划算并归档

**验收**：默认安装与打包体积不受影响；缺依赖时行为与现在完全一致。

---

## 8. 文档、UI 与可观测性

> **执行位置：第 2 波**（8.2 / 8.3 部分紧跟 §2.1 之后）。**依赖**：§2.1 先跑通，否则 UI 上没有东西可显示。

- [ ] 8.1 文档
  - [x] `README.md`「拍摄建议」与「已知限制」：已写明机位建议、单目"每骨段只剩一个比特"的限制，以及"要判朝前朝后必须用 `model` 后端并自备权重"
  - [ ] 剩余 1：机位表补「前踢腿 → 正侧优先」一行（与 §6.1 同一件事，只做一次）
  - [ ] 剩余 2：`docs/todo-depth-z-accuracy.md` §8 增加指向本清单的链接（当前无），避免两份文档口径漂移
  - [ ] `docs/benchmark.md` 增加「踢腿专项」小节（基线 / P0 / P1 / P2 / P3 对比表）
- [ ] 8.2 UI
  - [ ] 「数据质检」卡片显示当前深度后端（`off` / `temporal` / `model`）、符号翻转数与自校准结论；**后端名一律取 `depthSign.mode`**（见 §2.4 的覆盖规则）
  - [ ] 摆动腿、相位、方向先验生效帧数进任务摘要（只显示，不参与判定）
  - [ ] 幅度不可信时在 3D 预览旁给出明确提示，避免用户误以为是模型 bug
- [ ] 8.3 可观测性
  - [ ] `DepthSignReport` 新增 `lifter` / `direction_segments` / `swing_leg` / `auto_calibrate_ratio`（自校准一致率目前只存在于 `notes` 文本里，无法被程序消费）
  - [ ] 日志在 `DEBUG` 级别打印每骨段的一致率与丢弃原因

**验收**：任何一个消费方只看 UI 与文档就能判断"这条踢腿数据的幅度能不能用"。

---

## 完成定义（Definition of Done）

- [ ] §2 全部勾选（这是必达档，且零新依赖，可以先只做到这里上线）
- [ ] §1–§2 全部勾选时，`docs/benchmark.md` 有「改动前 / P0」完整对比表
- [ ] §3 至少一个非 VideoPose3D 的 lifter 可切换生效，三条降级路径有测试覆盖；若 §3.0 闸门判定无效，则改为"归档记录已写入 `docs/benchmark.md`"
- [ ] §4 方向先验默认生效时 `kick_sagittal_error_deg` 相对 P0 下降 ≥30%，且重投影恶化不超过 1 px 预算
- [ ] §5.1 完成后 `sign_accuracy` 与 `kick_sagittal_error_deg` 有提升，且 `reproj_error_px` 不劣化
- [ ] 所有新开关关闭时，输出与当前版本逐位一致（`legacy_postprocess_config()` 可还原）
- [ ] `pytest` 全绿；新增测试覆盖：backend 降级、方向先验、摆动腿识别、相位检测
- [ ] `scripts/bench_depth.py` 能一次性输出符号准确率与矢状面角误差，支持 `--depth-sign-mode`、`--labels`、`--baseline`
- [ ] 正面单机位素材在导出 `warnings` 中如实标注"矢状面幅度不可信"；正侧/双机位素材给出可用幅度
- [ ] 权重许可结论已写入 `docs/`，商用发行版不含 H36M / SMPL-X 权重

> 收缩条款：若决定不做 §3 / §4（不录素材、不做逐帧标注），DoD 收缩为 §2 + §5.1 + §6.1 + §8.2/8.3，
> 并把 §1 / §3 / §4 标记为「已归档：缺少可验证的真值与权重」。

---

## 附录 A：代码现状核定（2026-11-21）

### A.1 已落地（勿重复实现）

| 能力 | 位置 |
| --- | --- |
| 符号先验双后端（`temporal` / `model`）+ 默认关闭 | `app/backend/postprocess/depth_sign.py::DepthSignConfig(enabled=False)`、`core/settings.py::depth_sign_enabled=False` |
| 时序投票 / 短游程消除 / 一致率过滤 | `depth_sign.py::_vote()`、`_despike()` |
| H36M 17 点输入归一化（1.25× 人体框） | `depth_sign.py::_h36m_input()` |
| 自校准（<0.3 取反 / 0.3~0.55 丢弃 / ≥0.55 通过） | `depth_sign.py::_calibrate_model_signs()` |
| 推理失败静默退回 `temporal` | `depth_sign.py::_model_signals()`（权重缺失 / ort 缺失 / 形状不符） |
| 后处理阶段接入 | `postprocess/pipeline.py` 中 `emit("lift", "修正前后方向")` |
| 摘要输出 | `app/backend/pipeline.py::summarize()` → `depthSign.{mode,segments,flipsBefore,flipsAfter,disagreements,notes}` |
| `ik_fit` 执行层：等距镜像、hinge、压平抑制、子树排除、兜底保留符号 | `ik_fit.py::flip_signed_depth()`、`_residuals()`、`_make_context()` 的 `depth_sign_flat_suppression` / `temporal_rows`、`_rigidify()` |
| 质检矢状面提示（建议补拍正侧） | `postprocess/quality.py::build_quality_report()` 的 `enable_sagittal_hint` 分支 |
| README 拍摄建议 / 机位表 / 已知限制 | `README.md`「拍摄建议」「按动作类型选机位」「已知限制」 |
| ONNX 导出与校验脚本 | `scripts/export_videopose3d_onnx.py`（`--verify` 不依赖 torch） |
| 基准指标（翻转次数 / 矢状面占比 / 重投影 / `--baseline`） | `scripts/bench_depth.py` |

### A.2 缺口（本文档真正要做的事）

| 缺口 | 证据 |
| --- | --- |
| 前端硬编码后端 | `AnalysisPanel.tsx` 传 `mode: 'temporal'` |
| 前端 DTO 缺字段 | `analysis.ts::DepthSignOptions` 只有 `enabled/mode/window/min_agreement` |
| 权重缺失 | `models/` 只有 MediaPipe 的 `.task`，无 `videopose3d-h36m.onnx` |
| 基准脚本无法选 `model` 档 | `bench_depth.py::build_postprocess_config()` 只设 `enabled=True` |
| 无 `sign_accuracy` / `kick_sagittal_error_deg` / `--labels` | `bench_depth.py` |
| 无踢腿素材与人工标注 | `samples/` 无视频；`samples/benchmark/README.md` 只有 `BM-01`~`BM-04`（站立/深蹲/走近） |
| 无 lifter 抽象 | 全库无 `DETECT_BODY_LIFTER`；`resolve_model_path()` 单一路径；`_model_signals()` 内联 provider 列表，未复用 `engine/acceleration.py` |
| 无摆动腿 / 相位 | 全库无 `swing` 相关实现 |
| `docs/benchmark.md` 无「踢腿专项」 | 全文无 `kick` / 踢腿 |
| 机位表缺「前踢腿」行 | `README.md` 表中只有下蹲/弓步/前弓步/硬拉、转体/挥臂、走路/跑步、混合 |

### A.3 本次改正的 5 处偏差

1. **§5.1**：原稿"默认档位改用 `heavy`"与 `engine/acceleration.py` 的 auto 自动选档冲突 → 改为"条件触发 heavy 重跑"。
2. **§3.1**：原稿"复用 `engine/acceleration.py` 的 EP 选型"与现状不符（`_model_signals()` 自己写死 provider 列表）→ 改为"二选一、只保留一处"。
3. **§6.1**：质检 issue 已落地，原稿仍列为待做 → 标为已完成，只剩机位表补一行。
4. **§1.3 / §1.4**：`bench_depth.py` 无法选 `model` 档，原稿的三档基线不可执行 → 增加 `--depth-sign-mode` 前置项。
5. **§4**：原稿"真正改善幅度"未写代价 → 补充数学前提与"1 px 重投影预算"的验收口径，标题改为"用重投影换幅度"。

另：`§1.1` 的 `KICK-01`~`KICK-03` 与 `samples/benchmark/README.md` 的 `BM-01`~`BM-04` 是两套并行编号，已统一为 **BM 系列**（踢腿为 `BM-05`~`BM-07`）。
