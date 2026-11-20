# 性能基准

本文件记录"加速是否真的生效"的证据，由 `scripts/bench.py` 生成（不要手改 BENCH 标记之间的内容）。

## 测量方法

- 只测**推理链路**：解码（生产者线程）→ 有界队列 → 批处理推理（消费者线程），
  不含后处理与骨骼映射，便于横向比较 EP / 档位 / batch。
- 帧率 = 新处理帧数 ÷ 总耗时（含解码与队列等待），单位 fps。
- 内存峰值 = `tracemalloc` 峰值（Python 侧；GPU 显存请用 `nvidia-smi` 另看）。
- 同一视频、同一 `stride`、同一 `batch` 才可横向比较。

## 复现命令

```powershell
# 真实视频 + 自动选型（先按 --batch 1,4,8 扫一遍，取吞吐最高且内存不炸的 batch）
python scripts/bench.py samples/dance.mp4 --engine auto --batch 1,4,8 --tiers lite,full

# 指定引擎 / 只测 ONNX
python scripts/bench.py samples/dance.mp4 --engine onnx --batch 4 --tiers full

# 无模型、无素材时自检工具与生产/消费链路（stub 引擎）
python scripts/bench.py --engine stub --synthetic --batch 1,4 --stride 2
```

> `stub` 引擎只做固定矩阵运算，**不代表任何真实模型性能**；它的作用是证明
> 基准工具、有界队列与批处理路径本身可用（CI 可跑）。

## 判读标准

- `batch=4` 相比 `batch=1` 应有可见提升（GPU 上通常 20%~60%），否则检查 EP 是否真的启用；
- CPU 上 batch 提升通常很小，甚至因为内存带宽下降，属于正常；
- 内存峰值随 batch 线性上升，若超过可用内存的 60% 应回退；
- 同一行 fps 波动 > 15% 说明机器被打扰（杀毒、浏览器、其它推理进程），重跑。

## 结果

<!-- BENCH:START -->
### 运行记录：harness baseline

- 机器：`Windows / AMD64 / Python 3.13.3`
- ONNX EP：`AzureExecutionProvider, CPUExecutionProvider`

| 引擎 | 档位 | EP | batch | stride | 帧数 | fps | 内存峰值 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stub | stub | stub | 1 | 2 | 30 | 1763.31 | 6.6 MiB | ok |
| stub | stub | stub | 4 | 2 | 30 | 2374.47 | 1.6 MiB | ok |

> 说明：stub 行只验证基准工具与生产/消费链路，不代表真实模型性能；真实结论请用目标机器 + 目标模型重跑。
<!-- BENCH:END -->

---

## 深度精度基准（Z 轴）

`scripts/bench.py` 只测"推理链路吞吐"，这一节测的是**输出姿态的几何质量**：
本项目的输入是普通单目 RGB，`x/y` 由 2D 检测直接给出、可信；`z` 只是模型基于人体先验的估计。
所以这里衡量的不是"Z 有多准"（单目物理上做不到），而是 **Z 的误差有没有被压在可控范围**。

### 素材

4 段基准视频的编号、拍摄角度、机位、距离、分辨率与帧率见 `samples/benchmark/README.md`
（`BM-01` 正面全身 / `BM-02` 正侧面全身 / `BM-03` 侧面下蹲 / `BM-04` 朝镜头走近）。
视频本体不入库，放到 `samples/` 下按清单命名即可。

### 四项指标

| 指标 | 定义 | 期望方向 |
| --- | --- | --- |
| `z_jitter_{torso,limb,extremity}` | 相邻帧 `world_landmarks.z` 差值的 RMS（米）；按 躯干(11/12/23/24) / 四肢(13/14/25/26) / 末端(15/16/27/28/31/32) 分组 | 越小越好 |
| `bone_length_cv_{max,mean}` | 每条骨段长度的变异系数 `std/mean`（变异系数 = 标准差 ÷ 均值） | 基线预期 > 3%，目标 < 1.5% |
| `reproj_error_px` | 把 `world_landmarks` 反投影回归一化坐标后与 `landmarks` 的平均像素误差 | 越小说明 3D 与 2D 越一致 |
| `angle_drift_deg` | 膝/肘/髋角帧间差值的标准差（度） | 越小越好（Z 抖动会直接放大成角度抖动） |

### 复现命令

```powershell
# 真实素材（走完整链路，指标统计在后处理之后的帧上）
python scripts/bench_depth.py samples/BM-01-front-full.mp4 --json out/BM-01.json --label BM-01

# 与上次结果对比（回归检测，超过阈值会标"劣化"）
python scripts/bench_depth.py samples/BM-01-front-full.mp4 --baseline out/BM-01.json

# A/B：关掉新增处理，验证改动确实起作用
python scripts/bench_depth.py samples/BM-01-front-full.mp4 --no-ik-fit --no-z-smoothing

# 无素材 / 无模型的自检（确定性合成序列 + 固定种子的 Z 噪声）
python scripts/bench_depth.py --synthetic --json out/depth_baseline_synthetic.json
```

### 优化前基线

采集时间 2026-11-18，环境 `Windows / AMD64 / Python 3.13.3`。

**合成序列**（30 帧、1080×1920、`--z-noise 0.02` → Z 噪声 σ=0.02 m / 横向 0.006 m）：
序列由 `tests/synth_pose.py` 生成 —— 先造米制真值骨架，再用已知弱透视相机投影出 2D，
因此重投影误差与骨长误差都有明确物理含义。绝对数值不代表真实视频，只用于 A/B 对照。

三档对照（同一份数据，只切开关）：

| 指标 | A 改动前<br>`--no-smoothing --no-ik-fit` | B 只开平滑 | C 默认<br>（平滑 + 骨长拟合） | A→C 变化 |
| --- | --- | --- | --- | --- |
| `z_jitter_torso` | 0.02815 m | 0.00180 m | **0.00123 m** | −95.6% |
| `z_jitter_limb` | 0.03130 m | 0.00168 m | **0.00050 m** | −98.4% |
| `z_jitter_extremity` | 0.02948 m | 0.00157 m | **0.00055 m** | −98.1% |
| `z_jitter_overall` | 0.02967 m | 0.00168 m | **0.00083 m** | −97.2% |
| `xy_jitter_overall` | 0.01330 m | 0.00504 m | **0.00234 m** | −82.4% |
| `bone_length_cv_max` | 32.18% | 29.30% | **3.88%** | −87.9% |
| `bone_length_cv_mean` | 8.67% | 4.81% | **0.42%** | −95.2% |
| `reproj_error_px` | 6.14 px | 3.58 px | 11.05 px | 见下方说明 |
| `angle_drift_deg` | 4.80° | 0.48° | **0.18°** | −96.2% |

耗时：骨长约束拟合 **33~36 ms/帧**（纯 CPU，含 2 趟迭代细化），低于 50 ms 预算。

> `reproj_error_px` 的上升是**合成数据本身的特性**，不是拟合退化：`synth_pose` 用"整体平移手腕"
> 模拟摆臂，而平移改变了 `肘→腕` 的骨长（最多 0.12 m），所以这段数据**物理上不存在骨长恒定的姿态**。
> 骨长被钉死之后，拟合只能在"骨长对"和"2D 完全重合"之间折中，残差约 2 cm。
> 反证：对**不含摆臂**的帧（第 0/5 帧）单帧拟合，重投影误差是 **0.0 px**；
> 对整段相机自洽的干净数据，拟合不引入任何 2D 偏差（把 `temporal_axes` 设为 `"xyz"` 会看到
> 时域项把 x/y 拽向上一帧，那才是真正的偏差来源，已默认改为只压 z）。
> 真实素材上重投影误差应当"不劣化"，判读时看 `--baseline` 的`劣化`标记即可。

**真实素材基线**：`BM-01`~`BM-04` 需在本机录制后运行上面的命令补填
（仓库不含视频本体）。补填方式：`python scripts/bench_depth.py samples/BM-0X-*.mp4 --json out/BM-0X.json`，
把输出表格粘到上表的"改动前 / 改动后"两列。

### Z 轴强平滑调参（任务 4.4）

三组参数（只改 z，x/y 固定 1.0 / 1.0 米制）在合成序列上的表现，
由 `scripts/bench_z_smoothing.py` 生成（静止姿态测抖动、Z 阶跃测延迟）：

```powershell
python scripts/bench_z_smoothing.py --json out/z_params.json
```

| 参数组 | `min_cutoff_z` | `beta_z` | z 抖动 | 相比不滤波下降 | 阶跃延迟（30 fps） |
| --- | --- | --- | --- | --- | --- |
| weak（= x/y 同参数，改造前行为） | 1.0 | 1.0 | 0.00436 m | 85.2% | 0.99 帧（33 ms） |
| **default（选定）** | **0.3** | **0.3** | **0.00159 m** | **94.6%** | 3.71 帧（124 ms） |
| strong | 0.15 | 0.15 | 0.00090 m | 97.0% | 6.78 帧（226 ms） |

结论与取舍：

- 选 `default`：z 抖动已经压掉 94.6%，而阶跃延迟 124 ms 仍在"肉眼可接受"的范围（z 只影响前后方向）；
- 再加强收益递减（97% 只多 1.4 个百分点）却把延迟翻倍到 226 ms，快速前后移动会明显拖影；
- 若素材是"快速朝镜头走动"，宁可用 `weak`（延迟 33 ms）——那种场景本来就不该相信 z；
- 这三个值都是 `one_euro.Z_MIN_CUTOFF_SCALE` / `Z_BETA_SCALE` 派生的常量，改系数即可整体缩放。


### 身体 yaw：宽度比 vs world z 推算（任务 3.4）

两种算法在同一段数据上的对比：

- **新**：肩线/髋线的**投影宽度比** `cos(yaw) ≈ 观测宽度 / 参考宽度`（`skeleton/orientation.py`），
  参考宽度来自整段标定的骨长，尺度由"躯干竖直跨度 ÷ 躯干骨长"逐帧自标定（自带透视校正）；
- **旧**：直接用 `world_landmarks` 的肩线 `z` 差算 `atan2(-dz, dx)`（`orientation.yaw_from_world_z`）。

复现命令：

```powershell
# 30 帧合成序列，整体旋转 40°，Z 噪声 σ=0.02 m；同时打印两种算法的均值 / 抖动
python scripts/bench_depth.py --synthetic --rotation 40 --no-ik-fit --no-z-smoothing --label yaw-raw
python scripts/bench_depth.py --synthetic --rotation 40 --label yaw-fit
pytest tests/test_yaw.py -q
```

采集时间 2026-11-18，`Windows / AMD64 / Python 3.13.3`，24 帧、`z_sigma=0.03 m`：

| 真实 yaw | 新算法均值 | 新算法帧间抖动 | 旧算法均值 | 旧算法帧间抖动 |
| --- | --- | --- | --- | --- |
| 20° | 20.02° | **0.00°** | 20.10° | 7.70° |
| 45° | 45.02° | **0.00°** | 44.96° | 5.96° |
| 60° | 59.97° | **0.00°** | 59.93° | 4.38° |

基准脚本口径（含整条后处理链路，30 帧、旋转 40°）：

| 指标 | 关掉拟合/平滑 | 默认（拟合开启） |
| --- | --- | --- |
| `yaw_error_width_deg` | 0.255° | 0.242° |
| `yaw_jitter_width_deg` | **0.000°** | **0.000°** |
| `yaw_jitter_depth_deg` | 4.664° | 0.198° |

读法：

- 合成数据里 2D 是干净的，所以宽度比方法的帧间抖动是 0，而 z 推算每帧都在抖 4~8° ——
  这正是"z 是猜出来的、宽度是量出来的"的量化体现；
- 真机上 2D 也带噪声，宽度比方法不会真的为 0，但它的误差来源与 2D 同量级（亚像素级），
  远小于深度噪声（厘米级）；`confidence` 在比值≈1（正对）与≈0（正侧）两端自动下降；
- 开启骨长约束拟合后 `yaw_jitter_depth_deg` 也降到 0.198°，说明"固定骨长 + 2D 重投影"
  本身就能压住 z 的随机抖动；但**结构性**的 z 偏差（例如整段时间内深度整体偏大）它压不住，
  那部分只有宽度比方法能纠，这也是任务 3.2 把 yaw 先验塞进拟合的原因。

