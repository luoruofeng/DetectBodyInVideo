# 深度（Z 轴）基准素材清单

这些素材用来量化"Z 方向的误差有多大、优化有没有真的生效"。视频本体体积大，
**不进仓库**（`.gitignore` 已忽略 `samples/*.mp4|mov|m4v`），只在这里记录编号与拍摄参数；
把文件按下列文件名放到本目录即可，`scripts/bench_depth.py` 直接用。

| 编号 | 文件名 | 动作 / 姿态 | 拍摄角度 | 机位高度 | 拍摄距离 | 分辨率 | 帧率 | 时长 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BM-01` | `BM-01-front-full.mp4` | 静止站立 + 抬臂（正面全身） | 正对镜头（yaw≈0°） | 与胸齐平（≈1.2 m） | 2.5 m | 1080×1920 | 30 | ≤10 s |
| `BM-02` | `BM-02-side-full.mp4` | 静止站立 + 抬臂（正侧面全身） | 正侧面（yaw≈90°） | 与胸齐平 | 2.5 m | 1080×1920 | 30 | ≤10 s |
| `BM-03` | `BM-03-side-squat.mp4` | 徒手深蹲（侧面） | 正侧面 | 与髋齐平（≈1.0 m） | 3.0 m | 1080×1920 | 30 | ≤10 s |
| `BM-04` | `BM-04-walk-in.mp4` | 从远处正面走近镜头 | 正对镜头 | 与胸齐平 | 5.0 m → 1.5 m | 1080×1920 | 30 | ≤10 s |
| `BM-05` | `BM-05-front-kick.mp4` | 原地前踢腿（正面全身，左右各 3 次） | 正对镜头（yaw≈0°） | 与胸齐平（≈1.2 m） | 2.5 m | 1080×1920 | 30 | ≤10 s |
| `BM-06` | `BM-06-side-kick.mp4` | 原地前踢腿（正侧全身，左右各 3 次） | 正侧面（yaw≈90°） | 与胸齐平 | 2.5 m | 1080×1920 | 30 | ≤10 s |
| `BM-07` | `BM-07-front30-kick.mp4` | 原地前踢腿（正面斜 30°） | 正面斜 30° | 与胸齐平 | 2.5 m | 1080×1920 | 30 | ≤10 s |

拍摄要求（直接影响指标能否复现）：

- 单一主体，画面里不要有其他人；主体占画面高度 60%–90%；
- 相机固定（三脚架），不开防抖、不裁剪；
- 光照均匀，不要逆光；衣服与背景有明显对比，避免宽松长袖遮住肘/腕；
- 每段都从"静止站立 3 秒"开始，方便量关节角漂移；`BM-04` 最后 3 秒保持站立。
- `BM-05`~`BM-07` 每段先**静止站立 3 秒**再踢：`temporal` 后端需要至少 30 帧的时序证据，
  开头那段静止画面就是它的样本；踢腿结束后也保持站立 2 秒（便于对齐回落相位）。

## 踢腿专项的人工标注

`BM-05`~`BM-07` 的幅度/符号真值统一放在 **`samples/benchmark/kick-labels.json`**（`BM-06` 正侧素材为准）：

```json
{
  "video": "BM-06-side-full.mp4",
  "status": "annotated",
  "fps": 30,
  "swingLeg": "left",
  "frames": [
    {"index": 0, "thighDeg": 8.0, "shankDeg": 3.0,
     "signs": {"left_knee": -1, "left_ankle": -1}}
  ],
  "phases": {"start": 12, "peak": 28, "return": 44}
}
```

- `thighDeg`：矢状面内大腿相对**竖直向下**的抬起角（站立 0°，水平前抬 90°）；
- `shankDeg`：小腿相对**大腿延长线**的屈曲角（伸直 0°，屈膝为正）；
- `signs`：该帧每根骨段 `sign(Δz)` 的真值，`-1 = 子关节更靠近镜头`（前踢方向）；
- `phases`：起脚 / 最大伸展 / 回落帧号，供 `detect_kick_phases()` 对比。

半自动流程（推荐）：先跑一遍侧视结果生成草稿，再逐帧修正角度与符号，最后把
`status` 改成 `annotated`：

```powershell
python scripts/bench_depth.py samples/benchmark/BM-06-side-full.mp4 `
  --depth-sign --depth-sign-mode temporal `
  --write-labels samples/benchmark/kick-labels.draft.json
```

> 草稿的 `status` 是 `auto-draft`，**不能当真值**：它的角度与符号来自估计结果本身。

指标口径与基线数值见 `docs/benchmark.md` 的「深度精度基准」与「踢腿专项」两节。
