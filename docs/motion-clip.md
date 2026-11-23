# 动作 JSON（iOS App 格式）

把「拖入视频 → 逐帧识别 → 导出」的结果再转换成 iOS App 直接播放的动作文件，
字段与语义对齐 `resources/air-squat-motion.json`（App 内置的深蹲示例）。

## 1. 本地参考素材（`resources/`，不入库）

| 文件 | 用途 |
| --- | --- |
| `AppleModel.usda` | 苹果 App 使用的骨骼参考模型。91 个 ARKit 关节（`root/hips_joint/.../left_handPinkyEnd_joint`），动作 JSON 的关节命名与骨长比例都以它为准。 |
| `demo.mp4` | 软件测试用视频（1080×1188，30fps，约 12.5 分钟）。 |
| `air-squat-motion.json` | App 的动作 JSON 样例（格式参考，51 帧 / 5 秒 / 10Hz）。 |

三份文件体积大或受第三方许可约束，`.gitignore` 里整体忽略 `resources/`，
克隆仓库后需自行放回。

## 2. 与「姿态 JSON」的区别

| | 姿态 JSON（`*_pose*.json`） | 动作 JSON（`*_motion.json`） |
| --- | --- | --- |
| 用途 | 本项目自有契约，便于复核/二次处理 | 直接喂给 iOS App 的 `MotionFrame` 解码器 |
| 坐标系 | `arkit-right-handed-y-up`（髋为原点） | `hip-local-meters-yup`（pelvis 为原点，Y 向上） |
| 时间轴 | 跟随视频真实帧率，`t` 为秒 | 固定间隔重采样，`t` 为毫秒（默认 100ms / 10Hz） |
| 关节 | 18 个 ARKit 关节（可选 22 个） | App 的 17 个身体关节，改名并去掉 `_joint` 后缀 |
| 骨盆升降 | 放在 `rootMotion` | `frames[].pelvisOffsetMeters` |
| 角度 / 质检 | 有 | 无（App 侧不用） |

运行时不把坐标直接拷到模型上：它只把「两个关节连成的向量」当作目标骨骼方向，
再换算成每根骨骼相对父骨骼的旋转；骨长与位置全部沿用当前模型自己的 rest 骨架。
因此 JSON 里的骨长比例不参与驱动，换模型也能得到同一个姿势。

**这条"只取方向 + 模型自身骨长"的约定，正是 Z 误差不会放大成体型变化的原因**：
单目视频的前后位置（`z`）只是估计值，如果把它当成"绝对坐标"去驱动模型，
深度误差会直接变成关节拉伸/忽大忽小；只取方向时，`z` 的误差最多表现为**旋转角度的轻微偏差**，
而体型的骨长完全由模型决定，永远不会被带偏。这也是为什么本项目可以放心地用
`world` 模式（竖直位移由脚部着地锚定恢复下蹲下沉，水平位移由脚底水平锚定反解成髋相对脚的运动）作为默认导出模式。

## 3. 关节映射

| 动作 JSON | 本项目（ARKit） |
| --- | --- |
| `root` / `pelvis` | `root` / `hips_joint`（原点，恒为 `[0,0,0]`） |
| `spine` | `spine_3_joint` |
| `centerShoulder` | `spine_7_joint` |
| `neck` | `neck_1_joint` |
| `leftShoulder` / `rightShoulder` | `left/right_shoulder_1_joint` |
| `leftElbow` / `rightElbow` | `left/right_arm_joint` |
| `leftWrist` / `rightWrist` | `left/right_hand_joint` |
| `leftHip` / `rightHip` | `left/right_upLeg_joint` |
| `leftKnee` / `rightKnee` | `left/right_leg_joint` |
| `leftAnkle` / `rightAnkle` | `left/right_foot_joint` |

手指关节（每手 24 个、共 48 个）不生成：主链路只有 33 点姿态，没有手部数据。
运行时未绑定的关节会继承父骨骼姿态，不会回弹到 T-Pose；后续接入手部关键点后
可以在这里补 `*_handIndex_1_joint` 等名字。

## 4. 字段说明

```json
{
  "schemaVersion": 1,
  "exerciseID": "air-squat",
  "name": "Air Squat",
  "nameZh": "深蹲",
  "coordinateSpace": "hip-local-meters-yup",
  "bodyOrientation": "upright",
  "referenceModel": "t-pose_humanbody_2048.usdz",
  "mirroredReference": true,
  "sampleIntervalMs": 100,
  "durationMs": 5000,
  "loop": false,
  "demonstrationLoops": 1,
  "skeleton": ["root", "pelvis", "spine", "..."],
  "keyframesMs": [0, 5000],
  "playbackHolds": [],
  "frames": [
    {
      "t": 0,
      "joints": { "pelvis": [0, 0, 0], "leftKnee": [0.12, -0.45, 0.01] },
      "jointVisibility": { "root": 0.85 },
      "pelvisOffsetMeters": [0, 0, 0],
      "confidence": 0.93
    }
  ],
  "viewpoint": { "subject": "source-video-subject-facing-player", "forwardAxis": "-Z" },
  "jointVisibilityNote": "jointVisibility 只列 visibility < 1 的关节，缺省视为 1；..."
}
```

- `t`：相对片段起点的毫秒；`durationMs` 等于最后一帧的 `t`，因此采样率恒定。
- `joints`：米制绝对坐标，原点在 pelvis（`pelvis` / `root` 恒为 0），Y 向上、-Z 向前。
- `jointVisibility`：只列 `visibility < 1` 的关节（保留 2 位），缺省视为 1。
- `confidence`：该帧 17 个关节的平均可见度（0~1）。
- `pelvisOffsetMeters`：骨盆相对「站立参考帧」的位移。参考帧取腿最伸直（骨盆离脚踝最高）的那一帧。
  - `rootMotion` 有非零位移（world 模式下由脚部着地锚定恢复的髋部下沉，以及脚底水平锚定反解的髋相对脚水平位移）时直接用 root 位移差分；
  - 没有可用位移时，用「骨盆到脚踝的竖直距离」变化估算下沉量，
    **腾空无法还原**（日志里会写明）。

## 5. 重采样规则

- 默认 100ms；`frames[].t` 严格递增，最后一帧不补齐到整秒。
- 两帧有效姿态之间线性插值；`valid=false` 的帧不参与插值。
- 两侧有效帧间隔超过 400ms（`hold_gap_ms`）时保持前一帧，不伪造中间动作。
- 坐标保留 4 位小数（0.1mm），并消除 `-0.0`。

## 6. 使用方式

界面：右侧「导出」卡片 → 选择「动作 JSON」→ 导出。

接口：

```http
POST /api/analysis/{jobId}/export
{
  "format": "motion",
  "output_dir": "D:/motions",
  "motion": {
    "sample_interval_ms": 100,
    "loop": false,
    "name": "Air Squat",
    "name_zh": "深蹲"
  }
}
```

文件名规则：`{原名}_{时间戳}_motion.json`，默认写入 `~/.detect-body/exports/`。

命令行（把已有的姿态 JSON 转成动作 JSON）：

```powershell
python -m app.backend.export.motion_clip clip_posefull.json -o clip_motion.json --name "Air Squat" --name-zh "深蹲"
```

## 7. 已知限制

- 手指、面部不导出（主链路没有手部关键点）。
- `pelvisOffsetMeters` 只能还原下蹲/屈膝带来的下沉，无法还原跳跃腾空（单目无绝对竖直位移）。
- `mirroredReference` 固定为 `true`（与 App 样例一致）；如果源视频是自拍镜像，
  人物左右会与真实相反，需在 App 侧或转码前翻转画面。
- **`rootMotion` 的前后（Z）位移不可信**：`mode == "world"` 时 `rootMotion.frames[].translation.z`
  是单目估计值，`totalTranslation` 也只能当"相对变化量"看。`x` / `z` 还来自脚底水平锚定，
  描述的是"脚踩住地面时髋相对脚的运动"（脚不滑步），人物真的整体平移时会被吃掉一部分。
  需要真实的走位请改用多机位/深度相机等外部数据；`pelvisOffsetMeters` 会带上这两个分量，
  App 侧若只想还原上下起伏，取 `[1]` 即可。
- 本文件不携带 `depthConfidence` / `warnings`（App 解码器不认识这些字段）。
  深度是否可信请看同时导出的姿态 JSON，或界面上的「深度置信度」；两者来自同一份分析结果。
