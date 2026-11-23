# 导出 JSON Schema（v1.1.0）

本文件是 `app/backend/export/schema.py` 的**面向消费方**说明。契约本身以 Pydantic 模型为准，
`schema.json` 是它的存档（`python -m app.backend.export.schema` 重新生成）。

参考样例：`samples/expected_pose.json`（确定性生成，不含模型权重）。

---

## 1. 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `version` | string | `schemaVersion`，当前 `1.1.0` |
| `generator` | string | 生成器标识 |
| `source` | object | 源视频元信息：`file` / `fps` / `width` / `height` / `frames` / `durationSeconds` |
| `coordinateSystem` | string | 恒为 `arkit-right-handed-y-up`（米制、右手系、y 向上、-z 向前） |
| `units` | string | 恒为 `meters` |
| `depthIsEstimated` | bool | **恒为 `true`**：`z`（前后方向）是单目估计值 |
| `depthConfidence` | number | 0~1，整段视频的深度可信度（与 `quality.depthConfidence` 同值） |
| `warnings` | string[] | 中文可信度提示；`depthConfidence` 低于 0.6 时才非空 |
| `personHeightMeters` | number | 由世界坐标自动测量的参考高度（头中心→脚底），用于核对模型缩放，不参与坐标计算 |
| `jointNames` | string[] | 本文件出现的关节名（ARKit `ARSkeleton.JointName` 命名） |
| `frames` | object[] | 逐帧数据，见第 2 节 |
| `quality` | object | 数据质检，见第 3 节 |
| `rootMotion` | object \| null | 根运动，见第 4 节 |
| `notes` | string[] | 生成过程中的说明 |

## 2. 帧（`frames[]`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | int | 帧号，严格递增 |
| `t` | number | 相对视频起点的秒数 |
| `valid` | bool | `false` 表示该帧没有可信姿态（遮挡 / 未检出） |
| `interpolated` | bool | 该帧坐标由插值补齐 |
| `joints` | object | `{关节名: {x, y, z, visibility}}`，米制 |
| `angles` | object | `{角度名: 0~180}`，仅 `full` 粒度 |
| `rawLandmarks` | object \| null | 原始 33 点，仅导出时显式开启 |

`joints` 是**髋局部位移**，播放时需叠加 `rootMotion` 把 root 放回场景。`world` 模式下：`y` 由脚部着地锚定按髋到脚的跨度恢复，使下蹲时髋部下沉、脚底保持在地面；`x` / `z` 由**脚底水平锚定**（双脚踝点按着地加权、零相位低通后逐帧累加）反解，表示"脚踩住地面时髋相对脚怎么动"，因此原地站立时脚不会跟着髋在地面上滑。`in_place` 模式的位移全部为 0（水平锚定只在预览里生效）。

> 不叠加 `rootMotion` 直接渲染 `joints` 时，脚仍会随髋摆动 —— 这是 `joints` 是髋局部坐标的必然结果，不是数据错误。

## 3. 质检（`quality`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `framesTotal` / `framesValid` | int | 总帧 / 有效帧 |
| `validFrameRatio` | number | 有效帧比例 0~1 |
| `interpolatedFrames` | int | 插值补齐的帧数 |
| `meanVisibility` / `meanScore` | number | 平均可见度 / 平均模型置信度 |
| `maxJump` / `maxJumpFrame` | number / int | 相邻帧最大位移（归一化）及其帧号 |
| `zJitterRms` | number | **深度抖动**：相邻帧 `z` 的 RMS（米） |
| `boneLengthCv` | number | **骨长变异系数**最大值（0~1），越小说明骨长越恒定 |
| `meanReprojError` | number | **2D 一致性**：3D 反投影与 2D 关键点的平均误差（归一化坐标） |
| `depthConfidence` | number | 上述三项加权得到的深度可信度 0~1 |
| `verdict` | string | `ready` / `review` / `rerecord` |
| `verdictLabel` | string | 中文结论 |
| `issues` | string[] | 问题清单（中文，可直接展示） |
| `smoothing` | object | 本次使用的平滑方法与参数（可复现） |
| `notes` | string[] | 运行说明 |

`depthConfidence` 的算法（`postprocess/quality.py::depth_confidence`）：

```
confidence = 1 - (0.5·min(1, z抖动/0.03) + 0.3·min(1, 骨长CV/0.30) + 0.2·min(1, 重投影/0.02))
```

判读：**≥0.75 可信、0.5~0.75 一般、<0.5 不可信**。低于 0.6 时结论最多只到 `review`。

> 参考样例 `samples/expected_pose.json` 的 `boneLengthCv` 约 12%，看起来和 0.88 的可信度"不匹配"：
> 那份样例的世界坐标是按归一化点位线性缩放出来的（并非真实骨长），骨长约束拟合会被
> 「2D 一致性不达标」拒绝执行，因此 `boneLengthCv` 反映的是原始输入的抖动，而
> `depthConfidence` 的权重里骨长只占 30%。真实视频上拟合生效后 `boneLengthCv` 通常 < 1%。

## 4. 根运动（`rootMotion`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | string | `world`（保留位移）/ `in_place`（位移恒为 0） |
| `totalTranslation` | number | 首尾净位移（米），**不是累计路程，前后方向不可信** |
| `convention` | string | 约定说明（`arkit-right-handed-y-up`） |
| `frames` | object[] | `{index, t, translation{x,y,z}, rotation{x,y,z,w}}` |

`translation` 的两个来源不同，都要按"相对量"使用：

- `y`：脚部着地锚定恢复的下蹲下沉（脚底保持在 `y = 0`，髋随下蹲下沉）。
- `x` / `z`：脚底水平锚定的结果 —— 双脚踝点按着地高度加权、3 Hz 零相位低通后逐帧累加。它描述"脚踩住地面时髋相对脚的运动"：脚固定不动、髋绕着脚晃，所以原地站立时脚不会滑步。它**不是绝对走位**：人物真的整体平移时，这部分位移会被抵掉（等同原地演示），`z` 还叠加了单目深度估计误差。

## 5. 消费方应当如何忽略 `z` 的绝对值

1. **不要**把 `joints[*].z` 当作"人物离镜头多远"。单目无法测绝对深度，误差可达数十厘米。
2. **不要**用 `rootMotion.frames[].translation.z` 判断"走了多远"。`mode == "world"` 时它的前后分量
   仍是估计值；需要原地播放时请用 `in_place`（位移恒 0，绝对不会跑偏）。
3. **可以**用 `z` 的**相对关系**：哪只手更靠前、转身方向、关节角度的三维计算。
4. 驱动 USDZ 时只取**关节方向**，骨长用模型自身骨架（见 `docs/motion-clip.md`），
   这样即便 `z` 有误差，也不会被放大成体型变化。
5. `warnings` 非空时请把内容展示给用户；`depthConfidence` 低意味着"这条数据的前后方向不要用"。

## 6. 版本与兼容

- 同 major（`1.x`）视为向后兼容：新增字段一律**可选且带默认值**，旧文件仍可读入；
  本次 `1.0.0 → 1.1.0` 只新增了 `depthIsEstimated` / `depthConfidence` / `warnings` 与
  `quality` 里的四个深度字段。
- 不兼容变更（改名、改单位、改必填）必须升 major，并在本文件写明迁移方式。
- 校验入口：`POST /api/analysis/{jobId}/export` 的响应里带 `schemaVersion`；
  iOS / macOS 侧可用 `schema.json` 做 JSON Schema 校验。

## 7. 重新生成

```powershell
python -m app.backend.export.schema    # 更新 app/backend/export/schema.json
python -m app.backend.export.samples   # 更新 samples/expected_pose.json（golden file）
pytest tests/test_export.py -q         # 校验 schema 与样例同步
```
