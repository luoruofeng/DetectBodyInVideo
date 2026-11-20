import type { PreviewFrame } from '../api/analysis'

/** MediaPipe BlazePose 33 点名称（顺序与后端 LANDMARK_NAMES 一致）。 */
export const LANDMARK_NAMES = [
  'nose',
  'left_eye_inner',
  'left_eye',
  'left_eye_outer',
  'right_eye_inner',
  'right_eye',
  'right_eye_outer',
  'left_ear',
  'right_ear',
  'mouth_left',
  'mouth_right',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_pinky',
  'right_pinky',
  'left_index',
  'right_index',
  'left_thumb',
  'right_thumb',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
  'left_heel',
  'right_heel',
  'left_foot_index',
  'right_foot_index',
] as const

export const LANDMARK_LABELS_ZH: Record<string, string> = {
  nose: '鼻尖',
  left_eye_inner: '左眼内角',
  left_eye: '左眼',
  left_eye_outer: '左眼外角',
  right_eye_inner: '右眼内角',
  right_eye: '右眼',
  right_eye_outer: '右眼外角',
  left_ear: '左耳',
  right_ear: '右耳',
  mouth_left: '嘴角左',
  mouth_right: '嘴角右',
  left_shoulder: '左肩',
  right_shoulder: '右肩',
  left_elbow: '左肘',
  right_elbow: '右肘',
  left_wrist: '左腕',
  right_wrist: '右腕',
  left_pinky: '左小指',
  right_pinky: '右小指',
  left_index: '左食指',
  right_index: '右食指',
  left_thumb: '左拇指',
  right_thumb: '右拇指',
  left_hip: '左髋',
  right_hip: '右髋',
  left_knee: '左膝',
  right_knee: '右膝',
  left_ankle: '左踝',
  right_ankle: '右踝',
  left_heel: '左足跟',
  right_heel: '右足跟',
  left_foot_index: '左脚尖',
  right_foot_index: '右脚尖',
}

/** MediaPipe Pose 标准连线。 */
export const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [27, 31],
  [28, 32],
]

/** ARKit 18 关节 + 可选关节的骨骼连线（3D 预览用）。 */
export const ARKIT_BONES: ReadonlyArray<readonly [string, string]> = [
  ['root', 'hips_joint'],
  ['hips_joint', 'spine_3_joint'],
  ['spine_3_joint', 'spine_7_joint'],
  ['spine_7_joint', 'neck_1_joint'],
  ['neck_1_joint', 'head_joint'],
  ['spine_7_joint', 'left_shoulder_1_joint'],
  ['left_shoulder_1_joint', 'left_arm_joint'],
  ['left_arm_joint', 'left_hand_joint'],
  ['left_arm_joint', 'left_forearm_joint'],
  ['left_forearm_joint', 'left_hand_joint'],
  ['spine_7_joint', 'right_shoulder_1_joint'],
  ['right_shoulder_1_joint', 'right_arm_joint'],
  ['right_arm_joint', 'right_hand_joint'],
  ['right_arm_joint', 'right_forearm_joint'],
  ['right_forearm_joint', 'right_hand_joint'],
  ['hips_joint', 'left_upLeg_joint'],
  ['left_upLeg_joint', 'left_leg_joint'],
  ['left_leg_joint', 'left_foot_joint'],
  ['left_foot_joint', 'left_toes_joint'],
  ['hips_joint', 'right_upLeg_joint'],
  ['right_upLeg_joint', 'right_leg_joint'],
  ['right_leg_joint', 'right_foot_joint'],
  ['right_foot_joint', 'right_toes_joint'],
]

/** AppleModel.usda 全关节模式下的附加连线：脊柱/颈细分、趾尖、面部、双手手指。 */
const FINGER_PARTS = ['Start', '_1', '_2', '_3', 'End'] as const
const HAND_FINGERS = ['Pinky', 'Ring', 'Mid', 'Index', 'Thumb'] as const
const EXTRA_BONES: ReadonlyArray<readonly [string, string]> = [
  ['hips_joint', 'spine_1_joint'],
  ['spine_1_joint', 'spine_2_joint'],
  ['spine_2_joint', 'spine_3_joint'],
  ['spine_3_joint', 'spine_4_joint'],
  ['spine_4_joint', 'spine_5_joint'],
  ['spine_5_joint', 'spine_6_joint'],
  ['spine_6_joint', 'spine_7_joint'],
  ['neck_1_joint', 'neck_2_joint'],
  ['neck_2_joint', 'neck_3_joint'],
  ['neck_3_joint', 'neck_4_joint'],
  ['neck_4_joint', 'head_joint'],
  ['left_toes_joint', 'left_toesEnd_joint'],
  ['right_toes_joint', 'right_toesEnd_joint'],
  ['head_joint', 'jaw_joint'],
  ['jaw_joint', 'chin_joint'],
  ['head_joint', 'nose_joint'],
  ['left_eye_joint', 'left_eyeUpperLid_joint'],
  ['left_eye_joint', 'left_eyeLowerLid_joint'],
  ['left_eye_joint', 'left_eyeball_joint'],
  ['right_eye_joint', 'right_eyeUpperLid_joint'],
  ['right_eye_joint', 'right_eyeLowerLid_joint'],
  ['right_eye_joint', 'right_eyeball_joint'],
  ...(['left', 'right'] as const).flatMap((side) =>
    HAND_FINGERS.flatMap((finger) => {
      const parts = finger === 'Thumb'
        ? (['Start', '_1', '_2', 'End'] as const)
        : FINGER_PARTS
      const names = parts.map((part) => `${side}_hand${finger}${part}_joint`)
      return names.slice(0, -1).map((name, index) => [name, names[index + 1]] as const)
    }),
  ),
]

/** 3D 预览骨骼连线：始终包含 22 基础关节 + 全关节模式的附加连线。 */
export const ALL_APPLE_BONES: ReadonlyArray<readonly [string, string]> = [
  ...ARKIT_BONES,
  ...EXTRA_BONES,
]

/** 取 ``t`` 时刻最近的一帧（最后一个 t ≤ time 的帧）；空数组返回 -1。 */
export function findFrameIndex(frames: PreviewFrame[], time: number): number {
  if (!frames.length) return -1
  let low = 0
  let high = frames.length - 1
  if (time <= frames[0].t) return 0
  if (time >= frames[high].t) return high
  while (low <= high) {
    const mid = (low + high) >> 1
    if (frames[mid].t <= time) low = mid + 1
    else high = mid - 1
  }
  return Math.max(0, Math.min(frames.length - 1, high))
}

export interface HoveredLandmark {
  index: number
  px: number
  py: number
}

/** 归一化坐标 → 画布像素；同时返回最靠近鼠标的关键点。 */
export function pickLandmark(
  frame: PreviewFrame | null,
  width: number,
  height: number,
  px: number,
  py: number,
  radius = 14,
): HoveredLandmark | null {
  if (!frame) return null
  let best: HoveredLandmark | null = null
  let bestDist = radius
  frame.landmarks.forEach((point, index) => {
    const x = point.x * width
    const y = point.y * height
    const distance = Math.hypot(x - px, y - py)
    if (distance <= bestDist && point.visibility > 0) {
      bestDist = distance
      best = { index, px: x, py: y }
    }
  })
  return best
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.00'
  const minutes = Math.floor(seconds / 60)
  const rest = seconds - minutes * 60
  return `${minutes}:${rest.toFixed(2).padStart(5, '0')}`
}

export function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`
  return `${Math.round(value)}`
}
