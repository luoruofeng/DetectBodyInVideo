import * as THREE from 'three'

import type { PreviewFrame } from '../api/analysis'

export type JointMap = Map<string, THREE.Vector3>

export interface MeshBall {
  position: THREE.Vector3
  radius: number
  color: number
}

type Vec3Tuple = readonly [number, number, number]

const SIDES = ['left', 'right'] as const
const FOOT_JOINTS = [
  'left_foot_joint',
  'right_foot_joint',
  'left_toes_joint',
  'right_toes_joint',
] as const

const WHITE = 0xf8fbff

/** 白膜半径按关键点量出的体格等比缩放；参考值取成年人站立高度。 */
const REFERENCE_STATURE = 1.9

/**
 * 腰腹半宽（米，1.9m 基准）。两条大腿的直径之和 ≤ 腰直径：
 * 单腿半径取腰半宽的 45%，两条合计 90% 腰宽，中间还留出裆部间隙。
 */
const WAIST_HALF_WIDTH = 0.11
const UPPER_THIGH_RADIUS = WAIST_HALF_WIDTH * 0.45

/** 髋局部坐标下的兜底姿态；仅用于补齐缺失关节，避免曲面断裂。 */
const REST_JOINTS: Record<string, Vec3Tuple> = {
  root: [0, 0, 0],
  hips_joint: [0, 0, 0],
  spine_3_joint: [0, 0.22, 0],
  spine_7_joint: [0, 0.48, 0],
  neck_1_joint: [0, 0.67, 0],
  head_joint: [0, 0.8, 0],
  left_shoulder_1_joint: [0.2, 0.52, 0],
  left_arm_joint: [0.32, 0.28, 0],
  left_forearm_joint: [0.38, 0.1, 0],
  left_hand_joint: [0.42, -0.06, 0],
  right_shoulder_1_joint: [-0.2, 0.52, 0],
  right_arm_joint: [-0.32, 0.28, 0],
  right_forearm_joint: [-0.38, 0.1, 0],
  right_hand_joint: [-0.42, -0.06, 0],
  left_upLeg_joint: [0.1, -0.06, 0],
  left_leg_joint: [0.12, -0.48, 0],
  left_foot_joint: [0.12, -0.86, 0],
  left_toes_joint: [0.12, -0.91, -0.15],
  right_upLeg_joint: [-0.1, -0.06, 0],
  right_leg_joint: [-0.12, -0.48, 0],
  right_foot_joint: [-0.12, -0.86, 0],
  right_toes_joint: [-0.12, -0.91, -0.15],
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function vecFromTuple(value: Vec3Tuple): THREE.Vector3 {
  return new THREE.Vector3(value[0], value[1], value[2])
}

function readJoint(frame: PreviewFrame | null, name: string): THREE.Vector3 | null {
  const joint = frame?.joints[name]
  if (!joint || joint.visibility <= 0.05) return null
  if (![joint.x, joint.y, joint.z].every(Number.isFinite)) return null
  return new THREE.Vector3(joint.x, joint.y, joint.z)
}

function midpoint(a: THREE.Vector3, b: THREE.Vector3): THREE.Vector3 {
  return a.clone().lerp(b, 0.5)
}

/** 把 a→b 投影到与 up 垂直的平面，得到稳定的身体横向轴。 */
function projectedAxis(
  from: THREE.Vector3,
  to: THREE.Vector3,
  up: THREE.Vector3,
  fallback: THREE.Vector3,
): THREE.Vector3 {
  const axis = to.clone().sub(from)
  axis.addScaledVector(up, -axis.dot(up))
  if (axis.lengthSq() < 1e-8) {
    axis.copy(fallback).addScaledVector(up, -fallback.dot(up))
  }
  if (axis.lengthSq() < 1e-8) {
    axis.set(1, 0, 0).addScaledVector(up, -up.x)
    if (axis.lengthSq() < 1e-8) axis.set(0, 0, 1).addScaledVector(up, -up.z)
  }
  return axis.normalize()
}

/**
 * 股骨头（大腿根）位置：两髋间距按腰宽钳制，避免误检把大腿甩到腰外；
 * 髋口球心略低于骨盆、略内收，衔接是渐变的楔形而不是横跨两髋的粗球链。
 */
function thighRootPoint(
  hip: THREE.Vector3,
  hips: THREE.Vector3,
  hipAxis: THREE.Vector3,
  up: THREE.Vector3,
  sideSign: number,
  scale: number,
  maxDrop: number,
): THREE.Vector3 {
  const relative = hip.clone().sub(hips)
  const minOffset = UPPER_THIGH_RADIUS * 1.08 * scale
  const maxOffset = (WAIST_HALF_WIDTH - UPPER_THIGH_RADIUS) * scale
  const offset = sideSign * clamp(Math.abs(relative.dot(hipAxis)), minOffset, maxOffset)
  const tilt = clamp(relative.dot(up), -0.025 * scale, 0.012 * scale)
  return hips
    .clone()
    .addScaledVector(hipAxis, offset)
    .addScaledVector(up, tilt - Math.min(0.03 * scale, maxDrop))
}

/** 将 PreviewFrame.joints 补齐成白膜构建所需的完整关节集合。 */
export function resolvePoseJoints(frame: PreviewFrame | null): JointMap {
  const joints: JointMap = new Map()
  for (const name of Object.keys(REST_JOINTS)) {
    joints.set(name, readJoint(frame, name) ?? vecFromTuple(REST_JOINTS[name]))
  }

  const hips = readJoint(frame, 'hips_joint')
  const neck = readJoint(frame, 'neck_1_joint')
  if (hips && neck) {
    if (!readJoint(frame, 'spine_3_joint')) {
      joints.set('spine_3_joint', hips.clone().lerp(neck, 0.4))
    }
    if (!readJoint(frame, 'spine_7_joint')) {
      joints.set('spine_7_joint', hips.clone().lerp(neck, 0.75))
    }
  }

  const head = readJoint(frame, 'head_joint')
  const leftShoulder = readJoint(frame, 'left_shoulder_1_joint')
  const rightShoulder = readJoint(frame, 'right_shoulder_1_joint')
  if (!neck && leftShoulder && rightShoulder) {
    joints.set(
      'neck_1_joint',
      midpoint(leftShoulder, rightShoulder).lerp(head ?? vecFromTuple(REST_JOINTS.head_joint), 0.35),
    )
  }

  for (const side of SIDES) {
    const forearmName = `${side}_forearm_joint`
    const elbow = readJoint(frame, `${side}_arm_joint`)
    const wrist = readJoint(frame, `${side}_hand_joint`)
    if (!readJoint(frame, forearmName) && elbow && wrist) {
      joints.set(forearmName, midpoint(elbow, wrist))
    }

    const toesName = `${side}_toes_joint`
    const ankle = readJoint(frame, `${side}_foot_joint`)
    if (!readJoint(frame, toesName) && ankle) {
      joints.set(toesName, ankle.clone().add(new THREE.Vector3(0, -0.035, -0.13)))
    }
  }

  return joints
}

/** 脚底贴地所需的髋高度（与 3D 骨架卡片保持一致）。 */
export function groundedHipHeightFromJoints(joints: JointMap): number {
  let lowest = Infinity
  for (const name of FOOT_JOINTS) {
    const point = joints.get(name)
    if (point && Number.isFinite(point.y)) lowest = Math.min(lowest, point.y)
  }
  if (!Number.isFinite(lowest)) return 0.9
  return clamp(-lowest, 0.45, 1.25)
}

/**
 * 躯干局部坐标：up 沿脊柱向上，lateral 指向人物左侧，front 指向人物正前方（-z）。
 * 用身体自身坐标而不是世界坐标摆放胸廓/头部体积，躺姿或侧身时白膜才不会歪。
 */
function bodyFrame(
  hips: THREE.Vector3,
  neck: THREE.Vector3,
  leftShoulder: THREE.Vector3,
  rightShoulder: THREE.Vector3,
): { up: THREE.Vector3; lateral: THREE.Vector3; front: THREE.Vector3 } {
  const up = neck.clone().sub(hips)
  if (up.lengthSq() < 1e-8) up.set(0, 1, 0)
  up.normalize()

  const lateral = leftShoulder.clone().sub(rightShoulder)
  lateral.addScaledVector(up, -lateral.dot(up))
  if (lateral.lengthSq() < 1e-8) {
    lateral.crossVectors(new THREE.Vector3(0, 0, 1), up)
  }
  lateral.normalize()

  const front = new THREE.Vector3().crossVectors(up, lateral).normalize()
  return { up, lateral, front }
}

/** 取关节坐标；缺失时退回髋局部坐标下的兜底姿态。 */
export function jointPoint(joints: JointMap, name: string): THREE.Vector3 {
  const point = joints.get(name)
  if (point) return point
  const rest = REST_JOINTS[name]
  return rest ? vecFromTuple(rest) : new THREE.Vector3()
}

/** 躯干坐标系（up/lateral/front）：头颅、双手等刚性部件与白膜共用同一套朝向。 */
export function bodyBasis(joints: JointMap): {
  up: THREE.Vector3
  lateral: THREE.Vector3
  front: THREE.Vector3
} {
  return bodyFrame(
    jointPoint(joints, 'hips_joint'),
    jointPoint(joints, 'neck_1_joint'),
    jointPoint(joints, 'left_shoulder_1_joint'),
    jointPoint(joints, 'right_shoulder_1_joint'),
  )
}

/** 体格缩放系数：用各段骨长估计，躺姿/倒立时比例也不会走样。 */
export function estimateBodyScale(joints: JointMap): number {
  const legLength =
    (jointPoint(joints, 'left_upLeg_joint').distanceTo(jointPoint(joints, 'left_leg_joint')) +
      jointPoint(joints, 'left_leg_joint').distanceTo(jointPoint(joints, 'left_foot_joint')) +
      jointPoint(joints, 'right_upLeg_joint').distanceTo(jointPoint(joints, 'right_leg_joint')) +
      jointPoint(joints, 'right_leg_joint').distanceTo(jointPoint(joints, 'right_foot_joint'))) /
    2
  const torsoLength =
    jointPoint(joints, 'hips_joint').distanceTo(jointPoint(joints, 'neck_1_joint')) +
    jointPoint(joints, 'neck_1_joint').distanceTo(jointPoint(joints, 'head_joint')) +
    0.1
  const stature = legLength + torsoLength + 0.08
  return Number.isFinite(stature) ? clamp(stature / REFERENCE_STATURE, 0.82, 1.22) : 1
}

/**
 * 把稀疏关节展开成隐式曲面的 metaball 采样点。
 * 白膜不是精确人体重建：这里用人体比例先验补全躯干厚度、胸腔与脚掌，
 * 让任意姿态下都能读出一个明确的人形轮廓。
 * 头颅与双手由 ``bodyParts`` 用解析几何单独生成——96³ 的隐式曲面画不出五指与颅型。
 */
export function buildMeshBalls(joints: JointMap): MeshBall[] {
  const balls: MeshBall[] = []
  const get = (name: string): THREE.Vector3 => jointPoint(joints, name)

  const hips = get('hips_joint')
  const spine3 = get('spine_3_joint')
  const spine7 = get('spine_7_joint')
  const neck = get('neck_1_joint')
  const head = get('head_joint')
  const leftShoulder = get('left_shoulder_1_joint')
  const rightShoulder = get('right_shoulder_1_joint')
  const leftHip = get('left_upLeg_joint')
  const rightHip = get('right_upLeg_joint')

  const s = estimateBodyScale(joints)
  const { up, lateral, front } = bodyBasis(joints)
  const shoulderSpan = leftShoulder.distanceTo(rightShoulder)
  const chestHalf = clamp(shoulderSpan * 0.2, 0.06, 0.1)
  // 髋横轴优先取两髋连线（投影到骨盆平面），比肩线更贴近下半身的实际朝向。
  const hipAxis = projectedAxis(rightHip, leftHip, up, lateral)
  const hipSocketRadius = UPPER_THIGH_RADIUS * 0.98
  const kneeRadius = UPPER_THIGH_RADIUS * 0.78

  const addBall = (point: THREE.Vector3, radius: number, color = WHITE) => {
    const scaled = radius * s
    if (![point.x, point.y, point.z, scaled].every(Number.isFinite)) return
    if (scaled <= 0) return
    balls.push({ position: point.clone(), radius: scaled, color })
  }

  /** 半径线性渐变的“肉柱”，比等半径胶囊更像四肢。 */
  const addLimb = (
    from: THREE.Vector3,
    to: THREE.Vector3,
    fromRadius: number,
    toRadius: number,
    color = WHITE,
  ) => {
    const distance = from.distanceTo(to)
    if (!Number.isFinite(distance) || distance < 1e-4) {
      addBall(from, Math.max(fromRadius, toRadius), color)
      return
    }
    const step = Math.max(Math.min(fromRadius, toRadius) * 0.45, 0.014)
    const count = Math.max(1, Math.ceil(distance / step))
    for (let i = 0; i <= count; i += 1) {
      const k = i / count
      addBall(from.clone().lerp(to, k), fromRadius + (toRadius - fromRadius) * k, color)
    }
  }

  /**
   * 脚掌：沿"脚跟→脚尖"铺一条连续的球链，球心高度按各自半径抬高，
   * 让所有球的底面落在同一水平面上，才不会出现脚背高、脚跟飘的畸形形状。
   */
  const addFoot = (ankle: THREE.Vector3, toes: THREE.Vector3) => {
    // 脚掌朝向只取水平分量（垂直于身体 up）：脚趾关键点偏低时脚不会往地里扎。
    const sole = toes.clone().sub(ankle)
    sole.addScaledVector(up, -sole.dot(up))
    if (sole.lengthSq() < 1e-6) sole.copy(front)
    sole.normalize()

    // 踝关节到脚底的高度由踝/脚尖高差推出来，并夹在人体比例的合理区间。
    const reach = ankle.distanceTo(toes) * s
    const toeTip = clamp(reach, 0.11 * s, 0.18 * s) + 0.02 * s
    const heel = -0.05 * s
    // 脚底比最低的踝/脚尖关节再低 5 mm：贴地又不至于把脚陷进地面网格里。
    const soleDrop = clamp(ankle.dot(up) - toes.dot(up) + 0.005, 0.035, 0.085)

    const addSoleBall = (along: number, radius: number) => {
      addBall(
        ankle
          .clone()
          .addScaledVector(sole, along)
          .addScaledVector(up, radius - soleDrop),
        radius,
      )
    }

    // 从脚跟到脚尖逐渐收窄；间距压到半径以内，脚趾不会脱离脚掌变成孤立的圆点。
    const segments = Math.max(4, Math.ceil((toeTip - heel) / (0.026 * s)))
    for (let i = 0; i <= segments; i += 1) {
      const k = i / segments
      addSoleBall(heel + (toeTip - heel) * k, 0.04 - 0.014 * k ** 1.6)
    }
  }

  // ── 骨盆与腰腹 ────────────────────────────────────────────────
  // 骨盆上口接腰、下口分成两条独立的股骨通道；不再横跨两髋拉粗球链，
  // 否则两腿会和胯糊成一整块，外沿比腰还宽。
  addBall(hips, WAIST_HALF_WIDTH * 0.98)
  addBall(
    hips.clone().addScaledVector(front, -0.03).addScaledVector(up, -0.01),
    WAIST_HALF_WIDTH * 0.9,
  )

  addLimb(hips, spine3, WAIST_HALF_WIDTH * 0.95, WAIST_HALF_WIDTH * 0.88)
  addLimb(spine3, spine7, 0.102, 0.122)
  addBall(spine7, 0.126)
  addBall(spine7.clone().addScaledVector(front, 0.02), 0.11)

  // ── 胸腔与肩带 ────────────────────────────────────────────────
  addBall(spine7.clone().addScaledVector(lateral, chestHalf).addScaledVector(front, 0.02), 0.09)
  addBall(spine7.clone().addScaledVector(lateral, -chestHalf).addScaledVector(front, 0.02), 0.09)
  addBall(spine7.clone().addScaledVector(front, -0.035), 0.096)
  addLimb(leftShoulder, rightShoulder, 0.055, 0.055)
  addLimb(neck, leftShoulder, 0.044, 0.048)
  addLimb(neck, rightShoulder, 0.044, 0.048)

  // ── 颈 ────────────────────────────────────────────────────────
  // 头颅由 bodyParts 的椭球生成（球堆在 96³ 网格上只会堆出一颗畸形的脑袋），
  // 这里只留一截脖子，末端球与颅底重叠，衔接处不会露缝。
  addLimb(neck, head, 0.042, 0.040)

  for (const side of SIDES) {
    // ── 手臂 ────────────────────────────────────────────────────
    const shoulder = get(`${side}_shoulder_1_joint`)
    const elbow = get(`${side}_arm_joint`)
    const forearm = get(`${side}_forearm_joint`)
    const hand = get(`${side}_hand_joint`)

    addBall(shoulder, 0.07)
    addLimb(shoulder, elbow, 0.056, 0.044)
    addBall(elbow, 0.044)
    addLimb(elbow, forearm, 0.044, 0.037)
    addLimb(forearm, hand, 0.037, 0.028)
    // 手掌与五指由 bodyParts 的解析几何生成，这里只把前臂收在腕关节上。
    addBall(hand, 0.028)

    // ── 腿与脚 ──────────────────────────────────────────────────
    const hip = get(`${side}_upLeg_joint`)
    const knee = get(`${side}_leg_joint`)
    const ankle = get(`${side}_foot_joint`)
    const toes = get(`${side}_toes_joint`)

    // 用钳制后的大腿根代替原始髋点：误检/大劈叉时大腿也不会比腰还粗。
    const sideSign = side === 'left' ? 1 : -1
    const root = thighRootPoint(hip, hips, hipAxis, up, sideSign, s, hip.distanceTo(knee) * 0.12)
    const socket = root.clone().addScaledVector(front, -0.012 * s).addScaledVector(up, 0.008 * s)
    addLimb(hips, socket, WAIST_HALF_WIDTH * 0.7, hipSocketRadius)
    addBall(socket, hipSocketRadius)
    addBall(socket.clone().addScaledVector(front, -0.016 * s), hipSocketRadius * 0.9)

    addLimb(root, knee, UPPER_THIGH_RADIUS, kneeRadius)
    addBall(knee, kneeRadius)
    addLimb(knee, ankle, kneeRadius * 0.94, 0.032)
    addBall(ankle, 0.033)
    addFoot(ankle, toes)
  }

  return balls
}
