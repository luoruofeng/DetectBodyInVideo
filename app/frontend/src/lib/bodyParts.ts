import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

import type { PreviewFrame } from '../api/analysis'
import { bodyBasis, estimateBodyScale, jointPoint, type JointMap } from './poseMesh'

/**
 * 白膜里"解析几何"的部分：头颅与双手。
 *
 * 96³ 的 metaball 隐式曲面一个体素约 2 cm，比一根手指还粗，
 * 用球堆永远只会糊成一块连指手套。所以这两处改用解析几何：
 * 手掌是压扁的椭球，五指的每节指骨是一条独立胶囊，头颅是单个椭球。
 *
 * 手指不是烘焙死的形状：每帧按骨骼映射输出的 AppleModel 指关节
 * （``left_handIndexStart_joint`` …）逐节摆放。指关节缺失时按
 * 「先冻结手型，超时后缓动回默认放松手型」降级，见 ``updateHandPart``。
 */

export type BodySide = 'left' | 'right'
/** AppleModel 的五根手指（与后端 ``HAND_JOINT_SOURCES`` 同名）。 */
export type FingerKey = 'Index' | 'Mid' | 'Ring' | 'Pinky' | 'Thumb'
/** 每根手指的关节链（世界坐标，顺序与 ``parts`` 一致）；``null`` = 这一帧没数据。 */
export type FingerChains = (THREE.Vector3[] | null)[]

/** 手部局部坐标：+Y 指向指尖，+X 为拇指侧，+Z 为掌心朝向。 */
const PALM_NORMAL = new THREE.Vector3(0, 0, 1)

/** 头颅椭球半径（米，1.9 m 成年人）：左右 / 上下 / 前后。 */
const HEAD_RADIUS = { lateral: 0.085, vertical: 0.125, front: 0.1 }
/** 颅心相对 head_joint（耳线/鼻尖附近）的偏移：略高一点，下颌才不会缺一截。 */
const HEAD_LIFT = 0.03

/** 指关节可信度门槛；低于它视为该手指这一帧没有数据。 */
const HAND_JOINT_VISIBILITY = 0.05
/** 指关节丢失后先原地冻结手型多久，再开始回弹（秒）。 */
const HAND_HOLD_SECONDS = 0.4
/** 回弹到默认放松手型的时长常数（秒）。 */
const HAND_RELAX_SECONDS = 0.18
/** 每帧向观测值靠拢的比例；播放时的手部高频抖动会被抹平。 */
const HAND_SHAPE_SMOOTHING = 0.35
/**
 * 观测到的整根手指跨度（掌指 → 指尖）相对默认手型的合理区间。
 * 超出说明这一帧的关键点整体塌陷/炸开，只保留上一手型，免得手指乱摆。
 * 注意判据只有"整体跨度"一项：单节指骨的长度在远景下本来就抖得厉害，
 * 拿它逐节卡容差会让整根手指被反复丢弃（表现为白膜手指一动都不动）。
 */
const HAND_SPAN_MIN_RATIO = 0.15
const HAND_SPAN_MAX_RATIO = 2.5
/** 掌指关节位置沿用观测值，但与默认位置的偏差按该权重收敛，手指不会飞离手掌。 */
const HAND_KNUCKLE_WEIGHT = 0.55
/** 指尖关节（End）由后端从末节方向外推 0.35 节，这里保持一致。 */
const TIP_END_RATIO = 0.35

interface FingerSpec {
  /** 掌指关节在掌面上的横向位置（+X 为食指侧）。 */
  x: number
  /** 掌指关节到腕关节的距离。 */
  knuckle: number
  /** 近节 / 中节 / 末节指骨长度。 */
  segments: readonly [number, number, number]
  /** 近节指骨半径。 */
  radius: number
  /** 横向张开量，正为食指侧。 */
  splay: number
}

/** 食指 → 小指；中指最长、小指最短，掌指关节呈自然弧线。 */
const FINGERS: readonly FingerSpec[] = [
  { x: 0.0255, knuckle: 0.083, segments: [0.037, 0.025, 0.02], radius: 0.0094, splay: 0.1 },
  { x: 0.008, knuckle: 0.089, segments: [0.042, 0.028, 0.021], radius: 0.0098, splay: 0.02 },
  { x: -0.0095, knuckle: 0.086, segments: [0.039, 0.026, 0.02], radius: 0.0092, splay: -0.05 },
  { x: -0.026, knuckle: 0.079, segments: [0.031, 0.021, 0.017], radius: 0.0084, splay: -0.13 },
]

const FINGER_KEYS: readonly FingerKey[] = ['Index', 'Mid', 'Ring', 'Pinky']

/** 掌指 / 近端 / 远端关节的放松弯曲角（弧度），逐节累加。 */
const FINGER_CURL: readonly [number, number, number] = [0.19, 0.3, 0.26]

/** 拇指：掌根偏拇指侧起，向外下方伸出，再收向掌心；只有两节半指骨。 */
const THUMB = {
  base: new THREE.Vector3(0.026, 0.03, 0.006),
  direction: new THREE.Vector3(0.8, 0.56, 0.2),
  segments: [0.031, 0.024, 0.019],
  radius: 0.0118,
  curl: [0.16, 0.26, 0.2],
} as const

/** 四指 5 个关节、拇指 4 个关节（与后端 AppleModel 指关节命名一致）。 */
const FOUR_FINGER_PARTS = ['Start', '_1', '_2', '_3', 'End'] as const
const THUMB_PARTS = ['Start', '_1', '_2', 'End'] as const

interface FingerChainSpec {
  key: FingerKey
  parts: readonly string[]
  /** 默认放松姿态下的关节折线（手部局部坐标，米）。 */
  points: THREE.Vector3[]
  /** 每节指骨的半径，长度 = ``points.length - 1``。 */
  radii: number[]
  /** 默认姿态下近节指骨的方向；观测退化时用它兜底。 */
  initialDirection: THREE.Vector3
}

function ellipsoid(rx: number, ry: number, rz: number, widthSegments = 20): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(
    1,
    widthSegments,
    Math.max(10, Math.round(widthSegments * 0.6)),
  )
  geometry.scale(rx, ry, rz)
  return geometry
}

/** 让方向绕自身与 target 的公垂轴转 angle，即向 target 一侧弯曲。 */
function bendToward(direction: THREE.Vector3, target: THREE.Vector3, angle: number): THREE.Vector3 {
  const axis = new THREE.Vector3().crossVectors(direction, target)
  if (axis.lengthSq() < 1e-8) return direction.clone()
  return direction.clone().applyAxisAngle(axis.normalize(), angle)
}

/** 默认放松手型：按指骨长度逐节累加弯曲，生成局部关节折线。 */
function buildChain(
  key: FingerKey,
  parts: readonly string[],
  start: THREE.Vector3,
  initialDirection: THREE.Vector3,
  segments: readonly number[],
  radius: number,
  curl: readonly number[],
): FingerChainSpec {
  const points = [start.clone()]
  const radii: number[] = []
  const restDirection = initialDirection.clone().normalize()
  let direction = restDirection.clone()
  // End 是后端从末节方向外推出来的指尖点，长度只有末节指骨的一小截。
  const lastLength = segments[segments.length - 1] * TIP_END_RATIO
  for (let index = 0; index < parts.length - 1; index += 1) {
    const length = index < segments.length ? segments[index] : lastLength
    points.push(points[points.length - 1].clone().addScaledVector(direction, length))
    radii.push(radius * (1 - 0.14 * index))
    direction = bendToward(direction, PALM_NORMAL, curl[index] ?? curl[curl.length - 1])
  }
  return { key, parts, points, radii, initialDirection: restDirection }
}

const CHAINS: readonly FingerChainSpec[] = [
  ...FINGERS.map((finger, index) =>
    buildChain(
      FINGER_KEYS[index],
      FOUR_FINGER_PARTS,
      new THREE.Vector3(finger.x, finger.knuckle, 0.004),
      new THREE.Vector3(finger.splay, 1, 0),
      finger.segments,
      finger.radius,
      FINGER_CURL,
    ),
  ),
  buildChain(
    'Thumb',
    THUMB_PARTS,
    THUMB.base,
    THUMB.direction,
    THUMB.segments,
    THUMB.radius,
    THUMB.curl,
  ),
]

interface BoneSpec {
  chain: number
  segment: number
  /** 默认姿态下的指骨长度（米，局部坐标）。 */
  length: number
  radius: number
}

const BONES: readonly BoneSpec[] = CHAINS.flatMap((chain, chainIndex) =>
  chain.radii.map((radius, segment) => ({
    chain: chainIndex,
    segment,
    length: chain.points[segment].distanceTo(chain.points[segment + 1]),
    radius,
  })),
)

// --------------------------------------------------------------------------- //
// 几何缓存：左右手共用同一份顶点，姿态只体现在各自的矩阵上。
// --------------------------------------------------------------------------- //

let palmGeometryCache: THREE.BufferGeometry | null = null

/** 手掌：腕 → 掌根 → 掌指关节 + 大鱼际，四块椭球叠出梯形手型。 */
function buildPalmGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = []

  const wrist = ellipsoid(0.027, 0.032, 0.019)
  wrist.translate(0, 0.01, 0)
  parts.push(wrist)

  // 掌根略窄、掌指关节处最宽，两段椭球叠出梯形而不是一个圆饼
  const heel = ellipsoid(0.035, 0.031, 0.017)
  heel.translate(0, 0.032, 0.001)
  parts.push(heel)

  const knuckles = ellipsoid(0.041, 0.032, 0.0155)
  knuckles.translate(0, 0.064, 0.002)
  parts.push(knuckles)

  // 大鱼际：拇指侧的隆起，手掌才不是一个平板
  const thenar = ellipsoid(0.019, 0.027, 0.014)
  thenar.translate(0.026, 0.029, 0.008)
  parts.push(thenar)

  const merged = mergeGeometries(parts, false)
  for (const part of parts) part.dispose()
  return merged ?? ellipsoid(0.042, 0.085, 0.02)
}

function palmGeometry(): THREE.BufferGeometry {
  if (!palmGeometryCache) palmGeometryCache = buildPalmGeometry()
  return palmGeometryCache
}

let boneGeometryCache: THREE.BufferGeometry[] | null = null

/** 每节指骨一个胶囊：球头卡在关节上，弯曲时接缝不会裂开。 */
function boneGeometries(): THREE.BufferGeometry[] {
  if (!boneGeometryCache) {
    boneGeometryCache = BONES.map(
      (bone) => new THREE.CapsuleGeometry(bone.radius, bone.length, 3, 10),
    )
  }
  return boneGeometryCache
}

let headGeometryCache: THREE.BufferGeometry | null = null

/**
 * 头颅：单个椭球，下半部收窄成下颌、后脑略扁。
 * 单球变形而不是拼接多个球，表面不会有接缝，也不会出现"土豆"式的凸起。
 */
export function headGeometry(): THREE.BufferGeometry {
  if (headGeometryCache) return headGeometryCache
  const geometry = new THREE.SphereGeometry(1, 40, 26)
  const position = geometry.attributes.position as THREE.BufferAttribute
  const vertex = new THREE.Vector3()
  for (let i = 0; i < position.count; i += 1) {
    vertex.fromBufferAttribute(position, i)
    const below = Math.max(0, -vertex.y)
    // 越往下收得越窄 → 下颌线；后脑略扁 → 颅型不会是个正球
    const jaw = 1 - 0.26 * below ** 1.5
    const back = vertex.z < 0 ? 0.95 : 1
    position.setXYZ(
      i,
      vertex.x * HEAD_RADIUS.lateral * jaw,
      vertex.y * HEAD_RADIUS.vertical,
      vertex.z * HEAD_RADIUS.front * jaw * back,
    )
  }
  position.needsUpdate = true
  geometry.computeVertexNormals()
  headGeometryCache = geometry
  return geometry
}

// --------------------------------------------------------------------------- //
// 手部件：手掌 1 个 Mesh + 19 节指骨 Mesh，逐帧写矩阵。
// --------------------------------------------------------------------------- //

export interface HandPart {
  /** 场景根节点，隐藏/显示走它，避免逐个 Mesh 改 visible。 */
  root: THREE.Group
  palm: THREE.Mesh
  /** 与 ``BONES`` 一一对应的指骨。 */
  bones: THREE.Mesh[]
  /** 每根手指当前显示的局部折线（已平滑）；``null`` = 还没拿到过数据。 */
  shapes: (THREE.Vector3[] | null)[]
  /** 每根手指连续缺失指关节的累计时长（秒）。 */
  lost: number[]
  lastUpdate: number
}

/** 建一套手部件；几何是模块级缓存，左右手共用，只有矩阵不同。 */
export function createHandPart(material: THREE.Material): HandPart {
  const root = new THREE.Group()
  const palm = new THREE.Mesh(palmGeometry(), material)
  // 矩阵由本文件按关节直接写入，跳过 position/quaternion 的自动推导。
  palm.matrixAutoUpdate = false
  palm.frustumCulled = false
  root.add(palm)

  const geometries = boneGeometries()
  const bones = BONES.map((_, index) => {
    const mesh = new THREE.Mesh(geometries[index], material)
    mesh.matrixAutoUpdate = false
    mesh.frustumCulled = false
    mesh.visible = false
    root.add(mesh)
    return mesh
  })

  root.visible = false
  return {
    root,
    palm,
    bones,
    shapes: CHAINS.map(() => null),
    lost: CHAINS.map(() => 0),
    lastUpdate: 0,
  }
}

export function setHandVisible(part: HandPart, visible: boolean): void {
  part.root.visible = visible
}

/** 从一帧预览数据里取出五根手指的关节折线（缺任一节即整根视为缺失）。 */
export function readFingerChains(frame: PreviewFrame | null, side: BodySide): FingerChains {
  return CHAINS.map((chain) => {
    const points: THREE.Vector3[] = []
    for (const part of chain.parts) {
      const joint = frame?.joints[`${side}_hand${chain.key}${part}_joint`]
      if (!joint || joint.visibility <= HAND_JOINT_VISIBILITY) return null
      if (![joint.x, joint.y, joint.z].every(Number.isFinite)) return null
      points.push(new THREE.Vector3(joint.x, joint.y, joint.z))
    }
    return points
  })
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

/** 正交化：把 reference 投影到与 axis 垂直的平面上。 */
function perpendicular(reference: THREE.Vector3, axis: THREE.Vector3): THREE.Vector3 | null {
  const result = reference.clone().addScaledVector(axis, -reference.dot(axis))
  return result.lengthSq() > 1e-6 ? result.normalize() : null
}

function applyMatrix(
  mesh: THREE.Mesh,
  xAxis: THREE.Vector3,
  yAxis: THREE.Vector3,
  zAxis: THREE.Vector3,
  origin: THREE.Vector3,
  scale: number,
): void {
  mesh.matrix
    .makeBasis(xAxis, yAxis, zAxis)
    .scale(new THREE.Vector3(scale, scale, scale))
    .setPosition(origin)
  mesh.matrixWorldNeedsUpdate = true
  mesh.visible = true
}

/** 世界 → 手部局部坐标（以腕为原点，按体格缩放归一化）。 */
function toLocal(
  point: THREE.Vector3,
  wrist: THREE.Vector3,
  xAxis: THREE.Vector3,
  yAxis: THREE.Vector3,
  zAxis: THREE.Vector3,
  scale: number,
): THREE.Vector3 {
  const relative = point.clone().sub(wrist)
  return new THREE.Vector3(relative.dot(xAxis), relative.dot(yAxis), relative.dot(zAxis))
    .divideScalar(scale)
}

/** 手部局部坐标 → 世界。 */
function toWorld(
  point: THREE.Vector3,
  wrist: THREE.Vector3,
  xAxis: THREE.Vector3,
  yAxis: THREE.Vector3,
  zAxis: THREE.Vector3,
  scale: number,
): THREE.Vector3 {
  return wrist
    .clone()
    .addScaledVector(xAxis, point.x * scale)
    .addScaledVector(yAxis, point.y * scale)
    .addScaledVector(zAxis, point.z * scale)
}

function blendChains(
  from: readonly THREE.Vector3[],
  to: readonly THREE.Vector3[],
  ratio: number,
): THREE.Vector3[] {
  return from.map((point, index) => point.clone().lerp(to[index], ratio))
}

/**
 * 只用"整根手指跨度"判断这一帧的指关节是否可用。
 *
 * 远景全身视频里单节指骨长度会被 2D 估算严重压缩（实测只有默认值的 0.5~0.8 倍，
 * 帧间还会跳到 0.1 倍），逐节卡容差等于每帧都把手指判成不可信 —— 白膜就会一直
 * 停在默认手型上，看起来"手指完全不跟动作"。所以这里只拦整体塌陷/炸开的坏帧。
 */
function observedSpanOk(chain: FingerChainSpec, local: readonly THREE.Vector3[]): boolean {
  if (local.length !== chain.points.length) return false
  const finite = local.every(
    (point) =>
      Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z),
  )
  if (!finite) return false
  const restSpan = chain.points[0].distanceTo(chain.points[chain.points.length - 1])
  const span = local[0].distanceTo(local[local.length - 1])
  if (restSpan < 1e-6 || !Number.isFinite(span)) return false
  const ratio = span / restSpan
  return ratio >= HAND_SPAN_MIN_RATIO && ratio <= HAND_SPAN_MAX_RATIO
}

/**
 * 把观测到的关节折线重建成"骨长钉死、方向取自观测"的折线。
 *
 * 指关节只有旋转是可以从单目视频里读出来的量，骨长测量在小目标下几乎全是噪声；
 * 与身体的 IK 拟合同一套思路（固定骨长），这样手指既能跟随握拳/张开，又不会被
 * 拉长压扁，也不会因为某一节超出容差就整根手指被冻结。
 */
function fitChain(chain: FingerChainSpec, local: readonly THREE.Vector3[]): THREE.Vector3[] {
  const restStart = chain.points[0]
  // 掌指关节：观测值向默认位置收敛，手指不会从掌面飞出去（Splay 仍保留一半）。
  const start = restStart.clone().lerp(local[0], HAND_KNUCKLE_WEIGHT)
  const result: THREE.Vector3[] = [start]
  for (let index = 1; index < local.length; index += 1) {
    const restLength = chain.points[index].distanceTo(chain.points[index - 1])
    const direction = local[index].clone().sub(local[index - 1])
    if (direction.lengthSq() < 1e-10) {
      direction.copy(chain.points[index]).sub(chain.points[index - 1])
    }
    if (direction.lengthSq() < 1e-10) {
      direction.copy(chain.initialDirection)
    }
    result.push(result[index - 1].clone().addScaledVector(direction.normalize(), restLength))
  }
  return result
}

function placeBone(
  mesh: THREE.Mesh,
  from: THREE.Vector3,
  to: THREE.Vector3,
  restLength: number,
  wrist: THREE.Vector3,
  thumbSide: THREE.Vector3,
  fingers: THREE.Vector3,
  palmNormal: THREE.Vector3,
  front: THREE.Vector3,
  scale: number,
): void {
  const start = toWorld(from, wrist, thumbSide, fingers, palmNormal, scale)
  const end = toWorld(to, wrist, thumbSide, fingers, palmNormal, scale)
  const length = start.distanceTo(end)
  if (!Number.isFinite(length) || length < 1e-6) {
    mesh.visible = false
    return
  }

  const axisY = end.clone().sub(start).normalize()
  const axisZ =
    perpendicular(palmNormal, axisY) ??
    perpendicular(front, axisY) ??
    perpendicular(new THREE.Vector3(1, 0, 0), axisY) ??
    new THREE.Vector3(0, 0, 1)
  const axisX = new THREE.Vector3().crossVectors(axisY, axisZ).normalize()

  // 胶囊的球头按半径外扩，骨长只拉伸中间那段，关节处依然是圆的。
  mesh.matrix
    .makeBasis(axisX, axisY, axisZ)
    .scale(new THREE.Vector3(scale, length / restLength, scale))
    .setPosition(start.lerp(end, 0.5))
  mesh.matrixWorldNeedsUpdate = true
  mesh.visible = true
}

/** 按 head_joint + 躯干朝向摆放头颅。 */
export function updateHeadPart(mesh: THREE.Mesh, joints: JointMap): void {
  const head = jointPoint(joints, 'head_joint')
  const { up, lateral, front } = bodyBasis(joints)
  const scale = estimateBodyScale(joints)
  // head_joint 大致落在耳线/鼻尖上，颅心比它高一点、靠后一点
  const center = head
    .clone()
    .addScaledVector(up, HEAD_LIFT * scale)
    .addScaledVector(front, -0.008 * scale)
  applyMatrix(mesh, lateral.clone().negate(), up, front, center, scale)
}

/**
 * 按手腕位置 + 前臂朝向摆放整只手，再按指关节逐节摆出五指。
 *
 * 掌面朝向取"掌心偏向身体中线"，拇指侧取"朝前"（自然站立/行走时的常态），
 * 因此左右手会自动镜像，五指与掌心弯曲方向都是对的。
 *
 * 指关节降级策略（每根手指独立）：
 * - 有数据：向观测值平滑靠拢（``HAND_SHAPE_SMOOTHING``）；
 * - 缺失 ≤ ``HAND_HOLD_SECONDS``：手型原地冻结，手的根位置仍跟着腕关节走；
 * - 缺失更久：在 ``HAND_RELAX_SECONDS`` 内缓动回默认放松手型，不会突然弹开；
 * - 整段素材都没检出过：始终是默认放松手型。
 */
export function updateHandPart(
  part: HandPart,
  joints: JointMap,
  side: BodySide,
  chains: FingerChains,
  now: number,
): void {
  const wrist = jointPoint(joints, `${side}_hand_joint`)
  const forearm = jointPoint(joints, `${side}_forearm_joint`)
  const elbow = jointPoint(joints, `${side}_arm_joint`)
  const { up, lateral, front } = bodyBasis(joints)
  const scale = estimateBodyScale(joints)

  // +Y：指尖方向。没有腕关节旋转数据，用前臂→手腕最稳。
  const fingers = wrist.clone().sub(forearm)
  if (fingers.lengthSq() < 1e-6) fingers.subVectors(wrist, elbow)
  if (fingers.lengthSq() < 1e-6) fingers.copy(up).negate()
  fingers.normalize()

  // +X：拇指侧（自然姿态朝身体前方），退化时依次退回身体上下轴
  const thumbSide =
    perpendicular(front, fingers) ??
    perpendicular(up, fingers) ??
    perpendicular(lateral, fingers) ??
    new THREE.Vector3(0, 0, 1)

  // 掌面法线：由 (拇指侧, 指尖) 叉乘得到右手系，再用"掌心朝身体中线"矫正正负。
  // 左手该修正为 +1、右手为 -1，负行列式正好把手部几何镜像过去。
  const palmNormal = new THREE.Vector3().crossVectors(thumbSide, fingers).normalize()
  const medial = perpendicular(
    lateral.clone().multiplyScalar(side === 'left' ? -1 : 1),
    fingers,
  )
  if (medial && palmNormal.dot(medial) < 0) palmNormal.negate()

  applyMatrix(part.palm, thumbSide, fingers, palmNormal, wrist, scale)

  const dt = part.lastUpdate > 0 ? clamp((now - part.lastUpdate) / 1000, 0, 0.25) : 0
  part.lastUpdate = now

  CHAINS.forEach((chain, chainIndex) => {
    const measured = chains[chainIndex]
    let shape = part.shapes[chainIndex]
    let usable = false

    if (measured && measured.length === chain.points.length) {
      const local = measured.map((point) =>
        toLocal(point, wrist, thumbSide, fingers, palmNormal, scale),
      )
      if (observedSpanOk(chain, local)) {
        usable = true
        // 观测只提供"朝哪弯"，骨长一律钉回默认值：手指才会跟随动作且不变形。
        const target = fitChain(chain, local)
        shape = shape
          ? fitChain(chain, blendChains(shape, target, HAND_SHAPE_SMOOTHING))
          : target
        part.lost[chainIndex] = 0
      }
    }

    if (!usable) {
      part.lost[chainIndex] += dt
      if (!shape) {
        shape = chain.points.map((point) => point.clone())
      } else if (part.lost[chainIndex] > HAND_HOLD_SECONDS) {
        shape = fitChain(
          chain,
          blendChains(shape, chain.points, 1 - Math.exp(-dt / HAND_RELAX_SECONDS)),
        )
      }
    }

    part.shapes[chainIndex] = shape
  })

  let boneIndex = 0
  CHAINS.forEach((chain, chainIndex) => {
    const shape = part.shapes[chainIndex]
    for (let segment = 0; segment < chain.radii.length; segment += 1) {
      const bone = BONES[boneIndex]
      const mesh = part.bones[boneIndex]
      boneIndex += 1
      if (!shape) continue
      placeBone(
        mesh,
        shape[segment],
        shape[segment + 1],
        bone.length,
        wrist,
        thumbSide,
        fingers,
        palmNormal,
        front,
        scale,
      )
    }
  })
}
