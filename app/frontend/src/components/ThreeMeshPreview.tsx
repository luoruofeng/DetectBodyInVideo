import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { MarchingCubes } from 'three/examples/jsm/objects/MarchingCubes.js'

import type { PreviewFrame } from '../api/analysis'
import {
  createHandPart,
  headGeometry,
  readFingerChains,
  setHandVisible,
  updateHandPart,
  updateHeadPart,
  type FingerChains,
  type HandPart,
} from '../lib/bodyParts'
import {
  buildMeshBalls,
  groundedHipHeightFromJoints,
  resolvePoseJoints,
  type JointMap,
} from '../lib/poseMesh'

interface ThreeMeshPreviewProps {
  frame: PreviewFrame | null
  showGrid: boolean
  theme: 'dark' | 'light'
}

interface SceneRefs {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  group: THREE.Group
  surface: MarchingCubes
  material: THREE.MeshPhysicalMaterial
  partsMaterial: THREE.MeshPhysicalMaterial
  head: THREE.Mesh
  hands: { left: HandPart; right: HandPart }
  smoothed: JointMap | null
  lastFrameTime: number | null
  pending: {
    joints: JointMap
    hipHeight: number
    fingers: { left: FingerChains; right: FingerChains }
  } | null
  lastBuild: number
  fitted: boolean
  frameId: number
}

// 56 的分辨率会让手臂和躯干糊成一团；96 才能在 1.9 m 身高内画出可辨认的人形轮廓。
const RESOLUTION = 96
const MAX_POLY_COUNT = 200_000
const SMOOTHING = 0.55
// 逐帧重建隐式曲面的上限频率，避免播放时把主线程占满。
const BUILD_INTERVAL_MS = 45

// 自定义标量场：单球在 d²=1 处取到 1，等值面 1 正好对应球的标称半径。
// 重叠球用“平滑取大”而不是线性相加——否则胸口的六七个球会把躯干吹成粗圆柱，
// 手臂也被膨胀的侧壁吞进去。平滑取大只让接缝融合，不会放大整体轮廓。
const FIELD_ISOLATION = 1
const FIELD_FALLOFF = 2.2
const FIELD_CUTOFF = 2.6
const FIELD_CUTOFF_RADIUS = Math.sqrt(FIELD_CUTOFF)
const FIELD_BLEND = 0.28
const FIELD_BLEND_FLOOR = FIELD_ISOLATION * 0.25

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

/**
 * 脚底贴地所需的髋高。优先用后端平滑过的 `hipHeight`（原地动作不会上下抖），
 * 只有推理进行中 / 该帧没有着地参考时才退回"当前帧最低脚"。
 */
function resolveHipHeight(frame: PreviewFrame, joints: JointMap): number {
  const backend = frame.hipHeight
  if (typeof backend === 'number' && Number.isFinite(backend) && backend > 0) {
    return clamp(backend, 0.45, 1.25)
  }
  return groundedHipHeightFromJoints(joints)
}

function smoothUnion(a: number, b: number): number {
  const high = Math.max(a, b)
  const low = Math.min(a, b)
  if (low < FIELD_BLEND_FLOOR || high - low >= FIELD_BLEND) return high
  const h = FIELD_BLEND - (high - low)
  return high + (h * h) / (4 * FIELD_BLEND)
}

function addFieldBall(
  surface: MarchingCubes,
  point: THREE.Vector3,
  radius: number,
  colorHex: number,
  colors: Map<number, THREE.Color>,
): void {
  if (![point.x, point.y, point.z, radius].every(Number.isFinite) || radius <= 0) return

  const field = surface.field
  const palette = surface.palette
  const size = surface.size
  // field 坐标与关节坐标同为 [-1,1] 的世界坐标：等值面正好落在 d = radius 上。
  // 之前多除了一次 2，导致四肢只有标称粗细的一半，头/手（解析几何）却按原尺寸画，比例失衡。
  const invRadiusSquared = 1 / (radius * radius)
  const extent = radius * FIELD_CUTOFF_RADIUS
  const minX = clamp(Math.floor((point.x - extent + 1) * size * 0.5), 0, size - 1)
  const maxX = clamp(Math.ceil((point.x + extent + 1) * size * 0.5), 0, size - 1)
  const minY = clamp(Math.floor((point.y - extent + 1) * size * 0.5), 0, size - 1)
  const maxY = clamp(Math.ceil((point.y + extent + 1) * size * 0.5), 0, size - 1)
  const minZ = clamp(Math.floor((point.z - extent + 1) * size * 0.5), 0, size - 1)
  const maxZ = clamp(Math.ceil((point.z + extent + 1) * size * 0.5), 0, size - 1)

  let color = colors.get(colorHex)
  if (!color) {
    color = new THREE.Color(colorHex)
    colors.set(colorHex, color)
  }
  const red = Math.min(1, color.r)
  const green = Math.min(1, color.g)
  const blue = Math.min(1, color.b)

  for (let z = minZ; z <= maxZ; z += 1) {
    const dz = (z * 2) / size - 1 - point.z
    for (let y = minY; y <= maxY; y += 1) {
      const dy = (y * 2) / size - 1 - point.y
      for (let x = minX; x <= maxX; x += 1) {
        const dx = (x * 2) / size - 1 - point.x
        const distanceSquared = (dx * dx + dy * dy + dz * dz) * invRadiusSquared
        if (distanceSquared >= FIELD_CUTOFF) continue

        const value = Math.exp(FIELD_FALLOFF * (1 - distanceSquared))
        const index = (z * size + y) * size + x
        const previous = field[index]
        const next = smoothUnion(previous, value)
        if (next <= previous) continue

        field[index] = next
        const weight = clamp(value / FIELD_ISOLATION, 0, 1)
        const paletteIndex = index * 3
        palette[paletteIndex] = red * weight
        palette[paletteIndex + 1] = green * weight
        palette[paletteIndex + 2] = blue * weight
      }
    }
  }
}

function rebuildSurface(surface: MarchingCubes, joints: JointMap): void {
  const colors = new Map<number, THREE.Color>()
  surface.reset()
  for (const ball of buildMeshBalls(joints)) {
    addFieldBall(surface, ball.position, ball.radius, ball.color, colors)
  }
  surface.update()
}

/** 头颅与双手是解析几何：不重建曲面，但手指要逐帧按指关节摆放。 */
function updateParts(
  refs: SceneRefs,
  joints: JointMap,
  fingers: { left: FingerChains; right: FingerChains },
): void {
  updateHeadPart(refs.head, joints)
  const now = performance.now()
  updateHandPart(refs.hands.left, joints, 'left', fingers.left, now)
  updateHandPart(refs.hands.right, joints, 'right', fingers.right, now)
}

function setPartsVisible(refs: SceneRefs, visible: boolean): void {
  refs.head.visible = visible
  setHandVisible(refs.hands.left, visible)
  setHandVisible(refs.hands.right, visible)
}

/** 首次出膜时把相机拉到刚好框住人体的距离，别让人形缩在角落里看不清。 */
function fitCameraToJoints(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  joints: JointMap,
  hipHeight: number,
): void {
  const box = new THREE.Box3()
  joints.forEach((point) => {
    if ([point.x, point.y, point.z].every(Number.isFinite)) box.expandByPoint(point)
  })
  if (box.isEmpty()) return
  // 关节只是骨架，白膜还有一圈体积，外扩后再算取景。
  box.expandByScalar(0.14)

  const center = box.getCenter(new THREE.Vector3())
  center.y += hipHeight
  const radius = Math.max(0.4, box.getSize(new THREE.Vector3()).length() * 0.5)
  const verticalFov = THREE.MathUtils.degToRad(camera.fov)
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(0.5, camera.aspect))
  const distance =
    Math.max(radius / Math.sin(verticalFov / 2), radius / Math.sin(horizontalFov / 2)) * 1.12

  // 保留用户当前的观察方向，只调整距离和焦点。
  const direction = camera.position.clone().sub(controls.target)
  if (direction.lengthSq() < 1e-6) direction.set(0.45, 0.35, 1)
  direction.normalize()

  controls.target.copy(center)
  camera.position
    .copy(center)
    .addScaledVector(direction, clamp(distance, controls.minDistance, controls.maxDistance))
  controls.update()
}

/** 关键点 → metaball 隐式曲面 → Three.js 白膜 Mesh。 */
export function ThreeMeshPreview({ frame, showGrid, theme }: ThreeMeshPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<SceneRefs | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '当前环境不支持 WebGL')
      return
    }

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100)
    camera.position.set(1.25, 1.25, 2.75)

    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
    renderer.setSize(container.clientWidth || 320, container.clientHeight || 240, false)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0.9, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1.2
    controls.maxDistance = 7
    controls.update()

    scene.add(new THREE.HemisphereLight(0xffffff, 0x1e293b, 1.8))
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.35)
    keyLight.position.set(2, 4, 3)
    scene.add(keyLight)
    const fillLight = new THREE.DirectionalLight(0xbae6fd, 0.65)
    fillLight.position.set(-2.5, 2, 1.5)
    scene.add(fillLight)
    const rimLight = new THREE.DirectionalLight(0xfbcfe8, 0.55)
    rimLight.position.set(0, 2.5, -3)
    scene.add(rimLight)

    const grid = new THREE.GridHelper(3, 15, 0x22d3ee, 0x334155)
    grid.name = 'mesh-grid'
    grid.visible = showGrid
    scene.add(grid)

    const material = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.46,
      metalness: 0,
      clearcoat: 0.2,
      clearcoatRoughness: 0.6,
      // 不透明：半透明会露出曲面内部，人形轮廓反而更糊。
      side: THREE.DoubleSide,
    })
    const surface = new MarchingCubes(RESOLUTION, material, false, true, MAX_POLY_COUNT)
    surface.isolation = FIELD_ISOLATION
    surface.frustumCulled = false
    surface.visible = false

    // 头颅 / 双手用解析几何：材质与白膜一致，但没有顶点色（纯白）。
    const partsMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xf8fbff,
      roughness: 0.5,
      metalness: 0,
      clearcoat: 0.16,
      clearcoatRoughness: 0.62,
      side: THREE.DoubleSide,
    })
    const makePart = (geometry: THREE.BufferGeometry) => {
      const mesh = new THREE.Mesh(geometry, partsMaterial)
      // 矩阵由 bodyParts 按关节直接写入，跳过 position/quaternion 的自动推导。
      mesh.matrixAutoUpdate = false
      mesh.frustumCulled = false
      mesh.visible = false
      return mesh
    }
    const head = makePart(headGeometry())
    const hands = {
      left: createHandPart(partsMaterial),
      right: createHandPart(partsMaterial),
    }

    const group = new THREE.Group()
    group.add(surface, head, hands.left.root, hands.right.root)
    scene.add(group)

    const refs: SceneRefs = {
      renderer,
      scene,
      camera,
      controls,
      group,
      surface,
      material,
      partsMaterial,
      head,
      hands,
      smoothed: null,
      lastFrameTime: null,
      pending: null,
      lastBuild: 0,
      fitted: false,
      frameId: 0,
    }
    sceneRef.current = refs

    const resize = () => {
      const width = container.clientWidth || 320
      const height = container.clientHeight || 240
      renderer.setSize(width, height, false)
      camera.aspect = width / Math.max(1, height)
      camera.updateProjectionMatrix()
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(container)

    const animate = () => {
      refs.frameId = requestAnimationFrame(animate)
      controls.update()

      // 重建限制在 BUILD_INTERVAL_MS 一次，播放时始终用最新的姿态。
      const pending = refs.pending
      if (pending && performance.now() - refs.lastBuild >= BUILD_INTERVAL_MS) {
        refs.pending = null
        refs.lastBuild = performance.now()
        refs.group.position.y = pending.hipHeight
        rebuildSurface(refs.surface, pending.joints)
        updateParts(refs, pending.joints, pending.fingers)
        refs.surface.visible = true
        if (!refs.fitted) {
          fitCameraToJoints(camera, controls, pending.joints, pending.hipHeight)
          refs.fitted = true
        }
      }

      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(refs.frameId)
      observer.disconnect()
      controls.dispose()
      surface.geometry.dispose()
      material.dispose()
      partsMaterial.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      sceneRef.current = null
    }
    // 场景只创建一次；外部状态通过下面的 effect 增量更新。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const grid = sceneRef.current?.scene.getObjectByName('mesh-grid')
    if (grid) grid.visible = showGrid
  }, [showGrid])

  useEffect(() => {
    const refs = sceneRef.current
    if (!refs) return
    const emissive = theme === 'light' ? 0x111827 : 0x0b1220
    const intensity = theme === 'light' ? 0.04 : 0.08
    for (const material of [refs.material, refs.partsMaterial]) {
      material.emissive.set(emissive)
      material.emissiveIntensity = intensity
    }
  }, [theme])

  useEffect(() => {
    const refs = sceneRef.current
    if (!refs) return

    if (!frame) {
      refs.surface.visible = false
      setPartsVisible(refs, false)
      refs.smoothed = null
      refs.lastFrameTime = null
      refs.pending = null
      refs.fitted = false
      return
    }

    const target = resolvePoseJoints(frame)
    const shouldReset =
      refs.lastFrameTime === null || Math.abs(frame.t - refs.lastFrameTime) > 0.2

    if (!refs.smoothed || shouldReset) {
      refs.smoothed = new Map(
        Array.from(target.entries(), ([name, point]) => [name, point.clone()] as const),
      )
    } else {
      target.forEach((point, name) => {
        const previous = refs.smoothed?.get(name)
        if (previous) previous.lerp(point, SMOOTHING)
        else refs.smoothed?.set(name, point.clone())
      })
    }

    refs.lastFrameTime = frame.t
    refs.pending = {
      joints: refs.smoothed,
      hipHeight: resolveHipHeight(frame, refs.smoothed),
      fingers: {
        left: readFingerChains(frame, 'left'),
        right: readFingerChains(frame, 'right'),
      },
    }
  }, [frame])

  return (
    <div className="relative h-64 w-full overflow-hidden rounded-xl border border-line bg-[radial-gradient(120%_120%_at_50%_0%,rgba(244,114,182,0.10),transparent_60%)]">
      <div ref={containerRef} className="absolute inset-0" />
      {error ? (
        <div className="absolute inset-0 grid place-items-center px-4 text-center text-2xs text-danger">
          Mesh 预览不可用：{error}
        </div>
      ) : null}
      {!frame && !error ? (
        <div className="absolute inset-0 grid place-items-center text-2xs text-slate-500">
          完成分析后显示白膜 Mesh
        </div>
      ) : null}
      {frame && !frame.valid && !error ? (
        <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 rounded-md bg-warn/15 px-2 py-0.5 text-2xs text-warn">
          当前帧关键点可信度低
        </div>
      ) : null}
      <div className="pointer-events-none absolute bottom-2 left-3 flex gap-2 text-2xs text-slate-500">
        <span>关键点 → 隐式曲面</span>
        <span>·</span>
        <span>白色为近似白膜</span>
      </div>
      <div className="pointer-events-none absolute right-2 top-2 rounded-md bg-black/30 px-1.5 py-0.5 text-2xs text-slate-400">
        拖拽旋转 · 滚轮缩放
      </div>
    </div>
  )
}
