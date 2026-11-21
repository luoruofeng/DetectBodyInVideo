import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import type { PreviewFrame } from '../api/analysis'
import { ALL_APPLE_BONES } from '../lib/pose'

interface ThreePreviewProps {
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
  bones: THREE.LineSegments
  joints: Map<string, THREE.Mesh>
  jointGeometry: THREE.SphereGeometry
  jointMaterial: THREE.MeshBasicMaterial
  frameId: number
}

const JOINT_RADIUS = 0.018
const ACCENT = 0x22d3ee
const HIP_HEIGHT = 0.9
const FOOT_JOINTS = [
  'left_foot_joint',
  'right_foot_joint',
  'left_toes_joint',
  'right_toes_joint',
]
const MIN_FOOT_SPAN = 0.25
const MAX_FOOT_SPAN = 1.3

/**
 * 关节是髋局部坐标，髋本身恒在原点。直接渲染会让深蹲时脚踝随膝弯曲上移，
 * 看起来像双脚离地，所以要把髋放到"脚底高度"上。
 *
 * 优先用后端给的 `hipHeight`：它对支撑脚跨度做过零相位平滑，原地动作时不会
 * 因为逐帧关键点噪声让整个身体上下抖。只有推理进行中（没有后处理结果）才退回
 * "取当前帧最低脚"的旧算法 —— 那条路径会把噪声原样画出来。
 */
function groundedHipHeight(frame: PreviewFrame | null): number {
  if (!frame) return HIP_HEIGHT
  const backend = frame.hipHeight
  if (
    typeof backend === 'number' &&
    Number.isFinite(backend) &&
    backend >= MIN_FOOT_SPAN &&
    backend <= MAX_FOOT_SPAN
  ) {
    return backend
  }
  let lowest = Infinity
  for (const name of FOOT_JOINTS) {
    const joint = frame.joints[name]
    if (!joint || joint.visibility <= 0) continue
    lowest = Math.min(lowest, joint.y)
  }
  if (!Number.isFinite(lowest)) return HIP_HEIGHT
  const span = -lowest
  if (span < MIN_FOOT_SPAN || span > MAX_FOOT_SPAN) return HIP_HEIGHT
  return span
}

/** three.js 骨骼预览：按米制关节渲染球棍骨架，跟随播放头逐帧更新。 */
export function ThreePreview({ frame, showGrid, theme }: ThreePreviewProps) {
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
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(1.35, 1.25, 2.35)

    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
    renderer.setSize(container.clientWidth || 320, container.clientHeight || 240, false)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, HIP_HEIGHT, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 0.8
    controls.maxDistance = 8
    controls.update()

    scene.add(new THREE.HemisphereLight(0xe2e8f0, 0x1e293b, 1.6))
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2)
    keyLight.position.set(2, 4, 3)
    scene.add(keyLight)

    const grid = new THREE.GridHelper(3, 15, 0x22d3ee, 0x334155)
    grid.name = 'grid'
    scene.add(grid)
    scene.add(new THREE.AxesHelper(0.5))

    const group = new THREE.Group()
    group.position.y = HIP_HEIGHT
    scene.add(group)

    const bonesGeometry = new THREE.BufferGeometry()
    bonesGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(ALL_APPLE_BONES.length * 2 * 3), 3),
    )
    const bones = new THREE.LineSegments(
      bonesGeometry,
      new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.9 }),
    )
    bones.frustumCulled = false
    group.add(bones)

    const refs: SceneRefs = {
      renderer,
      scene,
      camera,
      controls,
      group,
      bones,
      joints: new Map(),
      jointGeometry: new THREE.SphereGeometry(JOINT_RADIUS, 12, 12),
      jointMaterial: new THREE.MeshBasicMaterial({ color: 0xe2e8f0 }),
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
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(refs.frameId)
      observer.disconnect()
      controls.dispose()
      bonesGeometry.dispose()
      ;(bones.material as THREE.Material).dispose()
      refs.jointGeometry.dispose()
      refs.jointMaterial.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      sceneRef.current = null
    }
  }, [])

  useEffect(() => {
    const grid = sceneRef.current?.scene.getObjectByName('grid')
    if (grid) grid.visible = showGrid
  }, [showGrid])

  // 浅色主题下把骨架换成深色，保证对比度（任务 8.6）
  useEffect(() => {
    const refs = sceneRef.current
    if (!refs) return
    const light = theme === 'light'
    ;(refs.bones.material as THREE.LineBasicMaterial).color.set(light ? 0x0e7490 : 0x22d3ee)
    refs.jointMaterial.color.set(light ? 0x0f172a : 0xe2e8f0)
  }, [theme])

  useEffect(() => {
    const refs = sceneRef.current
    if (!refs) return

    // 每帧按着地脚重新定位髋高度：深蹲时髋下沉，脚不再悬空。
    refs.group.position.y = groundedHipHeight(frame)

    const position = refs.bones.geometry.getAttribute('position') as THREE.BufferAttribute
    const array = position.array as Float32Array
    ALL_APPLE_BONES.forEach(([from, to], boneIndex) => {
      const a = frame?.joints[from]
      const b = frame?.joints[to]
      const offset = boneIndex * 6
      if (!a || !b || a.visibility <= 0 || b.visibility <= 0) {
        array.fill(0, offset, offset + 6)
        return
      }
      array[offset] = a.x
      array[offset + 1] = a.y
      array[offset + 2] = a.z
      array[offset + 3] = b.x
      array[offset + 4] = b.y
      array[offset + 5] = b.z
    })
    position.needsUpdate = true
    refs.bones.geometry.computeBoundingSphere()

    // 关节球按需创建（同一个 name 复用），避免首帧 jointNames 为空导致骨架丢失
    refs.joints.forEach((mesh) => {
      mesh.visible = false
    })
    if (frame) {
      for (const [name, joint] of Object.entries(frame.joints)) {
        if (joint.visibility <= 0) continue
        let mesh = refs.joints.get(name)
        if (!mesh) {
          mesh = new THREE.Mesh(refs.jointGeometry, refs.jointMaterial)
          refs.joints.set(name, mesh)
          refs.group.add(mesh)
        }
        mesh.visible = true
        mesh.position.set(joint.x, joint.y, joint.z)
        mesh.scale.setScalar(0.7 + Math.min(1, Math.max(0, joint.visibility)) * 0.4)
      }
    }
  }, [frame])

  return (
    <div className="relative h-56 w-full overflow-hidden rounded-xl border border-line bg-[radial-gradient(120%_120%_at_50%_0%,rgba(34,211,238,0.10),transparent_60%)]">
      <div ref={containerRef} className="absolute inset-0" />
      {error ? (
        <div className="absolute inset-0 grid place-items-center px-4 text-center text-2xs text-danger">
          3D 预览不可用：{error}
        </div>
      ) : null}
      {!frame && !error ? (
        <div className="absolute inset-0 grid place-items-center text-2xs text-slate-500">
          完成分析后显示 3D 骨架
        </div>
      ) : null}
      <div className="pointer-events-none absolute bottom-2 left-3 flex gap-2 text-2xs text-slate-500">
        <span>y 向上</span>
        <span>·</span>
        <span>-z 向前</span>
        <span>·</span>
        <span>单位米</span>
        <span>·</span>
        <span>脚部着地</span>
      </div>
      <div className="pointer-events-none absolute right-2 top-2 rounded-md bg-black/30 px-1.5 py-0.5 text-2xs text-slate-400">
        拖拽旋转 · 滚轮缩放
      </div>
    </div>
  )
}
