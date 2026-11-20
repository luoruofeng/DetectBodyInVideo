import { useCallback, useEffect, useRef, useState } from 'react'

import type { PreviewFrame } from '../api/analysis'
import {
  LANDMARK_LABELS_ZH,
  LANDMARK_NAMES,
  POSE_CONNECTIONS,
  pickLandmark,
  type HoveredLandmark,
} from '../lib/pose'

export type LineStyle = 'solid' | 'thin' | 'dotted'

interface SkeletonOverlayProps {
  frame: PreviewFrame | null
  showSkeleton: boolean
  showIndices: boolean
  showBox: boolean
  lineStyle: LineStyle
}

const MIN_VISIBILITY = 0.05

function visibilityColor(visibility: number): string {
  if (visibility >= 0.75) return 'rgba(34, 211, 238, 0.95)'
  if (visibility >= 0.45) return 'rgba(34, 211, 238, 0.7)'
  return 'rgba(148, 163, 184, 0.7)'
}

/** 2D 骨架叠加：Canvas 逐帧绘制 33 点 + 连线，支持编号与悬停查询。 */
export function SkeletonOverlay({
  frame,
  showSkeleton,
  showIndices,
  showBox,
  lineStyle,
}: SkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hovered, setHovered] = useState<HoveredLandmark | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const draw = () => {
      const width = parent.clientWidth
      const height = parent.clientHeight
      const dpr = Math.min(2.5, window.devicePixelRatio || 1)
      if (width <= 0 || height <= 0) return
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr)
        canvas.height = Math.round(height * dpr)
      }
      const context = canvas.getContext('2d')
      if (!context) return
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      context.clearRect(0, 0, width, height)

      if (!showSkeleton || !frame) return
      const points = frame.landmarks
      if (!points.length) return

      const visible = (index: number) =>
        Boolean(points[index]) && points[index].visibility >= MIN_VISIBILITY

      if (showBox && frame.bbox) {
        context.save()
        context.strokeStyle = 'rgba(34, 211, 238, 0.45)'
        context.lineWidth = 1
        context.setLineDash([6, 4])
        context.strokeRect(
          frame.bbox.x * width,
          frame.bbox.y * height,
          frame.bbox.width * width,
          frame.bbox.height * height,
        )
        context.restore()
      }

      context.save()
      context.strokeStyle = 'rgba(34, 211, 238, 0.85)'
      context.lineWidth = lineStyle === 'thin' ? 1.2 : 2.2
      context.lineCap = 'round'
      context.shadowColor = 'rgba(34, 211, 238, 0.45)'
      context.shadowBlur = lineStyle === 'thin' ? 0 : 6
      context.setLineDash(lineStyle === 'dotted' ? [3, 5] : [])
      for (const [from, to] of POSE_CONNECTIONS) {
        if (!visible(from) || !visible(to)) continue
        context.beginPath()
        context.moveTo(points[from].x * width, points[from].y * height)
        context.lineTo(points[to].x * width, points[to].y * height)
        context.stroke()
      }
      context.restore()

      points.forEach((point, index) => {
        if (point.visibility < MIN_VISIBILITY) return
        const x = point.x * width
        const y = point.y * height
        const isHovered = hovered?.index === index
        context.beginPath()
        context.arc(x, y, isHovered ? 5.5 : 3.4, 0, Math.PI * 2)
        context.fillStyle = isHovered ? '#ffffff' : visibilityColor(point.visibility)
        context.fill()
        if (isHovered) {
          context.beginPath()
          context.arc(x, y, 9, 0, Math.PI * 2)
          context.strokeStyle = 'rgba(34, 211, 238, 0.9)'
          context.lineWidth = 1.5
          context.stroke()
        }
        if (showIndices) {
          context.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'
          context.fillStyle = 'rgba(226, 232, 240, 0.85)'
          context.fillText(String(index), x + 6, y - 6)
        }
      })

      if (!frame.valid) {
        context.fillStyle = 'rgba(248, 113, 113, 0.9)'
        context.font = '12px system-ui, sans-serif'
        context.fillText('本帧无有效姿态（valid=false）', 12, height - 14)
      }
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(parent)
    return () => observer.disconnect()
  }, [frame, hovered, lineStyle, showBox, showIndices, showSkeleton])

  const handleMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      setHovered(pickLandmark(frame, rect.width, rect.height, x, y, 14))
    },
    [frame],
  )

  const hoveredPoint = hovered && frame ? frame.landmarks[hovered.index] : null

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full cursor-crosshair"
        onMouseMove={handleMove}
        onMouseLeave={() => setHovered(null)}
        aria-hidden
      />
      {hovered && hoveredPoint ? (
        <div
          className="pointer-events-none absolute z-20 rounded-lg border border-line bg-panel/95 px-2.5 py-1.5 text-2xs text-slate-200 shadow-panel backdrop-blur"
          style={{
            left: `${hovered.px}px`,
            top: `${hovered.py}px`,
            transform:
              hovered.py < 64
                ? 'translate(-50%, 12px)'
                : 'translate(-50%, calc(-100% - 12px))',
          }}
        >
          <div className="font-medium text-slate-100">
            #{hovered.index} {LANDMARK_LABELS_ZH[LANDMARK_NAMES[hovered.index]] ?? LANDMARK_NAMES[hovered.index]}
          </div>
          <div className="mt-0.5 font-mono text-slate-400">
            x {hoveredPoint.x.toFixed(3)} · y {hoveredPoint.y.toFixed(3)} · z{' '}
            {hoveredPoint.z.toFixed(3)}
          </div>
          <div className="font-mono text-slate-400">
            置信度 {hoveredPoint.visibility.toFixed(2)}
          </div>
        </div>
      ) : null}
    </>
  )
}
