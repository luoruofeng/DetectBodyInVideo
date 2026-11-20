import { useCallback, useEffect, useRef, useState } from 'react'

import type { PreviewFrame } from '../api/analysis'
import type { LoopRange } from '../hooks/usePlayback'
import { formatTime } from '../lib/pose'

interface TimelineProps {
  duration: number
  currentTime: number
  frames: PreviewFrame[]
  loop: LoopRange | null
  onSeek: (time: number) => void
  onSetLoopStart: () => void
  onSetLoopEnd: () => void
  onClearLoop: () => void
}

/** 底部时间轴：可拖拽 seek，红色刻度标记缺失帧，支持 A/B 循环区间。 */
export function Timeline({
  duration,
  currentTime,
  frames,
  loop,
  onSeek,
  onSetLoopStart,
  onSetLoopEnd,
  onClearLoop,
}: TimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const dragging = useRef(false)

  useEffect(() => {
    const canvas = canvasRef.current
    const track = trackRef.current
    if (!canvas || !track) return

    const draw = () => {
      const width = track.clientWidth
      const height = track.clientHeight
      if (width <= 0 || height <= 0) return
      const dpr = Math.min(2.5, window.devicePixelRatio || 1)
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      const context = canvas.getContext('2d')
      if (!context) return
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      context.clearRect(0, 0, width, height)

      const ratio = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0
      const progressX = ratio * width

      // 轨道底
      context.fillStyle = 'rgba(148, 163, 184, 0.16)'
      context.fillRect(0, height / 2 - 3, width, 6)

      // 循环区间
      if (loop && duration > 0) {
        const start = (Math.max(0, loop.a) / duration) * width
        const end = (Math.min(duration, loop.b) / duration) * width
        context.fillStyle = 'rgba(34, 211, 238, 0.12)'
        context.fillRect(start, 0, Math.max(1, end - start), height)
      }

      // 进度
      context.fillStyle = 'rgba(34, 211, 238, 0.85)'
      context.fillRect(0, height / 2 - 3, Math.max(2, progressX), 6)

      // 缺失帧刻度
      if (duration > 0 && frames.length) {
        context.fillStyle = 'rgba(248, 113, 113, 0.7)'
        const barHeight = Math.max(6, height * 0.5)
        for (const frame of frames) {
          if (frame.valid) continue
          const x = Math.round((frame.t / duration) * width)
          context.fillRect(x, height / 2 - barHeight / 2, 1, barHeight)
        }
      }

      // 播放头
      context.fillStyle = '#e2e8f0'
      context.fillRect(progressX - 1, 2, 2, height - 4)
      context.beginPath()
      context.arc(progressX, height / 2, 5, 0, Math.PI * 2)
      context.fill()
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(track)
    return () => observer.disconnect()
  }, [currentTime, duration, frames, loop])

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const track = trackRef.current
      if (!track || duration <= 0) return
      const rect = track.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      onSeek(ratio * duration)
    },
    [duration, onSeek],
  )

  const missingCount = frames.reduce((total, frame) => (frame.valid ? total : total + 1), 0)

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-2xs text-slate-500">
        <div className="flex items-center gap-3 font-mono">
          <span className="text-slate-300">{formatTime(currentTime)}</span>
          <span>/</span>
          <span>{formatTime(duration)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={missingCount ? 'text-danger' : 'text-slate-500'}>
            缺失帧 {missingCount}
          </span>
          {loop ? (
            <span className="chip border-accent/40 bg-accent/10 text-accent">
              循环 {formatTime(loop.a)} – {formatTime(loop.b)}
            </span>
          ) : null}
        </div>
      </div>

      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="时间轴"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration * 100) / 100}
        aria-valuenow={Math.round(currentTime * 100) / 100}
        aria-valuetext={formatTime(currentTime)}
        className="relative h-11 w-full cursor-pointer touch-none rounded-lg bg-white/[0.02] ring-1 ring-inset ring-white/5"
        onPointerDown={(event) => {
          dragging.current = true
          event.currentTarget.setPointerCapture(event.pointerId)
          seekFromEvent(event.clientX)
        }}
        onPointerMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect()
          const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width))
          setHoverTime(ratio * duration)
          if (dragging.current) seekFromEvent(event.clientX)
        }}
        onPointerUp={() => {
          dragging.current = false
        }}
        onPointerCancel={() => {
          dragging.current = false
        }}
        onPointerLeave={() => setHoverTime(null)}
        onKeyDown={(event) => {
          const step = event.shiftKey ? 5 : 1
          if (event.key === 'ArrowLeft') {
            event.preventDefault()
            onSeek(currentTime - step)
          } else if (event.key === 'ArrowRight') {
            event.preventDefault()
            onSeek(currentTime + step)
          } else if (event.key === 'Home') {
            event.preventDefault()
            onSeek(0)
          } else if (event.key === 'End') {
            event.preventDefault()
            onSeek(duration)
          }
        }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden />
        {hoverTime !== null && duration > 0 ? (
          <span
            className="pointer-events-none absolute -top-7 -translate-x-1/2 rounded-md bg-panel/95 px-2 py-0.5 text-2xs font-mono text-slate-200 shadow-panel"
            style={{ left: `${Math.min(100, (hoverTime / duration) * 100)}%` }}
          >
            {formatTime(hoverTime)}
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn-ghost px-2.5 py-1 text-2xs" onClick={onSetLoopStart}>
          标记循环起点 (A)
        </button>
        <button type="button" className="btn-ghost px-2.5 py-1 text-2xs" onClick={onSetLoopEnd}>
          标记循环终点 (B)
        </button>
        {loop ? (
          <button type="button" className="btn-ghost px-2.5 py-1 text-2xs" onClick={onClearLoop}>
            清除循环
          </button>
        ) : null}
        <span className="ml-auto text-2xs text-slate-600">
          拖拽时间轴或按 ← / → 逐帧，Shift 加速
        </span>
      </div>
    </div>
  )
}
