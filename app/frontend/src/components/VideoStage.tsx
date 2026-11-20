import { useEffect, useRef, useState, type RefObject } from 'react'

import type { PreviewFrame } from '../api/analysis'
import type { VideoMetadata } from '../api/videos'
import type { UploadPhase } from '../hooks/useVideoUpload'
import { formatBytes } from '../lib/format'
import { ProgressBar } from './ProgressBar'
import { AlertIcon, FilmIcon, UploadIcon } from './icons'
import { SkeletonOverlay, type LineStyle } from './SkeletonOverlay'

const SUPPORTED_FORMATS = ['MP4', 'MOV', 'AVI', 'MKV', 'H.264 / H.265']

function EmptyState({ onPick }: { onPick: () => void }) {
  return (
    <div className="on-video flex max-w-lg flex-col items-center gap-5 px-6 text-center">
      <div className="relative grid h-20 w-20 place-items-center rounded-2xl bg-gradient-to-br from-accent/20 to-transparent ring-1 ring-inset ring-accent/25">
        <FilmIcon className="h-8 w-8 text-accent" />
      </div>
      <div className="space-y-2">
        <h2 className="text-lg font-medium text-slate-100">拖入一段含人物的视频开始</h2>
        <p className="text-sm leading-relaxed text-slate-500">
          逐帧识别人体关键点、计算关节角度，导出可直接驱动 USDZ 模型的 JSON。
        </p>
      </div>
      <button type="button" className="btn-primary px-5 py-2.5" onClick={onPick}>
        <UploadIcon className="h-4 w-4" />
        选择视频文件
      </button>
      <div className="flex flex-wrap justify-center gap-1.5">
        {SUPPORTED_FORMATS.map((format) => (
          <span key={format} className="chip border-line bg-white/[0.03] text-slate-500">
            {format}
          </span>
        ))}
      </div>
      <p className="text-2xs text-slate-600">
        快捷键：空格 播放/暂停 · ← → 逐帧 · U 打开文件 · E 导出
      </p>
    </div>
  )
}

function ProgressState({ phase, progress }: { phase: UploadPhase; progress: number }) {
  const percent = Math.round(progress * 100)
  const label = phase === 'processing' ? '正在解析元信息…' : '正在上传视频…'
  return (
    <div className="w-[min(420px,80%)] space-y-3 rounded-2xl border border-line bg-panel/80 px-6 py-6 shadow-panel">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-accent">{percent}%</span>
      </div>
      <ProgressBar value={progress} label={label} />
      <p className="text-2xs text-slate-500">大文件会自动分块上传，本地处理不会消耗流量</p>
    </div>
  )
}

interface OverlayToggles {
  skeleton: boolean
  indices: boolean
  box: boolean
  lineStyle: LineStyle
  onSkeleton: (value: boolean) => void
  onIndices: (value: boolean) => void
  onBox: (value: boolean) => void
  onLineStyle: (value: LineStyle) => void
}

interface VideoStageProps {
  video: VideoMetadata | null
  previewUrl: string | null
  phase: UploadPhase
  progress: number
  error: string | null
  onPick: () => void
  videoRef: RefObject<HTMLVideoElement>
  frame: PreviewFrame | null
  overlay: OverlayToggles
  analyzing: boolean
  /** 识别还没跑到播放位置：此时不画骨架，提示用户往回收一点 */
  playheadAhead: boolean
  framesLoading: boolean
  framesLoaded: number
  totalPreviewFrames: number
}

export function VideoStage({
  video,
  previewUrl,
  phase,
  progress,
  error,
  onPick,
  videoRef,
  frame,
  overlay,
  analyzing,
  playheadAhead,
  framesLoading,
  framesLoaded,
  totalPreviewFrames,
}: VideoStageProps) {
  const busy = phase === 'uploading' || phase === 'processing'
  const measureRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const container = measureRef.current
    if (!container || !video) return
    const aspect = (video.display_width || video.width) / (video.display_height || video.height || 1)
    const update = () => {
      const availableWidth = container.clientWidth
      const availableHeight = container.clientHeight
      if (availableWidth <= 0 || availableHeight <= 0 || !Number.isFinite(aspect)) return
      let width = availableWidth
      let height = width / aspect
      if (height > availableHeight) {
        height = availableHeight
        width = height * aspect
      }
      setBox({ width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) })
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(container)
    return () => observer.disconnect()
  }, [video])

  return (
    <section
      aria-label="视频舞台"
      className="relative flex min-h-[280px] flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-black shadow-panel"
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-70"
        style={{
          background:
            'radial-gradient(60% 55% at 50% 42%, rgba(34,211,238,0.08), transparent 70%)',
        }}
      />

      {busy ? <ProgressState phase={phase} progress={progress} /> : null}

      {!busy && error ? (
        <div className="on-video flex max-w-md flex-col items-center gap-3 px-6 text-center">
          <div className="grid h-12 w-12 place-items-center rounded-2xl border border-danger/30 bg-danger/10">
            <AlertIcon className="h-5 w-5 text-danger" />
          </div>
          <div className="space-y-1">
            <h2 className="text-base font-medium text-slate-200">无法导入这段视频</h2>
            <p className="text-xs leading-relaxed text-slate-400">{error}</p>
          </div>
          <button type="button" className="btn-ghost" onClick={onPick}>
            重新选择视频
          </button>
        </div>
      ) : null}

      {!busy && !error && video && previewUrl ? (
        <>
          <div ref={measureRef} className="absolute inset-3 flex items-center justify-center">
            <div
              className="relative overflow-hidden rounded-lg bg-black ring-1 ring-white/5"
              style={{ width: `${box.width}px`, height: `${box.height}px` }}
            >
              <video
                key={previewUrl}
                ref={videoRef}
                src={previewUrl}
                poster={video.thumbnail_url ?? undefined}
                playsInline
                preload="auto"
                className="h-full w-full bg-black"
              />
              <SkeletonOverlay
                frame={frame}
                showSkeleton={overlay.skeleton}
                showIndices={overlay.indices}
                showBox={overlay.box}
                lineStyle={overlay.lineStyle}
              />
            </div>
          </div>

          <div className="pointer-events-none absolute left-3 top-3 flex max-w-[64%] flex-col gap-1">
            <span className="on-video truncate rounded-lg bg-black/60 px-2.5 py-1 text-2xs backdrop-blur">
              {video.filename}
            </span>
            <span className="on-video-dim rounded-lg bg-black/50 px-2.5 py-1 text-2xs backdrop-blur">
              {video.display_width}×{video.display_height} · {video.fps.toFixed(2)} fps ·{' '}
              {formatBytes(video.size_bytes)}
            </span>
          </div>

          <div className="absolute right-3 top-3 flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-1.5 rounded-lg border border-line bg-panel/90 p-1 backdrop-blur">
              <label className="flex cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-2xs text-slate-300 hover:bg-white/5">
                <input
                  type="checkbox"
                  className="h-3 w-3 accent-accent"
                  checked={overlay.skeleton}
                  onChange={(event) => overlay.onSkeleton(event.target.checked)}
                />
                骨架
              </label>
              <label className="hidden cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-2xs text-slate-300 hover:bg-white/5 sm:flex">
                <input
                  type="checkbox"
                  className="h-3 w-3 accent-accent"
                  checked={overlay.indices}
                  onChange={(event) => overlay.onIndices(event.target.checked)}
                />
                编号
              </label>
              <label className="hidden cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-2xs text-slate-300 hover:bg-white/5 md:flex">
                <input
                  type="checkbox"
                  className="h-3 w-3 accent-accent"
                  checked={overlay.box}
                  onChange={(event) => overlay.onBox(event.target.checked)}
                />
                检测框
              </label>
              <select
                value={overlay.lineStyle}
                onChange={(event) => overlay.onLineStyle(event.target.value as LineStyle)}
                className="rounded-md border-0 bg-transparent py-1 pl-1 pr-5 text-2xs text-slate-300 focus:outline-none"
                aria-label="连线样式"
              >
                <option value="solid">实线</option>
                <option value="thin">细线</option>
                <option value="dotted">点线</option>
              </select>
            </div>

            {analyzing ? (
              <span className="flex items-center gap-1.5 rounded-lg bg-black/60 px-2.5 py-1 text-2xs text-accent backdrop-blur">
                <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent" />
                识别中…{framesLoaded ? ` 已出 ${framesLoaded} 帧` : ''}
              </span>
            ) : null}
            {playheadAhead ? (
              <span className="on-video-dim rounded-lg bg-black/60 px-2.5 py-1 text-2xs backdrop-blur">
                这一段还没识别到
              </span>
            ) : null}
            {framesLoading ? (
              <span className="on-video-dim rounded-lg bg-black/60 px-2.5 py-1 text-2xs backdrop-blur">
                加载骨架 {framesLoaded}/{totalPreviewFrames || '…'}
              </span>
            ) : null}
          </div>
        </>
      ) : null}

      {!busy && !error && !video ? <EmptyState onPick={onPick} /> : null}
    </section>
  )
}
