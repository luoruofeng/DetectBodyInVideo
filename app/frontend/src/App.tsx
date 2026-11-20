import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { AnalysisOptions } from './api/analysis'
import { AppHeader } from './components/AppHeader'
import { PlaybackControls } from './components/PlaybackControls'
import { ShootingGuideDialog } from './components/ShootingGuide'
import { SidePanel } from './components/SidePanel'
import type { LineStyle } from './components/SkeletonOverlay'
import { Timeline } from './components/Timeline'
import { Toaster } from './components/Toaster'
import { UploadDropzone } from './components/UploadDropzone'
import { VideoStage } from './components/VideoStage'
import { useAnalysis } from './hooks/useAnalysis'
import { useApiHealth } from './hooks/useApiHealth'
import { useHardware } from './hooks/useHardware'
import { useModels } from './hooks/useModels'
import { usePlayback } from './hooks/usePlayback'
import { useTheme } from './hooks/useTheme'
import { useVideoUpload } from './hooks/useVideoUpload'
import { findFrameIndex } from './lib/pose'

const FALLBACK_DURATION = 1

export default function App() {
  const { status, info } = useApiHealth()
  const hardware = useHardware()
  const upload = useVideoUpload()
  const analysis = useAnalysis()
  const models = useModels()
  const { theme, toggle: toggleTheme } = useTheme()

  const videoRef = useRef<HTMLVideoElement>(null)
  const openPickerRef = useRef<() => void>(() => {})
  const video = upload.state.video
  const frames = useMemo(() => analysis.frames?.frames ?? [], [analysis.frames])

  const playback = usePlayback(videoRef, {
    duration: video?.duration_s ?? 0,
    fps: video?.fps ?? 30,
    fallbackDuration: FALLBACK_DURATION,
    mediaKey: video?.id ?? null,
  })
  const { seek, toggle: togglePlayback, step } = playback

  const [overlay, setOverlay] = useState({
    skeleton: true,
    indices: false,
    box: true,
    lineStyle: 'solid' as LineStyle,
  })

  const frameIndex = useMemo(
    () => (frames.length ? findFrameIndex(frames, playback.currentTime) : -1),
    [frames, playback.currentTime],
  )
  // 采样间隔（秒）：推理进行中帧数据只到"已识别位置"，用于判断画面能否对上骨架
  const frameInterval = useMemo(() => {
    const stride = analysis.frames?.stride ?? 1
    const fps = analysis.frames?.fps ?? video?.fps ?? 30
    return stride / Math.max(1, fps)
  }, [analysis.frames?.fps, analysis.frames?.stride, video?.fps])
  const frame = useMemo(() => {
    if (frameIndex < 0) return null
    const candidate = frames[frameIndex]
    if (!candidate) return null
    // 播放位置已超过已识别范围时不要把上一帧骨架硬贴在画面上（看起来像骨架不跟随）
    if (playback.currentTime - candidate.t > Math.max(0.25, frameInterval * 2)) return null
    return candidate
  }, [frameIndex, frameInterval, frames, playback.currentTime])

  // 识别进行中、播放头已经超出"已出帧"范围时给个提示，避免误以为叠加层坏了
  const analyzedUntil = frames.length ? frames[frames.length - 1].t : null
  const playheadAhead = useMemo(() => {
    if (analysis.phase !== 'running' && analysis.phase !== 'queued') return false
    if (analyzedUntil === null) return false
    return playback.currentTime > analyzedUntil + Math.max(0.25, frameInterval * 2)
  }, [analysis.phase, analyzedUntil, frameInterval, playback.currentTime])

  // 换视频时清空上一轮分析结果
  const videoId = video?.id ?? null
  useEffect(() => {
    analysis.reset()
    playback.seek(0)
    playback.clearLoop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoId])

  const startAnalysis = useCallback(
    (options: AnalysisOptions) => {
      if (!video) return
      void analysis.start(video.id, options)
    },
    [analysis, video],
  )

  const exportJson = useCallback(() => {
    void analysis.exportJson({
      include_raw_landmarks: analysis.granularity === 'full',
    })
  }, [analysis])

  // 全局快捷键（任务 8.3）
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target?.closest('input, textarea, select, button, a, [contenteditable="true"]')) return

      const key = event.key
      if (key === ' ') {
        if (!video) return
        event.preventDefault()
        togglePlayback()
        return
      }
      if (key === 'ArrowLeft' || key === 'ArrowRight') {
        if (!video) return
        event.preventDefault()
        step((key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? 10 : 1))
        return
      }
      if (key === 'Home' || key === 'End') {
        if (!video) return
        event.preventDefault()
        seek(key === 'Home' ? 0 : playback.duration)
        return
      }

      const lower = key.toLowerCase()
      if (lower === 't') toggleTheme()
      else if (lower === 'e') exportJson()
      else if (lower === 'u') openPickerRef.current()
      else if (lower === 's')
        setOverlay((previous) => ({ ...previous, skeleton: !previous.skeleton }))
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [exportJson, playback.duration, seek, step, togglePlayback, toggleTheme, video])

  return (
    <UploadDropzone onFile={(file) => void upload.start(file)}>
      {(openPicker) => {
        openPickerRef.current = openPicker
        return (
          <div className="flex h-full min-h-0 flex-col overflow-hidden bg-canvas">
          <AppHeader
            apiStatus={status}
            health={info}
            hardware={hardware.report}
            hardwareLoading={hardware.loading}
            hardwareError={hardware.error}
            onRefreshHardware={hardware.refresh}
            theme={theme}
            onToggleTheme={toggleTheme}
            canExport={analysis.phase === 'done'}
            exporting={analysis.exporting}
            onExport={exportJson}
          />

          <main className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3 sm:p-4 lg:flex-row lg:overflow-hidden">
            <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
              <VideoStage
                video={video}
                previewUrl={upload.state.previewUrl}
                phase={upload.state.phase}
                progress={upload.state.progress}
                error={upload.state.error}
                onPick={openPicker}
                videoRef={videoRef}
                frame={frame}
                overlay={{
                  skeleton: overlay.skeleton,
                  indices: overlay.indices,
                  box: overlay.box,
                  lineStyle: overlay.lineStyle,
                  onSkeleton: (value) => setOverlay((previous) => ({ ...previous, skeleton: value })),
                  onIndices: (value) => setOverlay((previous) => ({ ...previous, indices: value })),
                  onBox: (value) => setOverlay((previous) => ({ ...previous, box: value })),
                  onLineStyle: (value) =>
                    setOverlay((previous) => ({ ...previous, lineStyle: value })),
                }}
                analyzing={analysis.phase === 'running' || analysis.phase === 'queued'}
                playheadAhead={playheadAhead}
                framesLoading={analysis.framesLoading}
                framesLoaded={analysis.framesLoaded}
                totalPreviewFrames={analysis.frames?.totalFrames ?? 0}
              />

              <div className="card shrink-0 space-y-2.5 p-3">
                <PlaybackControls
                  playing={playback.playing}
                  currentTime={playback.currentTime}
                  duration={playback.duration}
                  rate={playback.rate}
                  frameIndex={frame?.index ?? -1}
                  totalFrames={analysis.frames?.totalFrames ?? 0}
                  frameValid={frame?.valid ?? true}
                  disabled={!video}
                  onToggle={playback.toggle}
                  onStep={playback.step}
                  onSetRate={playback.setRate}
                />
                <Timeline
                  duration={playback.duration}
                  currentTime={playback.currentTime}
                  frames={frames}
                  loop={playback.loop}
                  onSeek={playback.seek}
                  onSetLoopStart={playback.setLoopStart}
                  onSetLoopEnd={playback.setLoopEnd}
                  onClearLoop={playback.clearLoop}
                />
              </div>
            </div>

            <SidePanel
              health={info}
              apiStatus={status}
              hardware={hardware.report}
              hardwareLoading={hardware.loading}
              hardwareError={hardware.error}
              onRefreshHardware={hardware.refresh}
              video={video}
              onReplaceVideo={openPicker}
              analysis={analysis}
              onStart={startAnalysis}
              frame={frame}
              theme={theme}
              models={models}
            />
          </main>

            <Toaster />
            <ShootingGuideDialog />
          </div>
        )
      }}
    </UploadDropzone>
  )
}
