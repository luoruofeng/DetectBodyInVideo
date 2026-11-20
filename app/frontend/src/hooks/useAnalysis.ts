import { useCallback, useEffect, useRef, useState } from 'react'

import {
  FRAME_PAGE_LIMIT,
  MAX_PREVIEW_FRAMES,
  cancelAnalysis,
  clearAnalysisCache,
  exportAnalysis,
  fetchAnalysis,
  fetchFrames,
  startAnalysis,
  type AnalysisOptions,
  type AnalysisProgress,
  type ExportArtifact,
  type ExportFormat,
  type ExportOptions,
  type FramesPayload,
  type Granularity,
  type JobStatus,
} from '../api/analysis'
import { pickExportTarget, suggestExportFilename } from '../lib/saveTarget'
import { useAppStore } from '../store/appStore'

const IDLE_PROGRESS: AnalysisProgress = {
  frames_done: 0,
  frames_total: 0,
  processed_fps: 0,
  eta_seconds: null,
  message: '',
  stage: 'inference',
  stage_done: 0,
  stage_total: 0,
}

export interface UseAnalysis {
  jobId: string | null
  status: JobStatus | null
  phase: JobStatus['phase'] | 'idle'
  progress: AnalysisProgress
  frames: FramesPayload | null
  framesLoading: boolean
  framesLoaded: number
  error: string | null
  exporting: boolean
  artifact: ExportArtifact | null
  /** 导出文件在后端 exports 目录里时的下载地址；另存到别处时为 null */
  downloadUrl: string | null
  /** 最近一次导出的深度可信度提示（schema 1.1.0 的 warnings） */
  exportWarnings: string[]
  format: ExportFormat
  setFormat: (value: ExportFormat) => void
  granularity: Granularity
  setGranularity: (value: Granularity) => void
  start: (videoId: string, options: AnalysisOptions) => Promise<void>
  cancel: () => Promise<void>
  /** 清空后端断点缓存；成功返回删除的文件数，失败返回 null（已弹出错误提示）。 */
  clearCache: () => Promise<number | null>
  exportJson: (options?: Partial<ExportOptions>) => Promise<void>
  downloadArtifact: () => void
  reset: () => void
}

export function useAnalysis(): UseAnalysis {
  const { pushToast } = useAppStore()
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [frames, setFrames] = useState<FramesPayload | null>(null)
  const [framesLoading, setFramesLoading] = useState(false)
  const [framesLoaded, setFramesLoaded] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)
  const [artifact, setArtifact] = useState<ExportArtifact | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [exportWarnings, setExportWarnings] = useState<string[]>([])
  const [format, setFormat] = useState<ExportFormat>('pose')
  const [granularity, setGranularity] = useState<Granularity>('full')

  const mounted = useRef(true)
  const framesJob = useRef<string | null>(null)
  /** 已合并到 frames 里的帧数，也是下一次增量请求的 offset（推理中数据只追加不修改）。 */
  const framesOffset = useRef(0)
  const framesBusy = useRef(false)
  /** 请求途中收到"需要整体重载"（例如任务完成切到最终结果）时置位。 */
  const framesReload = useRef(false)
  const syncedFramesDone = useRef(-1)
  const timer = useRef<number | null>(null)
  const lastPhase = useRef<JobStatus['phase'] | 'idle'>('idle')
  const loadGeneration = useRef(0)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      if (timer.current !== null) window.clearTimeout(timer.current)
    }
  }, [])

  /**
   * 拉取帧数据。推理进行中后端会持续追加"已识别帧"，这里按 offset 增量合并，
   * 这样边算边播也能看到骨架；``reset`` 用于任务完成后整体换成最终结果。
   */
  const loadFrames = useCallback(
    async (id: string, reset = false) => {
      if (reset) framesReload.current = true
      if (framesBusy.current) return
      framesBusy.current = true
      try {
        // 一次请求期间可能又来了重载请求（预览 → 最终结果），处理完后再补一轮
        for (;;) {
          const switchedJob = framesJob.current !== id
          if (framesReload.current || switchedJob) {
            framesJob.current = id
            framesOffset.current = 0
            framesReload.current = false
            loadGeneration.current += 1
            // 换视频才清空；预览 → 最终结果整体重载时先留着旧帧，避免闪白
            if (switchedJob) setFrames(null)
            setFramesLoaded(0)
            setFramesLoading(true)
          }
          const generation = loadGeneration.current
          const isCurrent = () => mounted.current && generation === loadGeneration.current
          let offset = framesOffset.current
          try {
            while (offset < MAX_PREVIEW_FRAMES) {
              const page = await fetchFrames(id, offset, FRAME_PAGE_LIMIT)
              if (!isCurrent()) return
              const start = offset
              // 不能用 {...page, frames: [...]} 的旧值：这里只追加，永远拼在已有帧之后
              setFrames((previous) => ({
                ...page,
                offset: 0,
                frames: [...(start === 0 ? [] : previous?.frames ?? []), ...page.frames],
              }))
              offset += page.frames.length
              framesOffset.current = offset
              setFramesLoaded(offset)
              if (!page.hasMore || page.frames.length === 0) break
            }
          } catch (cause) {
            if (!isCurrent()) return
            const message = cause instanceof Error ? cause.message : '读取帧数据失败'
            setFramesLoading(false)
            setError(message)
            pushToast({ kind: 'error', message: '读取分析结果失败', detail: message })
            return
          }
          if (!isCurrent()) return
          setFramesLoading(false)
          if (!framesReload.current) return
        }
      } finally {
        framesBusy.current = false
      }
    },
    [pushToast],
  )

  const poll = useCallback(
    async (id: string) => {
      try {
        const next = await fetchAnalysis(id)
        if (!mounted.current) return
        setStatus(next)
        setError(next.error ?? null)
        const wasPreview = framesJob.current === id && lastPhase.current !== 'done'
        lastPhase.current = next.phase
        if (next.phase === 'done') {
          if (next.canExport) await loadFrames(id, wasPreview)
          return
        }
        // 推理中边算边看：帧数涨了就增量拉一次，不必等推理/后处理全部跑完。
        if (
          next.canPreview &&
          (next.phase === 'queued' || next.phase === 'running') &&
          next.progress.frames_done !== syncedFramesDone.current
        ) {
          syncedFramesDone.current = next.progress.frames_done
          void loadFrames(id)
        }
        if (next.phase === 'queued' || next.phase === 'running') {
          timer.current = window.setTimeout(() => void poll(id), 320)
        }
      } catch (cause) {
        if (!mounted.current) return
        const message = cause instanceof Error ? cause.message : '获取任务状态失败'
        setError(message)
        timer.current = window.setTimeout(() => void poll(id), 1200)
      }
    },
    [loadFrames],
  )

  useEffect(() => {
    if (!jobId) return
    void poll(jobId)
    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current)
        timer.current = null
      }
    }
  }, [jobId, poll])

  const start = useCallback(
    async (videoId: string, options: AnalysisOptions) => {
      if (timer.current !== null) window.clearTimeout(timer.current)
      framesJob.current = null
      framesOffset.current = 0
      framesReload.current = false
      syncedFramesDone.current = -1
      loadGeneration.current += 1
      lastPhase.current = 'idle'
      setFrames(null)
      setFramesLoaded(0)
      setArtifact(null)
      setDownloadUrl(null)
      setExportWarnings([])
      setError(null)
      setStatus(null)
      try {
        const created = await startAnalysis(videoId, options)
        if (!mounted.current) return
        setJobId(created.id)
        setStatus(created)
        pushToast({
          kind: 'info',
          message: '已开始识别关键点',
          detail: `任务 ${created.id} · ${options.sampling?.every_n_frames && options.sampling.every_n_frames > 1 ? `每 ${options.sampling.every_n_frames} 帧取一帧` : '全帧分析'}`,
        })
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '启动分析失败'
        setError(message)
        pushToast({ kind: 'error', message: '启动分析失败', detail: message })
        throw cause
      }
    },
    [pushToast],
  )

  const cancel = useCallback(async () => {
    if (!jobId) return
    try {
      const next = await cancelAnalysis(jobId)
      if (mounted.current) setStatus(next)
      pushToast({ kind: 'warn', message: '已请求取消', detail: '正在释放线程与显存' })
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : '取消失败'
      pushToast({ kind: 'error', message, detail: '任务可能已经结束' })
    }
  }, [jobId, pushToast])

  const exportJson = useCallback(
    async (options?: Partial<ExportOptions>) => {
      if (!jobId || !status || status.phase !== 'done') {
        pushToast({ kind: 'warn', message: '请先完成一次分析再导出' })
        return
      }
      setExporting(true)
      try {
        // 先弹系统「另存为」：用户自己定文件名与目录（默认落在桌面）。
        // 桌面壳不可用时返回 undefined，退回后端默认目录 + 下载链接的老逻辑。
        const target = await pickExportTarget(
          suggestExportFilename(status.filename, format, granularity),
        )
        if (target === null) return // 用户取消，不写任何文件
        const response = await exportAnalysis(jobId, {
          format,
          granularity,
          include_raw_landmarks: options?.include_raw_landmarks ?? false,
          include_root_motion: options?.include_root_motion ?? true,
          output_dir: target?.directory ?? options?.output_dir ?? null,
          filename: target?.filename ?? options?.filename ?? null,
          motion:
            format === 'motion'
              ? { sample_interval_ms: 100, ...options?.motion }
              : null,
        })
        if (!mounted.current) return
        setArtifact(response.artifact)
        setDownloadUrl(response.downloadUrl ?? null)
        setExportWarnings(response.warnings ?? [])
        const depthNote =
          response.warnings && response.warnings.length
            ? ` · 深度可信度 ${(response.depthConfidence * 100).toFixed(0)}%（前后方向仅为估计值）`
            : ''
        pushToast({
          kind: response.warnings && response.warnings.length ? 'warn' : 'success',
          message: response.warnings && response.warnings.length ? '导出完成（含深度提示）' : '导出完成',
          detail: `${response.artifact.filename} · ${response.artifact.frames} 帧 · ${(response.artifact.bytes / 1024).toFixed(1)} KiB${depthNote}`,
        })
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '导出失败'
        pushToast({ kind: 'error', message: '导出失败', detail: message })
      } finally {
        if (mounted.current) setExporting(false)
      }
    },
    [format, granularity, jobId, pushToast, status],
  )

  const downloadArtifact = useCallback(() => {
    if (!artifact) return
    if (!downloadUrl) {
      // 用户把文件存到了自选目录：后端 exports 目录里没有副本，只能提示实际位置
      pushToast({ kind: 'info', message: '文件已保存到自选位置', detail: artifact.path })
      return
    }
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = artifact.filename
    link.rel = 'noopener'
    document.body.appendChild(link)
    link.click()
    link.remove()
  }, [artifact, downloadUrl, pushToast])

  const reset = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    framesJob.current = null
    framesOffset.current = 0
    framesReload.current = false
    syncedFramesDone.current = -1
    loadGeneration.current += 1
    lastPhase.current = 'idle'
    setJobId(null)
    setStatus(null)
    setFrames(null)
    setFramesLoaded(0)
    setFramesLoading(false)
    setError(null)
    setArtifact(null)
    setDownloadUrl(null)
    setExportWarnings([])
    setExporting(false)
  }, [])

  const clearCache = useCallback(async () => {
    try {
      const result = await clearAnalysisCache()
      pushToast({
        kind: 'success',
        message: '已清空识别缓存',
        detail: result.removed
          ? `删除 ${result.removed} 个断点文件，重新识别会从第 1 帧开始`
          : '没有残留断点文件，重新识别会从第 1 帧开始',
      })
      return result.removed
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : '清空缓存失败'
      pushToast({ kind: 'error', message: '清空缓存失败', detail: message })
      return null
    }
  }, [pushToast])

  return {
    jobId,
    status,
    phase: status?.phase ?? 'idle',
    progress: status?.progress ?? IDLE_PROGRESS,
    frames,
    framesLoading,
    framesLoaded,
    error,
    exporting,
    artifact,
    downloadUrl,
    exportWarnings,
    format,
    setFormat,
    granularity,
    setGranularity,
    start,
    cancel,
    clearCache,
    exportJson,
    downloadArtifact,
    reset,
  }
}
