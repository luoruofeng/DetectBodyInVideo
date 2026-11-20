import { useCallback, useEffect, useRef, useState } from 'react'

import {
  downloadModel,
  listModels,
  type DownloadTask,
  type ModelKind,
  type ModelsPayload,
} from '../api/models'
import { useAppStore } from '../store/appStore'

export interface UseModels {
  payload: ModelsPayload | null
  loading: boolean
  error: string | null
  active: DownloadTask | null
  mirror: boolean
  setMirror: (value: boolean) => void
  download: (tier: string, kind?: ModelKind) => Promise<void>
  refresh: () => Promise<void>
}

const POLL_MS = 600

/** 首启模型检查 + 下载进度轮询（任务 10.4）。 */
export function useModels(): UseModels {
  const { pushToast } = useAppStore()
  const [payload, setPayload] = useState<ModelsPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [mirror, setMirror] = useState(false)
  const timer = useRef<number | null>(null)
  const mounted = useRef(true)

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  const refresh = useCallback(async () => {
    try {
      const next = await listModels()
      if (!mounted.current) return
      setPayload(next)
      setError(null)
      const running = next.downloads.some(
        (task) => task.state === 'queued' || task.state === 'running',
      )
      stopPolling()
      if (running) {
        timer.current = window.setTimeout(() => void refresh(), POLL_MS)
      } else {
        setLoading(false)
      }
    } catch (cause) {
      if (!mounted.current) return
      const message = cause instanceof Error ? cause.message : '读取模型状态失败'
      setError(message)
    } finally {
      if (mounted.current) setLoading(false)
    }
  }, [stopPolling])

  useEffect(() => {
    mounted.current = true
    void refresh()
    return () => {
      mounted.current = false
      stopPolling()
    }
  }, [refresh, stopPolling])

  const download = useCallback(
    async (tier: string, kind: ModelKind = 'pose') => {
      try {
        const task = await downloadModel(kind, tier, mirror)
        if (!mounted.current) return
        pushToast({
          kind: 'info',
          message: '开始下载姿态模型',
          detail: `${task.filename}${mirror ? ' · 镜像' : ''}`,
        })
        stopPolling()
        timer.current = window.setTimeout(() => void refresh(), POLL_MS)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '下载失败'
        setError(message)
        pushToast({ kind: 'error', message: '模型下载失败', detail: message })
      }
    },
    [mirror, pushToast, refresh, stopPolling],
  )

  const active =
    payload?.downloads.find((task) => task.state === 'running' || task.state === 'queued') ??
    payload?.downloads[0] ??
    null

  return {
    payload,
    loading,
    error,
    active,
    mirror,
    setMirror,
    download,
    refresh,
  }
}
