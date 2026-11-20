import { useCallback, useEffect, useRef, useState } from 'react'

import {
  activateAccelerator,
  cancelAcceleratorInstall,
  fetchAccelerators,
  installAccelerator,
  removeAccelerator,
  restartApp,
  type AcceleratorPayload,
  type AcceleratorVariant,
} from '../api/runtime'
import { useAppStore } from '../store/appStore'

export interface UseAccelerators {
  payload: AcceleratorPayload | null
  loading: boolean
  error: string | null
  installing: boolean
  install: (variant: AcceleratorVariant) => Promise<void>
  cancel: () => Promise<void>
  remove: (variant: AcceleratorVariant) => Promise<void>
  activate: (variant: AcceleratorVariant) => Promise<void>
  restart: () => Promise<void>
  refresh: () => Promise<void>
}

const POLL_MS = 700

/** GPU 运行时安装状态：起任务后轮询，安装完成即停止（需重启才生效）。 */
export function useAccelerators(): UseAccelerators {
  const { pushToast } = useAppStore()
  const [payload, setPayload] = useState<AcceleratorPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
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
      const next = await fetchAccelerators()
      if (!mounted.current) return
      setPayload(next)
      setError(null)
      const busy = next.task?.state === 'queued' || next.task?.state === 'running'
      stopPolling()
      if (busy) {
        timer.current = window.setTimeout(() => void refresh(), POLL_MS)
      }
    } catch (cause) {
      if (!mounted.current) return
      setError(cause instanceof Error ? cause.message : '读取加速状态失败')
      stopPolling()
      timer.current = window.setTimeout(() => void refresh(), 2000)
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

  const install = useCallback(
    async (variant: AcceleratorVariant) => {
      try {
        const task = await installAccelerator(variant)
        pushToast({ kind: 'info', message: '开始安装 GPU 加速运行时', detail: task.variant })
        stopPolling()
        timer.current = window.setTimeout(() => void refresh(), POLL_MS)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '安装失败'
        pushToast({ kind: 'error', message: '无法开始安装', detail: message })
      }
    },
    [pushToast, refresh, stopPolling],
  )

  const cancel = useCallback(async () => {
    try {
      await cancelAcceleratorInstall()
      pushToast({ kind: 'info', message: '已请求取消安装' })
      stopPolling()
      timer.current = window.setTimeout(() => void refresh(), POLL_MS)
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : '取消失败'
      pushToast({ kind: 'error', message: '取消失败', detail: message })
    }
  }, [pushToast, refresh, stopPolling])

  const remove = useCallback(
    async (variant: AcceleratorVariant) => {
      try {
        await removeAccelerator(variant)
        pushToast({ kind: 'info', message: '已移除 GPU 运行时，重启后回到 CPU 推理' })
        await refresh()
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '移除失败'
        pushToast({ kind: 'error', message: '移除失败', detail: message })
      }
    },
    [pushToast, refresh],
  )

  const activate = useCallback(
    async (variant: AcceleratorVariant) => {
      try {
        await activateAccelerator(variant)
        pushToast({ kind: 'info', message: '已切换加速运行时', detail: '重启应用后生效' })
        await refresh()
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : '切换失败'
        pushToast({ kind: 'error', message: '切换失败', detail: message })
      }
    },
    [pushToast, refresh],
  )

  const restart = useCallback(async () => {
    try {
      await restartApp()
      pushToast({ kind: 'info', message: '正在重启应用', detail: '稍候会出现新的窗口' })
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : '重启失败'
      pushToast({ kind: 'error', message: '重启失败', detail: message })
    }
  }, [pushToast])

  const task = payload?.task ?? null
  const installing = task?.state === 'queued' || task?.state === 'running'

  return {
    payload,
    loading,
    error,
    installing: Boolean(installing),
    install,
    cancel,
    remove,
    activate,
    restart,
    refresh,
  }
}
