import { useEffect, useState } from 'react'

import { api } from '../api/client'

export interface HealthInfo {
  status: string
  name: string
  version: string
  python: string
  platform: string
}

export type ApiStatus = 'checking' | 'online' | 'offline'

interface ApiHealthState {
  status: ApiStatus
  info: HealthInfo | null
  error: string | null
}

/** 轮询 /api/health：桌面壳与开发模式都靠它判断服务是否可用。 */
export function useApiHealth(intervalMs = 10_000): ApiHealthState {
  const [state, setState] = useState<ApiHealthState>({
    status: 'checking',
    info: null,
    error: null,
  })

  useEffect(() => {
    let alive = true

    const check = async () => {
      try {
        const info = await api.get<HealthInfo>('/health')
        if (!alive) return
        setState({ status: 'online', info, error: null })
      } catch (cause) {
        if (!alive) return
        setState({
          status: 'offline',
          info: null,
          error: cause instanceof Error ? cause.message : '无法连接本地服务',
        })
      }
    }

    void check()
    const timer = window.setInterval(check, intervalMs)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [intervalMs])

  return state
}
