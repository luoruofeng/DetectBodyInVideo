import { useCallback, useEffect, useState } from 'react'

import { fetchHardware, type HardwareReport } from '../api/hardware'

interface HardwareState {
  report: HardwareReport | null
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useHardware(): HardwareState {
  const [report, setReport] = useState<HardwareReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (force: boolean) => {
    setLoading(true)
    try {
      const data = await fetchHardware(force)
      setReport(data)
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法获取硬件信息')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(false)
  }, [load])

  const refresh = useCallback(() => {
    void load(true)
  }, [load])

  return { report, loading, error, refresh }
}
