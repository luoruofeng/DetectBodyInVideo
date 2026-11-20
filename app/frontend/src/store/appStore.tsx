import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

export type ToastKind = 'info' | 'success' | 'warn' | 'error'

export interface Toast {
  id: string
  kind: ToastKind
  message: string
  detail?: string
}

interface AppStoreValue {
  toasts: Toast[]
  pushToast: (toast: Omit<Toast, 'id'> & { id?: string }) => string
  dismissToast: (id: string) => void
}

const AppStoreContext = createContext<AppStoreValue | null>(null)

let toastSeq = 0

export function AppProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timers = useRef<number[]>([])

  const dismissToast = useCallback((id: string) => {
    setToasts((list) => list.filter((toast) => toast.id !== id))
  }, [])

  const pushToast = useCallback(
    (toast: Omit<Toast, 'id'> & { id?: string }) => {
      const id = toast.id ?? `toast-${++toastSeq}`
      setToasts((list) => {
        const next = list.filter((item) => item.id !== id)
        return [...next.slice(-2), { ...toast, id }]
      })
      // 错误提示停留更久但也自动消失，避免一直挡住界面
      const delay = toast.kind === 'error' ? 8000 : 4500
      const timer = window.setTimeout(() => dismissToast(id), delay)
      timers.current.push(timer)
      return id
    },
    [dismissToast],
  )

  useEffect(() => {
    const pending = timers.current
    return () => {
      pending.forEach((timer) => window.clearTimeout(timer))
    }
  }, [])

  const value = useMemo(
    () => ({ toasts, pushToast, dismissToast }),
    [toasts, pushToast, dismissToast],
  )

  return <AppStoreContext.Provider value={value}>{children}</AppStoreContext.Provider>
}

export function useAppStore(): AppStoreValue {
  const value = useContext(AppStoreContext)
  if (!value) {
    throw new Error('useAppStore 必须在 <AppProvider> 内使用')
  }
  return value
}
