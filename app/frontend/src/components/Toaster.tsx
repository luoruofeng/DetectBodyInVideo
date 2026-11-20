import { useAppStore, type ToastKind } from '../store/appStore'
import { AlertIcon, CheckIcon, CloseIcon } from './icons'

const STYLE: Record<ToastKind, string> = {
  info: 'border-line bg-elevated text-slate-200',
  success: 'border-ok/30 bg-ok/10 text-ok',
  warn: 'border-warn/30 bg-warn/10 text-warn',
  error: 'border-danger/40 bg-danger/10 text-danger',
}

export function Toaster() {
  const { toasts, dismissToast } = useAppStore()

  if (toasts.length === 0) return null

  return (
    <div className="pointer-events-none fixed left-1/2 top-[68px] z-50 flex w-[min(420px,calc(100vw-2rem))] -translate-x-1/2 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          role="status"
          className={`pointer-events-auto flex items-start gap-2.5 rounded-xl border px-3.5 py-3 shadow-panel animate-fade-in ${STYLE[toast.kind]}`}
        >
          <span className="mt-0.5 shrink-0">
            {toast.kind === 'success' ? (
              <CheckIcon className="h-4 w-4" />
            ) : (
              <AlertIcon className="h-4 w-4" />
            )}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium leading-relaxed">{toast.message}</p>
            {toast.detail ? (
              <p className="mt-1 break-words text-2xs leading-relaxed text-slate-400">
                {toast.detail}
              </p>
            ) : null}
          </div>
          <button
            type="button"
            aria-label="关闭提示"
            onClick={() => dismissToast(toast.id)}
            className="shrink-0 rounded-md p-0.5 text-current/60 transition hover:bg-white/10"
          >
            <CloseIcon className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
    </div>
  )
}
