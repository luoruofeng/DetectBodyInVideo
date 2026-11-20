import { useEffect, useRef } from 'react'

import { providerLabel, type HardwareReport } from '../api/hardware'
import { CpuIcon } from './icons'

interface AccelerationBadgeProps {
  report: HardwareReport | null
  loading: boolean
  error: string | null
  onRefresh: () => void
}

type BadgeState = 'gpu' | 'cpu' | 'error' | 'loading'

const TONE: Record<BadgeState, string> = {
  gpu: 'border-ok/30 bg-ok/10 text-ok hover:border-ok/50',
  cpu: 'border-warn/30 bg-warn/10 text-warn hover:border-warn/50',
  error: 'border-danger/40 bg-danger/10 text-danger hover:border-danger/60',
  loading: 'border-line bg-white/[0.03] text-slate-400',
}

function Dot({ state }: { state: BadgeState }) {
  const color =
    state === 'gpu'
      ? 'bg-ok'
      : state === 'cpu'
        ? 'bg-warn'
        : state === 'error'
          ? 'bg-danger'
          : 'bg-slate-400'
  return <span className={`h-1.5 w-1.5 rounded-full ${color} ${state === 'loading' ? 'animate-pulse-soft' : ''}`} />
}

export function AccelerationBadge({ report, loading, error, onRefresh }: AccelerationBadgeProps) {
  const detailsRef = useRef<HTMLDetailsElement>(null)

  // 点击面板外部 / 按 Esc 收起，避免浮层一直盖在视频画面上
  useEffect(() => {
    const element = detailsRef.current
    if (!element) return
    const close = () => {
      element.open = false
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!element.open) return
      if (event.target instanceof Node && element.contains(event.target)) return
      close()
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (!element.open || event.key !== 'Escape') return
      close()
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  const state: BadgeState = error
    ? 'error'
    : loading && !report
      ? 'loading'
      : report?.acceleration.gpu_accelerated
        ? 'gpu'
        : 'cpu'

  const plan = report?.acceleration
  const label =
    state === 'loading'
      ? '检测加速能力…'
      : state === 'error'
        ? '加速不可用'
        : state === 'gpu'
          ? `GPU · ${providerLabel(plan!.preferred_provider)}`
          : `CPU 模式 · ${plan?.threads ?? '-'} 线程`

  const summary =
    state === 'error'
      ? (error ?? '硬件探测失败')
      : state === 'loading'
        ? '正在读取 CPU / GPU 与执行提供器'
        : state === 'gpu'
          ? `${plan?.device_name ?? 'GPU'} · ${plan?.model_tier} 档模型`
          : '未启用 GPU 加速，推理走 CPU 多线程'

  return (
    <details ref={detailsRef} className="group relative">
      <summary
        className={`chip cursor-pointer list-none transition [&::-webkit-details-marker]:hidden ${TONE[state]}`}
        title={summary}
        aria-label={`加速状态：${label}。${summary}`}
      >
        <Dot state={state} />
        {label}
        <svg viewBox="0 0 12 12" className="h-2.5 w-2.5 opacity-60 transition group-open:rotate-180" aria-hidden>
          <path d="m3 4.5 3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      </summary>

      <div className="absolute right-0 top-[calc(100%+8px)] z-50 max-h-[calc(100vh-5.5rem)] w-80 max-w-[calc(100vw-1.5rem)] overflow-y-auto rounded-xl border border-line bg-elevated p-3 text-slate-300 shadow-panel animate-fade-in">
        <div className="flex items-start gap-2 border-b border-line pb-2.5">
          <CpuIcon className="mt-0.5 h-4 w-4 text-accent" />
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-slate-200">{label}</p>
            <p className="mt-0.5 text-2xs leading-relaxed text-slate-500">{summary}</p>
          </div>
        </div>

        {plan ? (
          <div className="space-y-1 py-2.5">
            <div className="kv">
              <span className="kv-key">首选 EP</span>
              <span className="kv-value text-accent">{plan.preferred_provider}</span>
            </div>
            <div className="kv">
              <span className="kv-key">EP 链</span>
              <span className="kv-value" title={plan.execution_providers.join(' → ')}>
                {plan.execution_providers.map(providerLabel).join(' → ')}
              </span>
            </div>
            <div className="kv">
              <span className="kv-key">推理线程</span>
              <span className="kv-value">{plan.threads}</span>
            </div>
            <div className="kv">
              <span className="kv-key">模型档位</span>
              <span className="kv-value uppercase">{plan.model_tier}</span>
            </div>
          </div>
        ) : null}

        <ul className="max-h-40 space-y-1.5 overflow-y-auto border-t border-line pt-2.5">
          {(plan?.reasons ?? [error ?? '尚未获取到硬件报告']).map((reason) => (
            <li key={reason} className="flex gap-1.5 text-2xs leading-relaxed text-slate-500">
              <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-slate-600" />
              <span>{reason}</span>
            </li>
          ))}
        </ul>

        <button type="button" className="btn-ghost mt-2.5 w-full py-1.5 text-2xs" onClick={onRefresh}>
          重新探测硬件
        </button>
      </div>
    </details>
  )
}
