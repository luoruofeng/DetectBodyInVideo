import { providerLabel, type HardwareReport } from '../api/hardware'
import { useAccelerators } from '../hooks/useAccelerators'
import { AlertIcon, CheckIcon } from './icons'
import { RuntimeInstallPanel } from './RuntimeInstallPanel'

function formatGb(mb: number | null | undefined): string {
  if (!mb) return '未知'
  return `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB`
}

function avxLabel(value: boolean | null): { text: string; tone: string } {
  if (value === null) return { text: '未知', tone: 'text-slate-500' }
  return value ? { text: '支持', tone: 'text-ok' } : { text: '不支持', tone: 'text-warn' }
}

function SkeletonRows() {
  return (
    <div className="space-y-2 px-4 py-3">
      {[0, 1, 2, 3].map((row) => (
        <div key={row} className="flex items-center justify-between gap-4">
          <div className="h-2.5 w-16 animate-pulse-soft rounded bg-white/5" />
          <div className="h-2.5 w-28 animate-pulse-soft rounded bg-white/10" />
        </div>
      ))}
    </div>
  )
}

interface HardwareCardProps {
  report: HardwareReport | null
  loading: boolean
  error: string | null
  onRefresh: () => void
}

export function HardwareCard({ report, loading, error, onRefresh }: HardwareCardProps) {
  const system = report?.system
  const plan = report?.acceleration
  const avx2 = avxLabel(system?.cpu.avx2 ?? null)
  const gpuIdle = Boolean(system?.gpus.length) && !plan?.gpu_accelerated
  const accelerators = useAccelerators()
  const pendingRestart = Boolean(accelerators.payload?.restartRequired)

  return (
    <section className="card">
      <h2 className="card-title">
        硬件与加速
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="rounded-md px-1.5 py-0.5 text-2xs font-normal normal-case tracking-normal text-slate-500 transition hover:bg-white/5 hover:text-slate-300 disabled:opacity-40"
        >
          {loading ? '探测中…' : '重新探测'}
        </button>
      </h2>

      {error ? (
        <div className="flex items-start gap-2 px-4 py-3 text-2xs leading-relaxed text-danger">
          <AlertIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : !report ? (
        <SkeletonRows />
      ) : (
        <div className="space-y-1 px-4 py-3">
          <div className="kv">
            <span className="kv-key">CPU</span>
            <span className="kv-value" title={system?.cpu.model}>
              {system?.cpu.model}
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">核心</span>
            <span className="kv-value">
              {system?.cpu.physical_cores} 物理 / {system?.cpu.logical_cores} 逻辑
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">AVX2</span>
            <span className={`kv-value ${avx2.tone}`}>{avx2.text}</span>
          </div>
          <div className="kv">
            <span className="kv-key">内存</span>
            <span className="kv-value">
              {formatGb(system?.memory.total_mb)}
              {system?.memory.available_mb ? `（可用 ${formatGb(system.memory.available_mb)}）` : ''}
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">GPU</span>
            <span className="kv-value" title={system?.gpus.map((gpu) => gpu.name).join(', ')}>
              {system?.gpus.length
                ? `${system.gpus[0].name}${system.gpus.length > 1 ? ` +${system.gpus.length - 1}` : ''}`
                : '未检测到'}
            </span>
          </div>
          {system?.gpus.length ? (
            <div className="kv">
              <span className="kv-key">显存</span>
              <span className="kv-value">{formatGb(system.gpus[0].memory_mb)}</span>
            </div>
          ) : null}

          <div className="my-2 border-t border-line" />

          <div className="kv">
            <span className="kv-key">推理后端</span>
            <span className={`kv-value ${plan?.gpu_accelerated ? 'text-ok' : 'text-warn'}`}>
              {plan ? providerLabel(plan.preferred_provider) : '—'}
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">EP 链</span>
            <span className="kv-value" title={plan?.execution_providers.join(' → ')}>
              {plan?.execution_providers.map(providerLabel).join(' → ') ?? '—'}
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">线程 / 档位</span>
            <span className="kv-value">
              {plan?.threads ?? '—'} / {plan?.model_tier ?? '—'}
            </span>
          </div>

          {gpuIdle ? (
            <p className="mt-2 flex items-start gap-1.5 rounded-lg border border-warn/25 bg-warn/10 px-2.5 py-2 text-2xs leading-relaxed text-warn">
              <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
              {pendingRestart
                ? '已安装 GPU 运行时，重启应用后启用 GPU 推理。'
                : '检测到 GPU，但 onnxruntime 未启用对应 EP，当前仍为 CPU 推理。'}
            </p>
          ) : null}

          <RuntimeInstallPanel accelerators={accelerators} onRefreshHardware={onRefresh} />

          {system?.notes.length ? (
            <ul className="mt-2 space-y-1 border-t border-line pt-2">
              {system.notes.map((note) => (
                <li key={note} className="flex gap-1.5 text-2xs leading-relaxed text-slate-500">
                  <AlertIcon className="mt-0.5 h-3 w-3 shrink-0 text-warn/70" />
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 flex items-center gap-1.5 border-t border-line pt-2 text-2xs text-slate-500">
              <CheckIcon className="h-3 w-3 text-ok" />
              硬件探测完成，未发现异常
            </p>
          )}
        </div>
      )}
    </section>
  )
}
