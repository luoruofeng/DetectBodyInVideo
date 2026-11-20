import type { HardwareReport } from '../api/hardware'
import type { ApiStatus, HealthInfo } from '../hooks/useApiHealth'
import type { Theme } from '../hooks/useTheme'
import { AccelerationBadge } from './AccelerationBadge'
import { DownloadIcon, MoonIcon, PoseIcon, SunIcon } from './icons'

interface StatusPillProps {
  status: ApiStatus
  health: HealthInfo | null
}

function ApiStatusPill({ status, health }: StatusPillProps) {
  if (status === 'checking') {
    return (
      <span className="chip border-line bg-white/[0.03] text-slate-400">
        <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-slate-400" />
        正在连接本地服务
      </span>
    )
  }
  if (status === 'offline') {
    return (
      <span className="chip border-danger/40 bg-danger/10 text-danger" title="后端未启动">
        <span className="h-1.5 w-1.5 rounded-full bg-danger" />
        服务未连接
      </span>
    )
  }
  return (
    <span className="chip border-line bg-white/[0.03] text-slate-400" title={health?.platform}>
      <span className="h-1.5 w-1.5 rounded-full bg-ok" />
      服务已连接
    </span>
  )
}

interface AppHeaderProps {
  apiStatus: ApiStatus
  health: HealthInfo | null
  hardware: HardwareReport | null
  hardwareLoading: boolean
  hardwareError: string | null
  onRefreshHardware: () => void
  theme: Theme
  onToggleTheme: () => void
  canExport: boolean
  exporting: boolean
  onExport: () => void
}

export function AppHeader({
  apiStatus,
  health,
  hardware,
  hardwareLoading,
  hardwareError,
  onRefreshHardware,
  theme,
  onToggleTheme,
  canExport,
  exporting,
  onExport,
}: AppHeaderProps) {
  return (
    <header className="flex shrink-0 items-center justify-between gap-4 border-b border-line bg-panel/60 px-5 py-3 backdrop-blur">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-accent/10 ring-1 ring-inset ring-accent/30">
          <PoseIcon className="h-5 w-5 text-accent" />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-base font-semibold tracking-tight text-slate-100">
            DetectBodyInVideo
          </h1>
          <p className="truncate text-2xs text-slate-500">
            视频人体关键点 → RealityKit JSON
            {health ? ` · v${health.version}` : ''}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden sm:block">
          <ApiStatusPill status={apiStatus} health={health} />
        </div>
        <div className="hidden md:block">
          <AccelerationBadge
            report={hardware}
            loading={hardwareLoading}
            error={hardwareError}
            onRefresh={onRefreshHardware}
          />
        </div>
        <button
          type="button"
          className="btn-icon"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'}
          title={theme === 'dark' ? '浅色主题（T）' : '深色主题（T）'}
        >
          {theme === 'dark' ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
        </button>
        <button
          type="button"
          className="btn-primary px-3 py-2"
          onClick={onExport}
          disabled={!canExport || exporting}
          title={canExport ? '导出 JSON（E）' : '完成识别后可导出'}
        >
          <DownloadIcon className="h-4 w-4" />
          <span className="hidden sm:inline">{exporting ? '导出中…' : '导出 JSON'}</span>
        </button>
      </div>
    </header>
  )
}
