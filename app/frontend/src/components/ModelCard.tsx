import type { ModelFileInfo } from '../api/models'
import type { UseModels } from '../hooks/useModels'
import { formatBytes } from '../lib/format'
import { ProgressBar } from './ProgressBar'
import { AlertIcon, CheckIcon, DownloadIcon } from './icons'

interface ModelCardProps {
  models: UseModels
  /** 当前硬件推荐的姿态模型档位（lite/full/heavy） */
  recommendedTier?: string | null
}

const TIER_LABELS: Record<string, string> = {
  lite: '精简',
  full: '完整',
  heavy: '高精度',
}

function TierRow({
  entry,
  recommended,
  busy,
  onDownload,
}: {
  entry: ModelFileInfo
  recommended: boolean
  busy: boolean
  onDownload: () => void
}) {
  const label = entry.tier ? (TIER_LABELS[entry.tier] ?? entry.tier) : entry.filename
  return (
    <div className="flex items-center justify-between gap-2 rounded-lg border border-line bg-white/[0.02] px-2.5 py-2">
      <div className="min-w-0">
        <div className="flex items-center gap-1.5 text-xs text-slate-200">
          {entry.present ? (
            <CheckIcon className="h-3 w-3 text-ok" />
          ) : (
            <span className="h-1.5 w-1.5 rounded-full bg-slate-600" />
          )}
          {label}
          {recommended ? (
            <span className="chip border-accent/30 bg-accent/10 text-accent">推荐</span>
          ) : null}
        </div>
        <div className="truncate font-mono text-2xs text-slate-500">
          {entry.present ? `${formatBytes(entry.size_bytes)} · ${entry.filename}` : entry.filename}
        </div>
      </div>
      {entry.present ? (
        <span className="shrink-0 text-2xs text-ok">已就绪</span>
      ) : (
        <button
          type="button"
          className="btn-ghost shrink-0 px-2 py-1 text-2xs"
          disabled={busy}
          onClick={onDownload}
        >
          <DownloadIcon className="h-3 w-3" />
          下载
        </button>
      )}
    </div>
  )
}

/** 首启检查：缺少姿态模型时提供带进度和镜像的下载入口（任务 10.4）。 */
export function ModelCard({ models, recommendedTier }: ModelCardProps) {
  const { payload, active, error, mirror, setMirror, download } = models

  if (!payload) return null

  const poseModels = payload.models.filter((item) => item.kind === 'pose')
  const recommendedEntry = poseModels.find((item) => item.tier === recommendedTier)
  const recommendedMissing = Boolean(recommendedEntry && !recommendedEntry.present)
  const ready = payload.poseReady
  const downloading = Boolean(active && (active.state === 'running' || active.state === 'queued'))
  if (ready && !recommendedMissing && !downloading && !error) return null

  const ordered = ['lite', 'full', 'heavy']
    .map((tier) => poseModels.find((item) => item.tier === tier))
    .filter((item): item is ModelFileInfo => Boolean(item))

  const totalMb = active && active.total > 0 ? active.total / 1024 ** 2 : 0
  const downloadedMb = active ? active.downloaded / 1024 ** 2 : 0

  return (
    <section className="card border-warn/30">
      <header className="card-title">
        <span className="flex items-center gap-2 text-warn">
          <AlertIcon className="h-3.5 w-3.5" />
          {ready ? '推荐档位模型未就绪' : '首次运行：需要姿态模型'}
        </span>
      </header>

      <div className="space-y-2.5 p-4">
        <p className="text-2xs leading-relaxed text-slate-400">
          识别功能依赖 MediaPipe 姿态模型（每个档位约 5–30 MB）。下载后保存在本机
          <span className="font-mono"> ~/.detect-body/models</span>，不会上传任何数据。
          {recommendedMissing && recommendedEntry ? (
            <span className="block text-warn">
              当前硬件推荐 {TIER_LABELS[recommendedEntry.tier ?? ''] ?? recommendedEntry.tier} 档，
              但缺少 {recommendedEntry.filename}；未下载时会临时回退到已有档位。
            </span>
          ) : null}
        </p>

        <div className="space-y-1.5">
          {ordered.map((entry) => (
            <TierRow
              key={entry.filename}
              entry={entry}
              recommended={entry.filename === recommendedEntry?.filename || entry.tier === 'full'}
              busy={downloading}
              onDownload={() => void download(entry.tier ?? 'full')}
            />
          ))}
        </div>

        <label className="flex cursor-pointer items-start gap-2 text-2xs text-slate-400">
          <input
            type="checkbox"
            className="mt-0.5 h-3.5 w-3.5 accent-accent"
            checked={mirror}
            onChange={(event) => setMirror(event.target.checked)}
            disabled={downloading}
          />
          <span>
            使用镜像下载
            <span className="block text-slate-600">
              {payload.mirrorConfigured
                ? '已通过 DETECT_BODY_MODEL_MIRROR_URL 配置镜像前缀'
                : '服务端未配置镜像，勾选后仍走官方直链'}
            </span>
          </span>
        </label>

        {active ? (
          <div className="space-y-1.5 rounded-lg border border-line bg-white/[0.02] px-2.5 py-2">
            <div className="flex items-center justify-between text-2xs">
              <span className={active.state === 'error' ? 'text-danger' : 'text-slate-300'}>
                {active.state === 'error'
                  ? '下载失败'
                  : active.state === 'done'
                    ? '下载完成'
                    : '正在下载'}
                {active.mirror ? ' · 镜像' : ''}
              </span>
              <span className="font-mono text-slate-500">
                {totalMb > 0
                  ? `${downloadedMb.toFixed(1)} / ${totalMb.toFixed(1)} MB · ${active.percent.toFixed(0)}%`
                  : `${downloadedMb.toFixed(1)} MB`}
              </span>
            </div>
            <ProgressBar
              value={active.total > 0 ? active.downloaded / active.total : 0}
              label="模型下载进度"
              tone={active.state === 'error' ? 'danger' : 'accent'}
              indeterminate={active.state === 'running' && active.total === 0}
            />
            <div className="truncate font-mono text-2xs text-slate-600">{active.filename}</div>
            {active.error ? <p className="break-all text-2xs text-danger">{active.error}</p> : null}
          </div>
        ) : null}

        {error ? (
          <div className="flex items-start gap-1.5 text-2xs text-danger">
            <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
            <span className="break-all">{error}</span>
          </div>
        ) : null}

        <p className="text-2xs leading-relaxed text-slate-600">
          模型来自 MediaPipe 官方直链；若网络不可达，可设置{' '}
          <span className="font-mono">DETECT_BODY_MODEL_MIRROR_URL</span> 为可用镜像前缀。
        </p>
      </div>
    </section>
  )
}
