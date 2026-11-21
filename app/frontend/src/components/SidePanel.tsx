import { useState } from 'react'
import type { CSSProperties } from 'react'

import type { AnalysisOptions, PreviewFrame } from '../api/analysis'
import type { HardwareReport } from '../api/hardware'
import type { VideoMetadata } from '../api/videos'
import type { ApiStatus, HealthInfo } from '../hooks/useApiHealth'
import type { UseAnalysis } from '../hooks/useAnalysis'
import type { UseModels } from '../hooks/useModels'
import { AnalysisPanel } from './AnalysisPanel'
import { ExportCard } from './ExportCard'
import { HardwareCard } from './HardwareCard'
import { ModelCard } from './ModelCard'
import { Preview3DCard } from './Preview3DCard'
import { PreviewMeshCard } from './PreviewMeshCard'
import { ShootingGuideCard } from './ShootingGuide'
import { VideoCard } from './VideoCard'
import { CheckIcon } from './icons'

function EnvCard({ health, apiStatus }: { health: HealthInfo | null; apiStatus: ApiStatus }) {
  return (
    <section className="card">
      <h2 className="card-title">
        运行环境
        <span className="text-2xs font-normal normal-case tracking-normal text-slate-500">
          {apiStatus === 'online' ? '本地服务' : '未连接'}
        </span>
      </h2>
      <div className="space-y-1 px-4 py-3">
        <div className="kv">
          <span className="kv-key">版本</span>
          <span className="kv-value">{health?.version ?? '—'}</span>
        </div>
        <div className="kv">
          <span className="kv-key">Python</span>
          <span className="kv-value">{health?.python ?? '—'}</span>
        </div>
        <div className="kv">
          <span className="kv-key">系统</span>
          <span className="kv-value" title={health?.platform}>
            {health?.platform ?? '—'}
          </span>
        </div>
      </div>
    </section>
  )
}

const STEPS = [
  { title: '导入视频', desc: '拖拽或选择文件，自动解析帧率与分辨率' },
  { title: '识别关键点', desc: '逐帧推理 33 点 + 世界坐标，可逐帧叠加骨架' },
  { title: '导出 JSON', desc: '按 ARKit 关节映射导出，直接喂给 RealityKit' },
]

function StepsCard() {
  return (
    <section className="card">
      <h2 className="card-title">使用流程</h2>
      <ol className="space-y-3 px-4 py-3">
        {STEPS.map((step, index) => (
          <li key={step.title} className="flex gap-3">
            <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border border-line bg-white/[0.03] text-2xs font-medium text-slate-400">
              {index + 1}
            </span>
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-300">{step.title}</p>
              <p className="mt-0.5 text-2xs leading-relaxed text-slate-500">{step.desc}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="flex items-center gap-1.5 border-t border-line px-4 py-2.5 text-2xs text-slate-500">
        <CheckIcon className="h-3 w-3 text-ok" />
        所有处理均在本机完成，视频不会上传云端
      </div>
    </section>
  )
}

interface SidePanelProps {
  /** 右侧栏宽度（px），仅 lg 及以上生效；窄屏仍是整列铺满 */
  width: number
  resizing?: boolean
  health: HealthInfo | null
  apiStatus: ApiStatus
  hardware: HardwareReport | null
  hardwareLoading: boolean
  hardwareError: string | null
  onRefreshHardware: () => void
  video: VideoMetadata | null
  onReplaceVideo: () => void
  analysis: UseAnalysis
  onStart: (options: AnalysisOptions) => void
  frame: PreviewFrame | null
  theme: 'dark' | 'light'
  models: UseModels
}

export function SidePanel({
  width,
  resizing = false,
  health,
  apiStatus,
  hardware,
  hardwareLoading,
  hardwareError,
  onRefreshHardware,
  video,
  onReplaceVideo,
  analysis,
  onStart,
  frame,
  theme,
  models,
}: SidePanelProps) {
  const [showGrid, setShowGrid] = useState(true)

  return (
    <aside
      className={`panel-scroll flex w-full min-w-0 shrink-0 flex-col gap-3 overflow-y-auto overscroll-contain pb-1 lg:w-[var(--side-panel-width)] ${
        resizing ? 'lg:select-none' : ''
      }`}
      style={{ '--side-panel-width': `${width}px` } as CSSProperties}
    >
      <ModelCard models={models} recommendedTier={hardware?.acceleration.model_tier} />
      <ShootingGuideCard />
      {video ? (
        <AnalysisPanel
          video={video}
          analysis={analysis}
          disabled={false}
          onStart={onStart}
          models={models}
        />
      ) : null}
      {video ? <ExportCard analysis={analysis} /> : null}
      {video ? (
        <Preview3DCard
          frame={frame}
          showGrid={showGrid}
          onToggleGrid={setShowGrid}
          theme={theme}
        />
      ) : null}
      {video ? (
        <PreviewMeshCard
          frame={frame}
          showGrid={showGrid}
          onToggleGrid={setShowGrid}
          theme={theme}
        />
      ) : null}
      {video ? <VideoCard video={video} onReplace={onReplaceVideo} /> : <EnvCard health={health} apiStatus={apiStatus} />}
      <HardwareCard
        report={hardware}
        loading={hardwareLoading}
        error={hardwareError}
        onRefresh={onRefreshHardware}
      />
      {video ? null : <StepsCard />}
    </aside>
  )
}
