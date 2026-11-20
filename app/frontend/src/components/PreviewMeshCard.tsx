import type { PreviewFrame } from '../api/analysis'
import { formatTime } from '../lib/pose'
import { ThreeMeshPreview } from './ThreeMeshPreview'

interface PreviewMeshCardProps {
  frame: PreviewFrame | null
  showGrid: boolean
  onToggleGrid: (value: boolean) => void
  theme: 'dark' | 'light'
}

export function PreviewMeshCard({ frame, showGrid, onToggleGrid, theme }: PreviewMeshCardProps) {
  const joints = frame ? Object.keys(frame.joints).length : 0

  return (
    <section className="card">
      <header className="card-title">
        <span>Mesh 白膜预览</span>
        <label className="flex cursor-pointer items-center gap-1.5 font-normal normal-case tracking-normal text-slate-500">
          <input
            type="checkbox"
            className="h-3 w-3 accent-pink-400"
            checked={showGrid}
            onChange={(event) => onToggleGrid(event.target.checked)}
          />
          网格
        </label>
      </header>

      <div className="space-y-2 p-3">
        <ThreeMeshPreview frame={frame} showGrid={showGrid} theme={theme} />

        <div className="flex items-center justify-between font-mono text-2xs text-slate-500">
          <span>
            {frame ? `t=${formatTime(frame.t)} · ${joints} 关节` : '等待分析结果'}
          </span>
          <span>Three.js · Marching Cubes</span>
        </div>
        <p className="text-2xs leading-relaxed text-slate-600">
          关键点按人体比例展开为带胸腔与四肢的人形隐式曲面，头颅与五指按解剖结构单独建模；白膜为姿态驱动的近似体型，不代表真实外观。
        </p>
      </div>
    </section>
  )
}
