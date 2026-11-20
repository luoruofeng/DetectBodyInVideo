import type { PreviewFrame } from '../api/analysis'
import { formatTime } from '../lib/pose'
import { ThreePreview } from './ThreePreview'

interface Preview3DCardProps {
  frame: PreviewFrame | null
  showGrid: boolean
  onToggleGrid: (value: boolean) => void
  theme: 'dark' | 'light'
}

const ANGLE_LABELS: Record<string, string> = {
  left_elbow: '左肘',
  right_elbow: '右肘',
  left_shoulder: '左肩',
  right_shoulder: '右肩',
  left_knee: '左膝',
  right_knee: '右膝',
  left_hip: '左髋',
  right_hip: '右髋',
  torso_lean: '躯干前倾',
}

const ANGLE_ORDER = Object.keys(ANGLE_LABELS)

export function Preview3DCard({ frame, showGrid, onToggleGrid, theme }: Preview3DCardProps) {
  const angles = frame?.angles ?? {}
  const available = ANGLE_ORDER.filter((name) => angles[name] !== undefined)
  const joints = frame ? Object.keys(frame.joints).length : 0

  return (
    <section className="card">
      <header className="card-title">
        <span>3D 预览</span>
        <label className="flex cursor-pointer items-center gap-1.5 font-normal normal-case tracking-normal text-slate-500">
          <input
            type="checkbox"
            className="h-3 w-3 accent-accent"
            checked={showGrid}
            onChange={(event) => onToggleGrid(event.target.checked)}
          />
          网格
        </label>
      </header>

      <div className="space-y-2 p-3">
        <ThreePreview frame={frame} showGrid={showGrid} theme={theme} />

        <div className="flex items-center justify-between font-mono text-2xs text-slate-500">
          <span>
            {frame ? `t=${formatTime(frame.t)} · ${joints} 关节` : '等待分析结果'}
          </span>
          {frame?.interpolated ? <span className="text-warn">插值帧</span> : null}
        </div>

        {available.length ? (
          <div className="grid grid-cols-3 gap-1">
            {available.map((name) => (
              <div
                key={name}
                className="rounded-md border border-line bg-white/[0.02] px-1.5 py-1 text-center"
              >
                <div className="text-2xs text-slate-500">{ANGLE_LABELS[name]}</div>
                <div className="font-mono text-2xs text-slate-200">
                  {angles[name].toFixed(1)}°
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-2xs text-slate-600">当前帧没有可用的关节角度（关键点缺失）</p>
        )}
      </div>
    </section>
  )
}
