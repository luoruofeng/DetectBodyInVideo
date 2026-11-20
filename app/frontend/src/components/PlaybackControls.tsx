import { formatTime } from '../lib/pose'
import { PauseIcon, PlayIcon, StepBackIcon, StepForwardIcon } from './icons'

interface PlaybackControlsProps {
  playing: boolean
  currentTime: number
  duration: number
  rate: number
  frameIndex: number
  totalFrames: number
  frameValid: boolean
  disabled: boolean
  onToggle: () => void
  onStep: (frames: number) => void
  onSetRate: (rate: number) => void
}

const RATES = [0.25, 0.5, 1, 1.5, 2]

export function PlaybackControls({
  playing,
  currentTime,
  duration,
  rate,
  frameIndex,
  totalFrames,
  frameValid,
  disabled,
  onToggle,
  onStep,
  onSetRate,
}: PlaybackControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex items-center gap-1">
        <button
          type="button"
          className="btn-icon"
          onClick={() => onStep(-1)}
          disabled={disabled}
          aria-label="上一帧"
          title="上一帧（←）"
        >
          <StepBackIcon className="h-4 w-4" />
        </button>
        <button
          type="button"
          className="btn-primary h-9 w-11 px-0"
          onClick={onToggle}
          disabled={disabled}
          aria-label={playing ? '暂停' : '播放'}
          title={playing ? '暂停（空格）' : '播放（空格）'}
        >
          {playing ? <PauseIcon className="h-4 w-4" /> : <PlayIcon className="h-4 w-4" />}
        </button>
        <button
          type="button"
          className="btn-icon"
          onClick={() => onStep(1)}
          disabled={disabled}
          aria-label="下一帧"
          title="下一帧（→）"
        >
          <StepForwardIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="rounded-lg border border-line bg-white/[0.02] px-2.5 py-1 font-mono text-2xs text-slate-400">
        <span className="text-slate-200">{formatTime(currentTime)}</span>
        <span className="mx-1 text-slate-600">/</span>
        <span>{formatTime(duration)}</span>
        <span className="mx-1.5 text-slate-700">|</span>
        帧 {frameIndex >= 0 ? frameIndex : '—'} / {totalFrames || '—'}
        {frameIndex >= 0 ? (
          <span className={frameValid ? 'ml-1.5 text-ok' : 'ml-1.5 text-danger'}>
            {frameValid ? '有效' : '缺失'}
          </span>
        ) : null}
      </div>

      <label className="flex items-center gap-1.5 text-2xs text-slate-500">
        倍速
        <select
          value={rate}
          onChange={(event) => onSetRate(Number(event.target.value))}
          disabled={disabled}
          className="field w-[74px] px-2 py-1 text-2xs"
          aria-label="播放倍速"
        >
          {RATES.map((value) => (
            <option key={value} value={value}>
              {value}×
            </option>
          ))}
        </select>
      </label>

      <span className="ml-auto text-2xs text-slate-600">
        空格播放 / ← → 逐帧 / Shift 加速
      </span>
    </div>
  )
}
