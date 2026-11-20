interface ProgressBarProps {
  value: number
  label?: string
  tone?: 'accent' | 'danger'
  indeterminate?: boolean
  className?: string
}

/** 统一进度条（上传 / 分析共用，任务 8.2）。 */
export function ProgressBar({
  value,
  label,
  tone = 'accent',
  indeterminate = false,
  className = '',
}: ProgressBarProps) {
  const percent = Math.min(100, Math.max(0, Math.round(value * 100)))
  return (
    <div
      className={`h-1.5 w-full overflow-hidden rounded-full bg-white/10 ${className}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : percent}
    >
      <div
        className={`h-full rounded-full transition-[width] duration-200 ${
          tone === 'danger' ? 'bg-danger' : 'bg-accent'
        } ${indeterminate ? 'animate-pulse-soft' : ''}`}
        style={{ width: indeterminate ? '100%' : `${Math.max(2, percent)}%` }}
      />
    </div>
  )
}
