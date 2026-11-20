import type { VideoMetadata } from '../api/videos'
import { formatBytes, formatCount, formatDuration, formatFps } from '../lib/format'
import { AlertIcon, FilmIcon } from './icons'

interface VideoCardProps {
  video: VideoMetadata
  onReplace: () => void
}

/** 视频信息（采样参数在 AnalysisPanel 中统一配置，避免两处不一致）。 */
export function VideoCard({ video, onReplace }: VideoCardProps) {
  return (
    <section className="card">
      <h2 className="card-title">
        <span className="flex items-center gap-2">
          <FilmIcon className="h-3.5 w-3.5 text-slate-500" />
          视频信息
        </span>
        <button
          type="button"
          onClick={onReplace}
          className="rounded-md px-1.5 py-0.5 text-2xs font-normal normal-case tracking-normal text-slate-500 transition hover:bg-white/5 hover:text-slate-300"
        >
          更换
        </button>
      </h2>

      <div className="space-y-1 px-4 py-3">
        <div className="kv">
          <span className="kv-key">文件名</span>
          <span className="kv-value" title={video.filename}>
            {video.filename}
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">分辨率</span>
          <span className="kv-value">
            {video.display_width}×{video.display_height}
            {video.rotation ? `（旋转 ${video.rotation}°）` : ''}
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">帧率</span>
          <span className="kv-value">
            {formatFps(video.fps)}
            {video.is_variable_frame_rate ? ' · VFR' : ''}
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">帧数 / 时长</span>
          <span className="kv-value">
            {formatCount(video.frame_count)} / {formatDuration(video.duration_s)}
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">编码</span>
          <span className="kv-value">{(video.codec ?? '未知').toUpperCase()}</span>
        </div>
        <div className="kv">
          <span className="kv-key">体积</span>
          <span className="kv-value">{formatBytes(video.size_bytes)}</span>
        </div>

        {video.warnings.length ? (
          <ul className="mt-2 space-y-1 border-t border-line pt-2">
            {video.warnings.map((warning) => (
              <li key={warning} className="flex gap-1.5 text-2xs leading-relaxed text-warn">
                <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  )
}
