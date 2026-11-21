import { useMemo, useState } from 'react'

import type { AnalysisOptions, EngineKind, JobStatus } from '../api/analysis'
import type { VideoMetadata } from '../api/videos'
import type { UseAnalysis } from '../hooks/useAnalysis'
import { formatCompactNumber, formatTime } from '../lib/pose'
import { ProgressBar } from './ProgressBar'
import { AlertIcon, BoneIcon, SparkIcon } from './icons'

interface AnalysisPanelProps {
  video: VideoMetadata | null
  analysis: UseAnalysis
  disabled: boolean
  onStart: (options: AnalysisOptions) => void
}

const VERDICT_STYLE: Record<string, string> = {
  ready: 'border-ok/40 bg-ok/10 text-ok',
  review: 'border-warn/40 bg-warn/10 text-warn',
  rerecord: 'border-danger/40 bg-danger/10 text-danger',
}

const DEPTH_TOOLTIP =
  '深度（前后方向）由单目模型估计，x/y 来自 2D 检测可信，z 只是估计值：绝对值与前后位移不可信。' +
  '后期无法修复——想提高这个分数只能重拍（正侧机位、避免大幅俯仰与快速走近镜头）。'

function depthTone(confidence: number): { text: string; bar: string; label: string } {
  if (confidence >= 0.75) return { text: 'text-ok', bar: 'bg-ok', label: '可信' }
  if (confidence >= 0.5) return { text: 'text-warn', bar: 'bg-warn', label: '一般' }
  return { text: 'text-danger', bar: 'bg-danger', label: '不可信' }
}

function ProgressBlock({ status }: { status: JobStatus | null }) {
  if (!status) return null
  const { progress, phase } = status
  const active = phase === 'queued' || phase === 'running'
  // 推理帧数在后处理开始前就已满；此时用后处理阶段进度，避免显示假的 100%。
  const useStage = active && progress.stage !== '' && progress.stage !== 'inference'
  const done = useStage ? progress.stage_done : progress.frames_done
  const total = Math.max(1, useStage ? progress.stage_total : progress.frames_total)
  const percent = Math.min(100, Math.round((done / total) * 100))
  const label =
    phase === 'done'
      ? '完成'
      : phase === 'cancelled'
        ? '已取消'
        : phase === 'error'
          ? '失败'
          : progress.message || (phase === 'queued' ? '排队中' : '识别中')

  return (
    <div className="space-y-2 rounded-xl border border-line bg-white/[0.02] px-3 py-2.5">
      <div className="flex items-center justify-between text-2xs">
        <span className={active ? 'text-accent' : phase === 'error' ? 'text-danger' : 'text-slate-300'}>
          {label}
          {active && status.canPreview && (
            <span className="ml-2 rounded bg-ok/10 px-1.5 py-0.5 text-ok">预览已可用</span>
          )}
        </span>
        <span className="font-mono text-slate-400">
          {useStage
            ? `${done}/${total} ${progress.stage === 'ik_fit' ? '帧' : '步'} · ${percent}%`
            : `${progress.frames_done}/${progress.frames_total || '?'} 帧 · ${percent}%`}
        </span>
      </div>
      <ProgressBar
        value={done / total}
        label="分析进度"
        tone={phase === 'error' ? 'danger' : 'accent'}
        indeterminate={active && done === 0}
      />
      <div className="flex justify-between font-mono text-2xs text-slate-500">
        <span>{progress.processed_fps ? `${progress.processed_fps.toFixed(1)} fps` : '—'}</span>
        <span>
          {progress.eta_seconds !== null && active
            ? `剩余 ≈ ${formatTime(progress.eta_seconds)}`
            : status.elapsedSeconds
              ? `耗时 ${formatTime(status.elapsedSeconds)}`
              : ''}
        </span>
      </div>
    </div>
  )
}

function QualityBlock({ status, loading }: { status: JobStatus | null; loading: boolean }) {
  const summary = status?.summary
  if (loading) {
    return (
      <div className="space-y-2 p-4" aria-label="正在加载质检结果">
        <div className="skeleton-block h-3 w-2/3" />
        <div className="skeleton-block h-3 w-1/2" />
        <div className="skeleton-block h-16 w-full" />
      </div>
    )
  }
  if (!summary) {
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <BoneIcon className="h-6 w-6 text-slate-600" />
        <p className="text-xs text-slate-500">完成一次识别后，这里会显示数据质检结果</p>
      </div>
    )
  }

  const quality = summary.quality
  // 兼容旧后端（1.0.0 schema 没有深度专项字段）：缺字段时按 0 处理，不让 UI 变成 NaN
  const depthConfidence = Number.isFinite(quality.depth_confidence) ? quality.depth_confidence : 0
  const zJitter = Number.isFinite(quality.z_jitter_rms) ? quality.z_jitter_rms : 0
  const boneCv = Number.isFinite(quality.bone_length_cv) ? quality.bone_length_cv : 0
  const reproj = Number.isFinite(quality.mean_reproj_error) ? quality.mean_reproj_error : 0
  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`chip ${VERDICT_STYLE[quality.verdict] ?? VERDICT_STYLE.rerecord}`}
          title={quality.issues.join('\n') || '未发现问题'}
        >
          {quality.verdict === 'ready' ? '可交付' : quality.verdict === 'review' ? '需复核' : '建议重录'}
        </span>
        <span className="text-2xs text-slate-500">{quality.verdict_label}</span>
        <span className="ml-auto font-mono text-2xs text-slate-500">
          {summary.tracks.length} 条轨迹 · 主轨迹 {summary.primaryTrackId ?? '—'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {[
          ['有效帧', `${(quality.valid_frame_ratio * 100).toFixed(1)}%`],
          ['平均置信度', quality.mean_score.toFixed(3)],
          ['平均可见度', quality.mean_visibility.toFixed(3)],
          ['最大跳变', `${quality.max_jump.toFixed(3)} m`],
        ].map(([key, value]) => (
          <div key={key} className="rounded-lg border border-line bg-white/[0.02] px-2.5 py-2">
            <div className="text-2xs text-slate-500">{key}</div>
            <div className="mt-0.5 font-mono text-sm text-slate-200">{value}</div>
          </div>
        ))}
      </div>

      <div
        className="space-y-1.5 rounded-lg border border-line bg-white/[0.02] px-2.5 py-2"
        title={DEPTH_TOOLTIP}
      >
        <div className="flex items-center justify-between text-2xs">
          <span className="flex items-center gap-1 text-slate-400">
            深度置信度
            <span className="cursor-help text-slate-600">ⓘ</span>
          </span>
          <span className={`font-mono ${depthTone(depthConfidence).text}`}>
            {(depthConfidence * 100).toFixed(0)}% · {depthTone(depthConfidence).label}
          </span>
        </div>
        <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className={`h-full rounded-full ${depthTone(depthConfidence).bar}`}
            style={{ width: `${Math.min(100, Math.max(0, depthConfidence * 100))}%` }}
          />
        </div>
        <div className="flex flex-wrap gap-x-3 font-mono text-2xs text-slate-600">
          <span>深度抖动 {(zJitter * 100).toFixed(1)} cm/帧</span>
          <span>骨长波动 {(boneCv * 100).toFixed(1)}%</span>
          <span>重投影 {reproj.toFixed(4)}</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-slate-500">
        <span>总帧 {formatCompactNumber(quality.frames_total)}</span>
        <span>插值 {quality.interpolated_frames}</span>
        {quality.max_jump_frame !== null ? <span>跳变帧 #{quality.max_jump_frame}</span> : null}
        {summary.depthSign?.applied ? (
          <span title="深度方向（每骨段 1 个比特）的帧间翻转次数：平滑前后对比">
            方向翻转 {summary.depthSign.flipsBefore} → {summary.depthSign.flipsAfter}
          </span>
        ) : null}
        <span>处理 {summary.elapsedSeconds ? `${summary.elapsedSeconds.toFixed(1)}s` : '—'}</span>
      </div>

      {quality.issues.length ? (
        <ul className="space-y-1 rounded-lg border border-warn/20 bg-warn/[0.06] p-2.5">
          {quality.issues.map((issue) => (
            <li key={issue} className="flex items-start gap-1.5 text-2xs text-warn">
              <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
              <span>{issue}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {summary.notes.length ? (
        <details className="text-2xs text-slate-500">
          <summary className="cursor-pointer select-none text-slate-400">运行说明（{summary.notes.length}）</summary>
          <ul className="mt-1.5 space-y-1">
            {summary.notes.map((note) => (
              <li key={note}>· {note}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  )
}

/** 右侧识别参数 + 进度 + 质检。 */
export function AnalysisPanel({ video, analysis, disabled, onStart }: AnalysisPanelProps) {
  const [engine, setEngine] = useState<EngineKind>('auto')
  const [everyN, setEveryN] = useState(1)
  const [targetFps, setTargetFps] = useState('')
  const [rootMotion, setRootMotion] = useState<'world' | 'in_place'>('world')
  const [batch, setBatch] = useState(4)
  const [resume, setResume] = useState(true)
  const [ikFit, setIkFit] = useState(true)
  // 深度方向修正默认关闭：它需要"时序证据"（≥30 帧）且会改变既有输出，属于可选项
  const [depthSign, setDepthSign] = useState(false)

  const running = analysis.phase === 'queued' || analysis.phase === 'running'
  const selection = analysis.status?.selection
  const errorDetails = analysis.errorDetails
  const errorReasons = Array.isArray(errorDetails?.reasons)
    ? (errorDetails.reasons as unknown[]).filter(
        (item): item is string => typeof item === 'string' && item.length > 0,
      )
    : []
  const errorHint = typeof errorDetails?.hint === 'string' ? errorDetails.hint : ''
  // 骨长约束拟合必须要有 world landmarks（米制），引擎不支持时置灰
  const fitAvailable = !selection || selection.capabilities.world_landmarks
  const estimatedFrames = useMemo(() => {
    if (!video) return 0
    const fpsTarget = Number(targetFps)
    const every = targetFps && fpsTarget > 0 ? Math.max(1, Math.round(video.fps / fpsTarget)) : everyN
    return Math.ceil(video.frame_count / Math.max(1, every))
  }, [everyN, targetFps, video])

  /** 组装本次识别参数；``resumeOverride`` 用于「清空缓存」强制从头识别。 */
  const buildOptions = (resumeOverride?: boolean): AnalysisOptions => ({
    engine,
    sampling: {
      every_n_frames: everyN,
      target_fps: targetFps && Number(targetFps) > 0 ? Number(targetFps) : null,
      trim_empty: false,
    },
    skeleton: {
      root_motion_mode: rootMotion,
      include_optional: true,
      include_all_joints: true,
    },
    refinement: { hands: true, face: true },
    perf: {
      batch_size: batch,
      resume: resumeOverride ?? resume,
      // 是否继续写断点仍跟随开关；清缓存只影响"本次是否复用旧断点"
      checkpoint: resume,
    },
    ikFit: { enabled: ikFit && fitAvailable },
    postprocess: { depth_sign: { enabled: depthSign && fitAvailable, mode: 'temporal' } },
    max_frames: null,
  })

  const submit = () => onStart(buildOptions())

  /** 去掉缓存：先删后端断点文件，再用同一视频从第 1 帧重新识别。 */
  const restartFromScratch = async () => {
    const removed = await analysis.clearCache()
    if (removed === null) return
    onStart(buildOptions(false))
  }

  return (
    <div className="space-y-3">
      <section className="card">
        <header className="card-title">
          <span className="flex items-center gap-2">
            <SparkIcon className="h-3.5 w-3.5 text-accent" />
            识别设置
          </span>
          {selection ? (
            <span
              className="chip border-accent/30 bg-accent/10 text-accent"
              title={selection.capabilities.providers.join(', ') || selection.display_name}
            >
              {selection.display_name}
              {selection.degraded ? ' · 降级' : ''}
            </span>
          ) : null}
        </header>

        <div className="space-y-3 p-4">
          <div className="grid grid-cols-2 gap-2.5">
            <label className="space-y-1">
              <span className="label">推理引擎</span>
              <select
                className="field"
                value={engine}
                onChange={(event) => setEngine(event.target.value as EngineKind)}
                disabled={running}
              >
                <option value="auto">自动选择</option>
                <option value="mediapipe">MediaPipe</option>
                <option value="onnx">ONNX（RTMPose）</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="label">采样间隔</span>
              <select
                className="field"
                value={everyN}
                onChange={(event) => setEveryN(Number(event.target.value))}
                disabled={running}
              >
                {[1, 2, 3, 5, 10].map((value) => (
                  <option key={value} value={value}>
                    每 {value} 帧
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1">
              <span className="label">目标帧率（可空）</span>
              <input
                className="field font-mono"
                inputMode="decimal"
                placeholder="如 15"
                value={targetFps}
                onChange={(event) => setTargetFps(event.target.value.replace(/[^\d.]/g, ''))}
                disabled={running}
              />
            </label>
            <label className="space-y-1">
              <span className="label">根运动</span>
              <select
                className="field"
                value={rootMotion}
                onChange={(event) => setRootMotion(event.target.value as 'world' | 'in_place')}
                disabled={running}
              >
                <option value="world">保留位移（world）</option>
                <option value="in_place">原地（in_place）</option>
              </select>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-white/[0.02] px-3 py-2">
            <label className="flex items-center gap-1.5 text-2xs text-slate-400">
              批处理
              <select
                value={batch}
                onChange={(event) => setBatch(Number(event.target.value))}
                disabled={running}
                className="rounded-md border border-line bg-elevated px-1.5 py-0.5 text-2xs text-slate-200"
              >
                {[1, 2, 4, 8].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex cursor-pointer items-center gap-1.5 text-2xs text-slate-400">
              <input
                type="checkbox"
                className="h-3 w-3 accent-accent"
                checked={resume}
                onChange={(event) => setResume(event.target.checked)}
                disabled={running}
              />
              断点续算
            </label>
            <label
              className={`flex items-center gap-1.5 text-2xs ${
                fitAvailable ? 'cursor-pointer text-slate-400' : 'cursor-not-allowed text-slate-600'
              }`}
              title={
                fitAvailable
                  ? '固定骨长 + 2D 重投影拟合：把世界坐标约束到与画面 2D 关键点一致，骨长严格恒定。单目视频的深度只是估计值，这一步减少它对关节角度的污染。'
                  : '当前引擎没有输出米制世界坐标（world landmarks），无法做骨长约束拟合'
              }
            >
              <input
                type="checkbox"
                className="h-3 w-3 accent-accent"
                checked={ikFit && fitAvailable}
                onChange={(event) => setIkFit(event.target.checked)}
                disabled={running || !fitAvailable}
              />
              骨长约束拟合
            </label>
            <label
              className={`flex items-center gap-1.5 text-2xs ${
                fitAvailable ? 'cursor-pointer text-slate-400' : 'cursor-not-allowed text-slate-600'
              }`}
              title={
                fitAvailable
                  ? '深度方向修正：单目下每根骨段只剩「往前还是往后」一个比特，逐帧由引擎 z 决定会来回翻（前踢腿被解成后踢腿）。开启后用整段动作的时序一致性把方向定住，只在符号反了时纠正，不影响 2D 与骨长。至少需要 30 帧。'
                  : '当前引擎没有输出米制世界坐标（world landmarks），无法判断深度方向'
              }
            >
              <input
                type="checkbox"
                className="h-3 w-3 accent-accent"
                checked={depthSign && fitAvailable}
                onChange={(event) => setDepthSign(event.target.checked)}
                disabled={running || !fitAvailable}
              />
              深度方向修正
            </label>
            <span className="ml-auto text-2xs text-slate-500">
              预计 {estimatedFrames ? `${formatCompactNumber(estimatedFrames)} 帧` : '—'}
            </span>
          </div>

          {analysis.error ? (
            <div className="space-y-1.5 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-2xs text-danger">
              <div className="flex items-start gap-2">
                <AlertIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span className="break-all">{analysis.error}</span>
              </div>
              {errorReasons.length ? (
                <ul className="ml-5 list-disc space-y-0.5 text-slate-400">
                  {errorReasons.map((reason) => (
                    <li key={reason} className="break-all">
                      {reason}
                    </li>
                  ))}
                </ul>
              ) : null}
              {errorHint ? <p className="ml-5 break-all text-slate-400">{errorHint}</p> : null}
            </div>
          ) : null}

          <ProgressBlock status={analysis.status} />

          <div className="flex items-center gap-2">
            {running ? (
              <button
                type="button"
                className="btn-ghost flex-1 border-danger/30 text-danger hover:bg-danger/10"
                onClick={() => void analysis.cancel()}
              >
                取消任务
              </button>
            ) : (
              <>
                <button
                  type="button"
                  className="btn-primary flex-1"
                  onClick={submit}
                  disabled={disabled || !video}
                >
                  {analysis.phase === 'done' ? '重新识别' : '开始识别'}
                </button>
                <button
                  type="button"
                  className="btn-ghost shrink-0"
                  onClick={() => void restartFromScratch()}
                  disabled={disabled || !video}
                  title="删除断点续算缓存，用当前视频从第 1 帧重新识别"
                >
                  清空缓存
                </button>
              </>
            )}
          </div>
          {analysis.phase === 'done' ? (
            <p className="text-center text-2xs text-slate-500">
              已识别 {analysis.status?.summary?.frames ?? 0} 帧 ·{' '}
              {analysis.status?.summary?.elapsedSeconds.toFixed(1)}s
              {analysis.framesLoading
                ? ` · 骨架加载中 ${analysis.framesLoaded}/${analysis.frames?.totalFrames ?? '…'}`
                : ''}
            </p>
          ) : null}
        </div>
      </section>

      <section className="card">
        <header className="card-title">
          <span className="flex items-center gap-2">
            <BoneIcon className="h-3.5 w-3.5 text-accent" />
            数据质检
          </span>
          {analysis.status?.summary ? (
            <span className="font-mono text-2xs text-slate-500">
              {analysis.status.summary.frames} 帧
            </span>
          ) : null}
        </header>
        <QualityBlock
          status={analysis.status}
          loading={
            analysis.framesLoading ||
            ((analysis.phase === 'running' || analysis.phase === 'queued') && !analysis.status?.summary)
          }
        />
      </section>
    </div>
  )
}
