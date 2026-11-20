import type { ExportFormat, Granularity } from '../api/analysis'
import { schemaUrl } from '../api/analysis'
import type { UseAnalysis } from '../hooks/useAnalysis'
import { formatCompactNumber } from '../lib/pose'
import { CheckIcon, DownloadIcon } from './icons'

interface ExportCardProps {
  analysis: UseAnalysis
}

const FORMAT_OPTIONS: Array<{ value: ExportFormat; label: string; desc: string }> = [
  { value: 'pose', label: '姿态 JSON', desc: '本项目 schema v1.1.0：关键点 + 角度 + 根运动 + 深度可信度' },
  {
    value: 'motion',
    label: '动作 JSON',
    desc: 'iOS App 动作格式：10Hz、髋局部米制坐标，可直接驱动 USDZ 模型',
  },
]

const GRANULARITY_OPTIONS: Array<{ value: Granularity; label: string; desc: string }> = [
  { value: 'compact', label: '精简', desc: '关键帧 + 角度，便于快速预览' },
  { value: 'keypoints', label: '仅关键点', desc: '只有关键点与角度，无骨骼映射' },
  { value: 'full', label: '完整', desc: '完整骨骼 + 角度 + 元信息（推荐）' },
]

export function ExportCard({ analysis }: ExportCardProps) {
  const canExport = analysis.phase === 'done'
  const artifact = analysis.artifact
  const isMotion = analysis.format === 'motion'

  return (
    <section className="card">
      <header className="card-title">
        <span>导出</span>
        <a
          href={schemaUrl()}
          target="_blank"
          rel="noreferrer"
          className="rounded px-1.5 py-0.5 text-2xs font-normal normal-case tracking-normal text-slate-500 transition hover:bg-white/5 hover:text-accent"
          title="JSON Schema v1.1.0（可交给 iOS 侧做校验）"
        >
          JSON Schema
        </a>
      </header>

      <div className="space-y-3 p-4">
        <div className="grid grid-cols-2 gap-1.5" role="radiogroup" aria-label="导出格式">
          {FORMAT_OPTIONS.map((option) => {
            const active = analysis.format === option.value
            return (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={active}
                title={option.desc}
                onClick={() => analysis.setFormat(option.value)}
                className={`rounded-lg border px-2 py-1.5 text-2xs transition ${
                  active
                    ? 'border-accent/50 bg-accent/10 text-accent'
                    : 'border-line bg-white/[0.02] text-slate-400 hover:border-white/15 hover:text-slate-200'
                }`}
              >
                {option.label}
              </button>
            )
          })}
        </div>
        <p className="text-2xs leading-relaxed text-slate-500">
          {FORMAT_OPTIONS.find((option) => option.value === analysis.format)?.desc}
        </p>

        {isMotion ? (
          <div className="space-y-1 rounded-lg border border-line bg-white/[0.02] p-2.5 text-2xs leading-relaxed text-slate-400">
            <p>10Hz 重采样 · 17 个身体关节 · 米制髋局部坐标（pelvis 为原点，Y 向上）</p>
            <p className="text-slate-500">
              与 air-squat-motion.json 同格式：运行时只取关节方向，骨长沿用模型自身骨架。
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-1.5" role="radiogroup" aria-label="导出粒度">
              {GRANULARITY_OPTIONS.map((option) => {
                const active = analysis.granularity === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    title={option.desc}
                    onClick={() => analysis.setGranularity(option.value)}
                    className={`rounded-lg border px-2 py-1.5 text-2xs transition ${
                      active
                        ? 'border-accent/50 bg-accent/10 text-accent'
                        : 'border-line bg-white/[0.02] text-slate-400 hover:border-white/15 hover:text-slate-200'
                    }`}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
            <p className="text-2xs leading-relaxed text-slate-500">
              {GRANULARITY_OPTIONS.find((option) => option.value === analysis.granularity)?.desc}
            </p>
          </>
        )}

        <button
          type="button"
          className="btn-primary w-full"
          onClick={() =>
            void analysis.exportJson({
              include_raw_landmarks: !isMotion && analysis.granularity === 'full',
            })
          }
          disabled={!canExport || analysis.exporting}
        >
          <DownloadIcon className="h-4 w-4" />
          {analysis.exporting
            ? '正在导出…'
            : canExport
              ? isMotion
                ? '导出动作 JSON…'
                : '导出 JSON…'
              : '完成识别后可导出'}
        </button>
        <p className="text-2xs leading-relaxed text-slate-500">
          点击后会弹出系统保存对话框，可自己改文件名与目录（默认打开桌面）。
        </p>

        {artifact ? (
          <div className="space-y-2 rounded-lg border border-ok/25 bg-ok/[0.06] p-2.5">
            <div className="flex items-center gap-1.5 text-2xs text-ok">
              <CheckIcon className="h-3 w-3" />
              已生成 · {artifact.validated ? 'Schema 校验通过' : '未校验'}
            </div>
            <div className="break-all font-mono text-2xs text-slate-300">{artifact.filename}</div>
            <div className="flex items-center justify-between font-mono text-2xs text-slate-500">
              <span>{formatCompactNumber(artifact.frames)} 帧</span>
              <span>{(artifact.bytes / 1024).toFixed(1)} KiB</span>
              <span>{artifact.elapsedSeconds.toFixed(2)}s</span>
            </div>
            {analysis.downloadUrl ? (
              <button
                type="button"
                className="btn-ghost w-full py-1 text-2xs"
                onClick={analysis.downloadArtifact}
              >
                下载 JSON
              </button>
            ) : null}
            <p className="break-all text-2xs text-slate-600" title="导出文件所在目录">
              {artifact.path}
            </p>
          </div>
        ) : null}

        {analysis.exportWarnings.length ? (
          <ul className="space-y-1 rounded-lg border border-warn/25 bg-warn/[0.06] p-2.5">
            <li className="text-2xs font-medium text-warn">
              深度可信度提示（已写入 JSON 的 warnings 字段）
            </li>
            {analysis.exportWarnings.map((warning) => (
              <li key={warning} className="text-2xs leading-relaxed text-warn/90">
                · {warning}
              </li>
            ))}
          </ul>
        ) : null}

        <p className="text-2xs leading-relaxed text-slate-600">
          坐标系为 ARKit 右手系（x 向右、y 向上、-z 向前），关节顺序来自
          RealityKit 示例；导出文件只写到本机，不上传云端。
        </p>
      </div>
    </section>
  )
}
