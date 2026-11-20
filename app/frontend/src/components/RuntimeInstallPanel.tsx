import { useEffect } from 'react'

import type { AcceleratorOption } from '../api/runtime'
import type { UseAccelerators } from '../hooks/useAccelerators'
import { formatBytes } from '../lib/format'
import { AlertIcon, CheckIcon, DownloadIcon } from './icons'
import { ProgressBar } from './ProgressBar'

interface RuntimeInstallPanelProps {
  accelerators: UseAccelerators
  onRefreshHardware: () => void
}

function OptionRow({
  option,
  busy,
  onInstall,
}: {
  option: AcceleratorOption
  busy: boolean
  onInstall: () => void
}) {
  return (
    <div className="rounded-lg border border-line bg-white/[0.02] px-2.5 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs text-slate-200">
            {option.label}
            {option.recommended ? (
              <span className="chip border-accent/30 bg-accent/10 text-accent">推荐</span>
            ) : null}
          </div>
          <div className="mt-0.5 text-2xs leading-relaxed text-slate-500">{option.description}</div>
          <div className="mt-1 truncate font-mono text-2xs text-slate-600" title={option.packages.join(' ')}>
            约 {formatBytes(option.estimatedBytes)} · {option.packages.join(' ')}
          </div>
        </div>
        <button
          type="button"
          className="btn-ghost shrink-0 px-2 py-1 text-2xs"
          disabled={busy || Boolean(option.blockedReason)}
          onClick={onInstall}
        >
          <DownloadIcon className="h-3 w-3" />
          安装
        </button>
      </div>
      {option.blockedReason ? (
        <p className="mt-1.5 flex items-start gap-1.5 text-2xs leading-relaxed text-warn">
          <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
          <span>{option.blockedReason}</span>
        </p>
      ) : null}
    </div>
  )
}

/** 硬件卡片里的「GPU 加速」区块：一键安装 / 进度 / 重启生效 / 回滚。 */
export function RuntimeInstallPanel({ accelerators, onRefreshHardware }: RuntimeInstallPanelProps) {
  const { payload, error, installing, install, cancel, remove, activate, restart } = accelerators

  const task = payload?.task ?? null
  const pendingVariant = payload?.active ?? task?.variant ?? payload?.installed[0]?.variant ?? null
  const ready = Boolean(payload?.restartRequired) || task?.state === 'done'

  useEffect(() => {
    if (ready) onRefreshHardware()
  }, [ready, onRefreshHardware])

  if (!payload) return null
  // GPU 加速已经真正生效（或平台自带的 EP 够用）时不再打扰用户
  if (payload.gpuAccelerated && !ready) return null
  if (error && !payload.options.length) {
    return (
      <p className="mt-2 flex items-start gap-1.5 text-2xs leading-relaxed text-danger">
        <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
        <span className="break-all">{error}</span>
      </p>
    )
  }
  if (!payload.options.length && !payload.installed.length) return null

  const labelOf = (variant: string | null | undefined) =>
    payload.options.find((option) => option.variant === variant)?.label ?? variant ?? ''

  const errorLine = task?.state === 'error' ? (
    <p className="mt-1.5 break-all text-2xs leading-relaxed text-danger">{task.error ?? '安装失败'}</p>
  ) : null

  if (ready) {
    return (
      <div className="mt-2 rounded-lg border border-ok/25 bg-ok/10 px-2.5 py-2">
        <p className="flex items-center gap-1.5 text-2xs font-medium text-ok">
          <CheckIcon className="h-3 w-3" />
          已安装 {labelOf(pendingVariant)} 运行时，重启应用后生效
        </p>
        <p className="mt-1 text-2xs leading-relaxed text-slate-400">
          onnxruntime 的动态库已被当前进程加载，只有重启才能切换到 GPU 版本（数据与已装模型都不受影响）。
          若重启后仍是 CPU，说明该 CUDA 版本不含你的显卡架构，可移除后改装 DirectML。
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {payload.restartSupported ? (
            <button
              type="button"
              className="btn-primary px-2.5 py-1 text-2xs"
              onClick={() => void restart()}
            >
              重启应用
            </button>
          ) : (
            <span className="text-2xs text-warn">
              {payload.restartBlockedReason ?? '请手动重启应用'}
            </span>
          )}
          {pendingVariant ? (
            <button
              type="button"
              className="btn-ghost px-2.5 py-1 text-2xs"
              onClick={() => void remove(pendingVariant)}
            >
              移除并回到 CPU
            </button>
          ) : null}
        </div>
        {errorLine}
      </div>
    )
  }

  if (installing && task) {
    return (
      <div className="mt-2 space-y-1.5 rounded-lg border border-line bg-white/[0.02] px-2.5 py-2">
        <div className="flex items-center justify-between gap-2 text-2xs">
          <span className="truncate text-slate-300">{task.message || '正在安装'}</span>
          <span className="shrink-0 font-mono text-slate-500">{task.percent.toFixed(0)}%</span>
        </div>
        <ProgressBar
          value={task.percent / 100}
          label="GPU 运行时安装进度"
          indeterminate={task.percent <= 2}
        />
        {task.logTail.length ? (
          <div className="max-h-24 space-y-0.5 overflow-y-auto rounded bg-black/20 p-1.5 font-mono text-2xs leading-relaxed text-slate-500">
            {task.logTail.map((line, index) => (
              <div key={index} className="truncate" title={line}>
                {line}
              </div>
            ))}
          </div>
        ) : null}
        <div className="flex justify-end">
          <button
            type="button"
            className="btn-ghost px-2 py-1 text-2xs"
            onClick={() => void cancel()}
          >
            取消安装
          </button>
        </div>
      </div>
    )
  }

  const installed = payload.installed.filter((item) => !item.active)

  return (
    <div className="mt-2 space-y-1.5">
      <p className="text-2xs leading-relaxed text-slate-500">
        装到 <span className="font-mono">~/.detect-body/runtime</span> 的独立目录，不改动当前 Python
        环境；随时可以移除回到 CPU 推理。
      </p>
      {payload.canInstall
        ? payload.options.map((option) => (
            <OptionRow
              key={option.variant}
              option={option}
              busy={installing}
              onInstall={() => void install(option.variant)}
            />
          ))
        : (
          <p className="flex items-start gap-1.5 rounded-lg border border-warn/25 bg-warn/10 px-2.5 py-2 text-2xs leading-relaxed text-warn">
            <AlertIcon className="mt-0.5 h-3 w-3 shrink-0" />
            <span>{payload.blockedReason ?? '当前环境不支持在应用内安装 GPU 运行时'}</span>
          </p>
        )}
      {installed.length ? (
        <div className="space-y-1 border-t border-line pt-1.5">
          {installed.map((item) => (
            <div key={item.variant} className="flex items-center justify-between gap-2 text-2xs">
              <span className="truncate text-slate-400">
                已安装 {labelOf(item.variant)} · {formatBytes(item.sizeBytes)}
              </span>
              <span className="flex shrink-0 gap-1">
                <button
                  type="button"
                  className="btn-ghost px-2 py-0.5 text-2xs"
                  onClick={() => void activate(item.variant)}
                >
                  切换
                </button>
                <button
                  type="button"
                  className="btn-ghost px-2 py-0.5 text-2xs"
                  onClick={() => void remove(item.variant)}
                >
                  移除
                </button>
              </span>
            </div>
          ))}
        </div>
      ) : null}
      {errorLine}
    </div>
  )
}
