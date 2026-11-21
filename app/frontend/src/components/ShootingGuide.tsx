import { useCallback, useEffect, useState } from 'react'

/**
 * 拍摄建议（任务 6.1 / 6.3）。
 *
 * 文案口径：面向用户的文字一律用「深度（前后方向）」，不出现「Z 轴」；
 * 技术细节留给 docs/，界面上只回答"怎么拍、为什么"。
 */

const STORAGE_KEY = 'detect-body:shooting-guide-seen'

export const SHOOTING_RULES: { title: string; desc: string }[] = [
  { title: '正对或正侧拍摄', desc: '斜 45° 最差：两个方向都被压缩，深度只能靠猜' },
  { title: '机位与胸口齐平', desc: '俯拍/仰拍会让上下半身尺度不一致，透视越强深度越不准' },
  { title: '退到 3~5 米，用标准焦段', desc: '贴脸广角会把四肢拉长，深度误差被放大' },
  { title: '人物占画面高度 60%~90%', desc: '太小则点位精度不够，深度估计更飘' },
  { title: '不要快速朝镜头走动', desc: '前后位移是单目最不可靠的量，横向移动更安全' },
  { title: '画面稳定、光线均匀', desc: '不逆光、全身尽量不被裁切，关节才不会被遮住' },
]

export const ACTION_TIPS: { action: string; camera: string; reason: string }[] = [
  { action: '下蹲 / 弓步 / 前弓步 / 硬拉', camera: '正侧面', reason: '侧视图下膝角、髋角都在画面内，几乎没有前缩误差' },
  { action: '转体 / 挥臂 / 上肢伸展 / 开合跳', camera: '正对', reason: '需要看清左右对称与水平方向的角度' },
  { action: '走路 / 跑步', camera: '正侧（横向走）', reason: '前后走会把位移交给最不可信的深度' },
  { action: '混合动作', camera: '正侧优先', reason: '侧拍对深度的依赖最小，容错最高' },
]

function RuleList() {
  return (
    <ul className="space-y-2">
      {SHOOTING_RULES.map((rule) => (
        <li key={rule.title} className="flex gap-2">
          <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-accent" />
          <div className="min-w-0">
            <p className="text-xs font-medium text-slate-300">{rule.title}</p>
            <p className="text-2xs leading-relaxed text-slate-500">{rule.desc}</p>
          </div>
        </li>
      ))}
    </ul>
  )
}

function ActionTable() {
  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full border-collapse text-2xs">
        <thead>
          <tr className="bg-white/[0.03] text-left text-slate-400">
            <th className="px-2 py-1.5 font-medium">动作</th>
            <th className="px-2 py-1.5 font-medium">机位</th>
          </tr>
        </thead>
        <tbody>
          {ACTION_TIPS.map((tip) => (
            <tr key={tip.action} className="border-t border-line align-top">
              <td className="px-2 py-1.5 text-slate-400">{tip.action}</td>
              <td className="px-2 py-1.5">
                <span className="text-slate-300">{tip.camera}</span>
                <p className="mt-0.5 leading-relaxed text-slate-600">{tip.reason}</p>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** 侧栏里的常驻卡片（可折叠），随时能查。 */
export function ShootingGuideCard() {
  const [open, setOpen] = useState(false)
  return (
    <section className="card">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <h2 className="card-title mb-0 border-0 pb-0">
          拍摄建议
          <span className="text-2xs font-normal normal-case tracking-normal text-slate-500">
            决定精度
          </span>
        </h2>
        <span className="text-2xs text-slate-500">{open ? '收起' : '展开'}</span>
      </button>
      {open ? (
        <div className="space-y-3 border-t border-line px-4 py-3">
          <RuleList />
          <ActionTable />
        </div>
      ) : (
        <p className="border-t border-line px-4 py-2 text-2xs leading-relaxed text-slate-500">
          正对或正侧、机位与胸口齐平、人物占画面 60%~90%。矢状面动作（下蹲/弓步）建议从侧面拍。
        </p>
      )}
    </section>
  )
}

/** 首次使用时的弹层（可关闭，关闭后不再自动出现）。 */
export function ShootingGuideDialog() {
  const [open, setOpen] = useState(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) !== '1'
    } catch {
      return false
    }
  })

  const close = useCallback(() => setOpen(false), [])

  // 点击/Esc 只更新 React 状态；localStorage 同步写盘放到渲染完成后，
  // 避免在关闭弹层的同一帧里做磁盘 IO 拖慢首帧。
  useEffect(() => {
    if (open) return
    try {
      window.localStorage.setItem(STORAGE_KEY, '1')
    } catch {
      /* 隐私模式下 localStorage 不可用，忽略即可 */
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [close, open])

  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 grid place-items-center overflow-y-auto bg-black/70 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="shooting-guide-title"
        className="max-h-[calc(100vh-2rem)] w-full max-w-lg space-y-4 overflow-y-auto rounded-xl border border-line bg-panel p-5 shadow-2xl"
      >
        <div className="space-y-1">
          <h2 id="shooting-guide-title" className="text-sm font-medium text-slate-200">
            开拍前先看一眼：怎么拍，深度才可信
          </h2>
          <p className="text-2xs leading-relaxed text-slate-500">
            后期只能修正抖动，修不出没拍到的信息。同一个算法，素材不同结果能差一个量级。
          </p>
        </div>
        <RuleList />
        <ActionTable />
        <div className="flex items-center justify-between gap-3 border-t border-line pt-3">
          <p className="text-2xs text-slate-600">分析完成后，质检面板会显示「深度置信度」；偏低会提示重拍。</p>
          <button type="button" className="btn btn-primary shrink-0" onClick={close}>
            知道了
          </button>
        </div>
      </div>
    </div>
  )
}
