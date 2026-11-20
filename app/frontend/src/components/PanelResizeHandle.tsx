import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'

interface PanelResizeHandleProps {
  width: number
  minWidth: number
  maxWidth: number
  resizing: boolean
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerCancel: (event: ReactPointerEvent<HTMLDivElement>) => void
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void
  onReset: () => void
}

/**
 * 左右分栏之间的拖拽分隔条：拖动改宽、双击恢复默认、←/→ 键盘微调。
 * 仅在 lg 及以上显示（窄屏是上下布局）。
 */
export function PanelResizeHandle({
  width,
  minWidth,
  maxWidth,
  resizing,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerCancel,
  onKeyDown,
  onReset,
}: PanelResizeHandleProps) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="拖动调整右侧面板宽度"
      aria-valuenow={width}
      aria-valuemin={minWidth}
      aria-valuemax={maxWidth}
      aria-valuetext={`${width} 像素`}
      title="拖动调整宽度 · 双击恢复默认"
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
      className={`group relative hidden w-3 shrink-0 cursor-col-resize touch-none select-none place-items-center rounded-full outline-none transition-colors lg:grid ${
        resizing ? 'bg-accent/[0.07]' : 'hover:bg-slate-400/10'
      }`}
    >
      {/* 贯穿的细线 */}
      <span
        aria-hidden
        className={`pointer-events-none absolute inset-y-3 left-1/2 w-px -translate-x-1/2 rounded-full transition-colors ${
          resizing ? 'bg-accent/60' : 'bg-line group-hover:bg-accent/30'
        }`}
      />
      {/* 中央抓手 */}
      <span
        aria-hidden
        className={`pointer-events-none relative rounded-full transition-all duration-150 ${
          resizing
            ? 'h-16 w-[3px] bg-accent shadow-glow'
            : 'h-12 w-1 bg-slate-400/50 group-hover:h-16 group-hover:w-[3px] group-hover:bg-accent/80 group-focus-visible:h-16 group-focus-visible:w-[3px] group-focus-visible:bg-accent/80'
        }`}
      />
      {/* 拖动过程中的宽度提示 */}
      {resizing ? (
        <span
          aria-hidden
          className="pointer-events-none absolute right-full top-1/2 mr-2 -translate-y-1/2 whitespace-nowrap rounded-md border border-line bg-panel px-1.5 py-0.5 font-mono text-2xs text-accent shadow-panel"
        >
          {width}px
        </span>
      ) : null}
    </div>
  )
}
