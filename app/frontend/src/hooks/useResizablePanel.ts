import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'

/** 右侧栏宽度范围（px） */
export const PANEL_MIN_WIDTH = 300
export const PANEL_MAX_WIDTH = 780
export const PANEL_DEFAULT_WIDTH = 360

const STORAGE_KEY = 'detect-body-side-panel-width'
const KEYBOARD_STEP = 16
const KEYBOARD_STEP_LARGE = 64

/** 视口太窄时按比例收缩上限，避免把视频区挤没 */
function viewportMaxWidth(): number {
  if (typeof window === 'undefined') return PANEL_MAX_WIDTH
  return Math.max(PANEL_MIN_WIDTH, Math.min(PANEL_MAX_WIDTH, Math.round(window.innerWidth * 0.6)))
}

function clampWidth(value: number, max = viewportMaxWidth()): number {
  return Math.min(Math.max(Math.round(value), PANEL_MIN_WIDTH), max)
}

function readWidth(): number {
  if (typeof window === 'undefined') return PANEL_DEFAULT_WIDTH
  const stored = Number(window.localStorage.getItem(STORAGE_KEY))
  if (!Number.isFinite(stored) || stored <= 0) return PANEL_DEFAULT_WIDTH
  return clampWidth(stored)
}

export interface ResizablePanel {
  width: number
  minWidth: number
  maxWidth: number
  resizing: boolean
  /** 拖动分隔条 */
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerCancel: (event: ReactPointerEvent<HTMLDivElement>) => void
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void
  /** 恢复默认宽度 */
  reset: () => void
}

/**
 * 右侧栏拖拽改宽：指针拖动 + 键盘微调（←/→，Shift 加速），宽度写入 localStorage。
 * 宽度只在 lg 及以上生效，窄屏仍为整列铺满。
 */
export function useResizablePanel(): ResizablePanel {
  const [width, setWidth] = useState(readWidth)
  const [maxWidth, setMaxWidth] = useState(viewportMaxWidth)
  const [resizing, setResizing] = useState(false)
  const dragRef = useRef<{ pointerId: number; startX: number; startWidth: number } | null>(null)

  // 视口变化时收敛上限
  useEffect(() => {
    const handleResize = () => {
      const nextMax = viewportMaxWidth()
      setMaxWidth(nextMax)
      setWidth((previous) => Math.min(previous, nextMax))
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // 拖动期间锁定光标与文本选择，避免选中页面文字
  useEffect(() => {
    if (!resizing) return
    const { body } = document
    const previousCursor = body.style.cursor
    const previousSelect = body.style.userSelect
    body.style.cursor = 'col-resize'
    body.style.userSelect = 'none'
    return () => {
      body.style.cursor = previousCursor
      body.style.userSelect = previousSelect
    }
  }, [resizing])

  // 非拖动状态下持久化（拖动结束时会再写一次）
  useEffect(() => {
    if (resizing) return
    try {
      window.localStorage.setItem(STORAGE_KEY, String(width))
    } catch {
      /* 隐私模式下写入失败可忽略 */
    }
  }, [resizing, width])

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return
      event.preventDefault()
      event.currentTarget.setPointerCapture?.(event.pointerId)
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startWidth: width,
      }
      setResizing(true)
    },
    [width],
  )

  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    // 面板在右侧：指针左移变大，右移变小
    const delta = drag.startX - event.clientX
    setWidth(clampWidth(drag.startWidth + delta))
  }, [])

  const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    dragRef.current = null
    setResizing(false)
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  const onKeyDown = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? KEYBOARD_STEP_LARGE : KEYBOARD_STEP
    let next: number | null = null
    if (event.key === 'ArrowLeft') next = width + step
    else if (event.key === 'ArrowRight') next = width - step
    else if (event.key === 'Home') next = PANEL_MIN_WIDTH
    else if (event.key === 'End') next = viewportMaxWidth()
    if (next === null) return
    event.preventDefault()
    setWidth(clampWidth(next))
  }, [width])

  const reset = useCallback(() => setWidth(clampWidth(PANEL_DEFAULT_WIDTH)), [])

  return {
    width,
    minWidth: PANEL_MIN_WIDTH,
    maxWidth,
    resizing,
    onPointerDown,
    onPointerMove,
    onPointerUp: endDrag,
    onPointerCancel: endDrag,
    onKeyDown,
    reset,
  }
}
