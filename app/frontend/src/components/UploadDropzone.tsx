import { useCallback, useRef, useState, type DragEvent, type ReactNode } from 'react'

import { UploadIcon } from './icons'

const ACCEPT = '.mp4,.mov,.m4v,.avi,.mkv,video/*'

interface UploadDropzoneProps {
  onFile: (file: File) => void
  children: (openPicker: () => void) => ReactNode
}

/** 全屏拖拽导入：整个应用都是投放目标，拖入时高亮。 */
export function UploadDropzone({ onFile, children }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)

  const openPicker = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const resetDrag = useCallback(() => {
    dragDepth.current = 0
    setDragging(false)
  }, [])

  const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) {
      setDragging(false)
    }
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    resetDrag()
    const file = event.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  return (
    <div
      className="relative flex h-full min-h-0 flex-col"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children(openPicker)}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          event.target.value = ''
        }}
      />

      {dragging ? (
        <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-canvas/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-4 rounded-2xl border-2 border-dashed border-accent/70 bg-panel/70 px-16 py-12 shadow-glow">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-accent/15">
              <UploadIcon className="h-6 w-6 text-accent" />
            </div>
            <div className="text-center">
              <p className="text-base font-medium text-slate-100">松开鼠标，开始导入视频</p>
              <p className="mt-1 text-xs text-slate-400">支持 MP4 / MOV / AVI / MKV，视频仅在本机处理</p>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
