import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from '../api/client'
import { uploadVideo, type VideoMetadata } from '../api/videos'
import { useAppStore } from '../store/appStore'

const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi', '.mkv']
const MAX_SIZE = 8 * 1024 ** 3

export type UploadPhase = 'idle' | 'uploading' | 'processing' | 'ready' | 'error'

export interface UploadState {
  phase: UploadPhase
  progress: number
  video: VideoMetadata | null
  previewUrl: string | null
  error: string | null
  deduped: boolean
}

const INITIAL: UploadState = {
  phase: 'idle',
  progress: 0,
  video: null,
  previewUrl: null,
  error: null,
  deduped: false,
}

export interface UseVideoUpload {
  state: UploadState
  start: (file: File) => Promise<void>
  cancel: () => void
  reset: () => void
}

export function useVideoUpload(): UseVideoUpload {
  const { pushToast } = useAppStore()
  const [state, setState] = useState<UploadState>(INITIAL)
  const controller = useRef<AbortController | null>(null)
  const previewRef = useRef<string | null>(null)

  const revokePreview = useCallback(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current)
      previewRef.current = null
    }
  }, [])

  useEffect(() => () => revokePreview(), [revokePreview])

  const start = useCallback(
    async (file: File) => {
      const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
      if (!ALLOWED_EXTENSIONS.includes(suffix)) {
        pushToast({
          kind: 'error',
          message: '不支持的视频格式',
          detail: `"${suffix || '无扩展名'}" 无法导入，请使用 MP4 / MOV / AVI / MKV`,
        })
        return
      }
      if (file.size === 0) {
        pushToast({ kind: 'error', message: '文件为空（0 字节），请重新导出后再上传' })
        return
      }
      if (file.size > MAX_SIZE) {
        pushToast({
          kind: 'error',
          message: '文件超出上限',
          detail: `当前 ${(file.size / 1024 ** 3).toFixed(2)} GiB，上限 ${MAX_SIZE / 1024 ** 3} GiB`,
        })
        return
      }

      controller.current?.abort()
      controller.current = new AbortController()

      revokePreview()
      const previewUrl = URL.createObjectURL(file)
      previewRef.current = previewUrl

      setState({
        phase: 'uploading',
        progress: 0,
        video: null,
        previewUrl,
        error: null,
        deduped: false,
      })

      try {
        const result = await uploadVideo(
          file,
          (progress) => {
            setState((previous) => ({
              ...previous,
              phase: progress >= 0.95 ? 'processing' : 'uploading',
              progress,
            }))
          },
          controller.current.signal,
        )
        setState({
          phase: 'ready',
          progress: 1,
          video: result.video,
          previewUrl,
          error: null,
          deduped: result.deduped,
        })
        pushToast({
          kind: 'success',
          message: result.deduped ? '已复用库中相同视频' : '视频导入完成',
          detail: `${result.video.display_width}×${result.video.display_height} · ${result.video.fps.toFixed(2)} fps · ${result.video.frame_count} 帧`,
        })
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === 'AbortError') {
          setState(INITIAL)
          revokePreview()
          return
        }
        const message = cause instanceof Error ? cause.message : '上传失败'
        const details = cause instanceof ApiError ? cause.details : undefined
        setState({
          phase: 'error',
          progress: 0,
          video: null,
          previewUrl,
          error: message,
          deduped: false,
        })
        pushToast({
          kind: 'error',
          message: '视频导入失败',
          detail: details && typeof details['supported'] === 'string'
            ? `${message}（支持 ${details['supported']}）`
            : message,
        })
      }
    },
    [pushToast, revokePreview],
  )

  const cancel = useCallback(() => {
    controller.current?.abort()
    controller.current = null
  }, [])

  const reset = useCallback(() => {
    controller.current?.abort()
    controller.current = null
    revokePreview()
    setState(INITIAL)
  }, [revokePreview])

  return { state, start, cancel, reset }
}
