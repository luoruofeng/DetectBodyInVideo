import { API_BASE, ApiError, api } from './client'

export interface VideoMetadata {
  id: string
  filename: string
  sha256: string
  size_bytes: number
  extension: string
  thumbnail_url: string | null
  created_at: string
  container: string
  codec: string | null
  width: number
  height: number
  rotation: number
  display_width: number
  display_height: number
  fps: number
  frame_count: number
  duration_s: number
  is_variable_frame_rate: boolean
  has_audio: boolean | null
  warnings: string[]
}

export interface UploadResponse {
  video: VideoMetadata
  deduped: boolean
}

export interface UploadSession {
  upload_id: string
  filename: string
  size_bytes: number
  chunk_size: number
  total_chunks: number
  received_chunks: number[]
  received_bytes: number
  progress: number
  created_at: string
}

export interface SamplingConfig {
  every_n_frames?: number
  target_fps?: number | null
  start_s?: number | null
  end_s?: number | null
  trim_empty?: boolean
}

export interface TimeRange {
  start_s: number
  end_s: number
}

export interface AnalysisPlan {
  total_frames: number
  selected_frames: number
  stride: number
  effective_fps: number
  start_s: number
  end_s: number
  target_fps: number | null
  trimmed: boolean
  active_ranges: TimeRange[]
  warnings: string[]
}

/** 小于该体积走单次上传，超过则分块（避免一次请求塞满内存/超时）。 */
export const CHUNKED_THRESHOLD = 16 * 1024 * 1024

export type ProgressHandler = (progress: number) => void

function parseUploadError(status: number, raw: string): ApiError {
  try {
    const payload = JSON.parse(raw) as {
      error?: { code?: string; message?: string; details?: Record<string, unknown> }
    }
    if (payload.error) {
      return new ApiError(payload.error.message ?? '上传失败', {
        status,
        code: payload.error.code ?? 'http_error',
        details: payload.error.details,
      })
    }
  } catch {
    // 非 JSON 响应，走兜底文案
  }
  return new ApiError(`上传失败（HTTP ${status}）`, { status, code: 'http_error' })
}

function uploadSingle(
  file: File,
  onProgress: ProgressHandler,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/videos/upload`)
    xhr.responseType = 'text'

    const onAbort = () => {
      xhr.abort()
      reject(new DOMException('上传已取消', 'AbortError'))
    }
    if (signal) {
      if (signal.aborted) {
        reject(new DOMException('上传已取消', 'AbortError'))
        return
      }
      signal.addEventListener('abort', onAbort, { once: true })
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.min(0.95, (event.loaded / event.total) * 0.95))
      }
    }
    xhr.onload = () => {
      signal?.removeEventListener('abort', onAbort)
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse)
        } catch {
          reject(new ApiError('服务端返回了无法解析的数据', { status: xhr.status, code: 'bad_response' }))
        }
      } else {
        reject(parseUploadError(xhr.status, xhr.responseText))
      }
    }
    xhr.onerror = () => {
      signal?.removeEventListener('abort', onAbort)
      reject(new ApiError('无法连接本地服务，请确认后端已启动', { status: 0, code: 'network_error' }))
    }

    const form = new FormData()
    form.append('file', file, file.name)
    xhr.send(form)
  })
}

export function initChunkedUpload(file: File): Promise<UploadSession> {
  return api.post<UploadSession>('/videos/uploads', {
    filename: file.name,
    size_bytes: file.size,
  })
}

export function putChunk(session: UploadSession, index: number, blob: Blob): Promise<UploadSession> {
  return api.put<UploadSession>(
    `/videos/uploads/${session.upload_id}/chunks/${index}`,
    blob,
    { headers: { 'Content-Type': 'application/octet-stream' } },
  )
}

export function completeChunkedUpload(uploadId: string): Promise<UploadResponse> {
  return api.post<UploadResponse>(`/videos/uploads/${uploadId}/complete`)
}

export function abortChunkedUpload(uploadId: string): void {
  void fetch(`${API_BASE}/videos/uploads/${uploadId}`, { method: 'DELETE' }).catch(() => undefined)
}

export async function uploadVideo(
  file: File,
  onProgress: ProgressHandler,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  if (file.size <= CHUNKED_THRESHOLD) {
    const result = await uploadSingle(file, onProgress, signal)
    onProgress(1)
    return result
  }

  const session = await initChunkedUpload(file)
  try {
    for (let index = 0; index < session.total_chunks; index += 1) {
      if (signal?.aborted) {
        throw new DOMException('上传已取消', 'AbortError')
      }
      const start = index * session.chunk_size
      const blob = file.slice(start, Math.min(file.size, start + session.chunk_size))
      await putChunk(session, index, blob)
      onProgress(((index + 1) / session.total_chunks) * 0.95)
    }
    const result = await completeChunkedUpload(session.upload_id)
    onProgress(1)
    return result
  } catch (error) {
    abortChunkedUpload(session.upload_id)
    throw error
  }
}

export function fetchVideos(): Promise<VideoMetadata[]> {
  return api.get<VideoMetadata[]>('/videos')
}

export function fetchVideo(videoId: string): Promise<VideoMetadata> {
  return api.get<VideoMetadata>(`/videos/${videoId}`)
}

export function planAnalysis(videoId: string, config: SamplingConfig): Promise<AnalysisPlan> {
  return api.post<AnalysisPlan>(`/videos/${videoId}/plan`, config)
}

export function videoFileUrl(videoId: string): string {
  return `${API_BASE}/videos/${videoId}/file`
}

export function thumbnailUrl(video: VideoMetadata): string | null {
  return video.thumbnail_url
}
