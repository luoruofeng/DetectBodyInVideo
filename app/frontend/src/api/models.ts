import { api } from './client'

export type ModelKind = 'pose' | 'hand' | 'face'

export interface ModelFileInfo {
  kind: ModelKind
  tier: string | null
  filename: string
  present: boolean
  size_bytes: number
  path: string
  url: string | null
  mirrorUrl: string | null
}

export interface DownloadTask {
  id: string
  filename: string
  state: 'queued' | 'running' | 'done' | 'error'
  downloaded: number
  total: number
  percent: number
  error: string | null
  mirror: boolean
  startedAt: number
  finishedAt: number | null
}

export interface ModelsPayload {
  models: ModelFileInfo[]
  poseReady: boolean
  mirrorConfigured: boolean
  downloads: DownloadTask[]
}

export function listModels(): Promise<ModelsPayload> {
  return api.get<ModelsPayload>('/models')
}

export function downloadModel(
  kind: ModelKind,
  tier: string,
  useMirror: boolean,
): Promise<DownloadTask> {
  return api.post<DownloadTask>('/models/download', {
    kind,
    tier,
    use_mirror: useMirror,
  })
}
