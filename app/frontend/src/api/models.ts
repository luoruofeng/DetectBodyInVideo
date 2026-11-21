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

/** 深度符号先验 model 后端的抬升器（§2.4：缺权重时不要把 model 档做成可选项）。 */
export interface LifterInfo {
  name: string
  filename: string
  present: boolean
  path: string
  receptiveField: number
}

export interface ModelsPayload {
  models: ModelFileInfo[]
  poseReady: boolean
  mirrorConfigured: boolean
  downloads: DownloadTask[]
  /** 后向兼容：旧后端没有这个字段 */
  lifters?: LifterInfo[]
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
