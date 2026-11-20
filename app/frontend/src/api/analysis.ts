import { API_BASE, api } from './client'
import type { BoundingBox } from './types'

export type Granularity = 'keypoints' | 'full' | 'compact'
export type ExportFormat = 'pose' | 'motion'
export type JobPhase = 'queued' | 'running' | 'done' | 'cancelled' | 'error'
export type EngineKind = 'auto' | 'mediapipe' | 'onnx'

export interface AnalysisProgress {
  frames_done: number
  frames_total: number
  processed_fps: number
  eta_seconds: number | null
  message: string
  /** 推理完成后用 stage_done/stage_total 表示后处理子阶段进度 */
  stage: string
  stage_done: number
  stage_total: number
}

export interface EngineCapabilities {
  name: string
  display_name: string
  multi_person: boolean
  world_landmarks: boolean
  supports_batching: boolean
  providers: string[]
  model_tier: string
  device: string | null
  notes: string[]
}

export interface EngineSelection {
  engine: string
  display_name: string
  requested: string
  degraded: boolean
  reasons: string[]
  capabilities: EngineCapabilities
  model_tier: string
}

export interface QualitySummary {
  frames_total: number
  frames_valid: number
  valid_frame_ratio: number
  interpolated_frames: number
  mean_visibility: number
  mean_score: number
  max_jump: number
  max_jump_frame: number | null
  /** 深度专项（schema 1.1.0）：z 抖动 RMS（米） */
  z_jitter_rms: number
  /** 骨长变异系数最大值（0~1） */
  bone_length_cv: number
  /** 3D 反投影与 2D 的平均误差（归一化坐标） */
  mean_reproj_error: number
  /** 深度可信度 0~1：1 = 深度误差未污染动作 */
  depth_confidence: number
  verdict: 'ready' | 'review' | 'rerecord'
  verdict_label: string
  issues: string[]
  smoothing: Record<string, unknown>
  notes: string[]
}

export interface TrackSummary {
  id: number
  frames: number
  first_index: number
  last_index: number
  span_frames: number
  coverage: number
  mean_score: number
  mean_bbox_area: number
}

export interface FitSummary {
  applied: boolean
  frames: number
  frames_fitted: number
  frames_skipped: number
  iterations: number
  cost: number
  reproj_error_norm: number
  bone_length_error_pct: number
  converged: boolean
  milliseconds_per_frame: number
  bone_lengths: Record<string, number>
  notes: string[]
}

export interface AnalysisSummary {
  engine: string
  engineDisplayName: string
  degraded: boolean
  reasons: string[]
  frames: number
  processedFps: number
  elapsedSeconds: number
  quality: QualitySummary
  fit: FitSummary
  tracks: TrackSummary[]
  primaryTrackId: number | null
  targetTrackId: number | null
  notes: string[]
}

export interface JobStatus {
  id: string
  video_id: string
  filename: string
  phase: JobPhase
  progress: AnalysisProgress
  selection: EngineSelection | null
  summary: AnalysisSummary | null
  error: string | null
  error_code: string | null
  cancel_requested: boolean
  elapsedSeconds: number
  canCancel: boolean
  canExport: boolean
  canPreview: boolean
}

export interface PreviewPoint {
  x: number
  y: number
  z: number
  visibility: number
}

export interface PreviewJoint {
  x: number
  y: number
  z: number
  visibility: number
}

export interface PreviewFrame {
  index: number
  t: number
  valid: boolean
  interpolated: boolean
  landmarks: PreviewPoint[]
  joints: Record<string, PreviewJoint>
  angles: Record<string, number>
  bbox: BoundingBox | null
}

export interface FramesPayload {
  jobId: string
  fps: number
  stride: number
  totalFrames: number
  offset: number
  limit: number
  hasMore: boolean
  jointNames: string[]
  frames: PreviewFrame[]
}

export interface ExportArtifact {
  path: string
  filename: string
  bytes: number
  frames: number
  mode: Granularity | 'motion' | string
  validated: boolean
  elapsedSeconds: number
}

export interface ExportResponse {
  artifact: ExportArtifact
  downloadUrl: string | null
  outputDir: string
  schemaVersion: string
  format: ExportFormat
  /** 深度可信度 0~1（schema 1.1.0） */
  depthConfidence: number
  /** 深度不可信时的中文提示，导出成功提示里要带一句 */
  warnings: string[]
}

export interface SamplingOptions {
  every_n_frames?: number
  target_fps?: number | null
  trim_empty?: boolean
}

export interface SkeletonOptions {
  root_motion_mode?: 'world' | 'in_place'
  include_optional?: boolean
  /** 输出 AppleModel.usda 的全部 91 个关节（含手指/面部细分） */
  include_all_joints?: boolean
}

export interface RefinementOptions {
  hands?: boolean
  face?: boolean
  max_hands?: number
  num_faces?: number
  refine_landmarks?: boolean
}

export interface PerfOptions {
  queue_size?: number
  batch_size?: number
  segment_frames?: number
  checkpoint?: boolean
  resume?: boolean
  fp16?: 'auto' | 'on' | 'off'
  worker_threads?: number
}

export interface FitOptions {
  enabled?: boolean
  max_nfev?: number
  temporal_weight?: number
  max_yaw_deviation?: number
  loss?: 'linear' | 'soft_l1' | 'huber' | 'cauchy' | 'arctan'
  min_visibility?: number
}

export interface AnalysisOptions {
  engine?: EngineKind
  sampling?: SamplingOptions
  skeleton?: SkeletonOptions
  refinement?: RefinementOptions
  perf?: PerfOptions
  /** 骨长约束拟合（后端字段 ik_fit，HTTP 别名为 ikFit） */
  ikFit?: FitOptions
  max_frames?: number | null
}

export interface MotionExportOptions {
  sample_interval_ms?: number
  loop?: boolean
  exercise_id?: string | null
  name?: string | null
  name_zh?: string | null
}

export interface ExportOptions {
  format?: ExportFormat
  granularity: Granularity
  include_raw_landmarks?: boolean
  include_root_motion?: boolean
  output_dir?: string | null
  filename?: string | null
  motion?: MotionExportOptions | null
}

export function startAnalysis(videoId: string, options: AnalysisOptions): Promise<JobStatus> {
  return api.post<JobStatus>('/analysis', { video_id: videoId, options })
}

export function fetchAnalysis(jobId: string): Promise<JobStatus> {
  return api.get<JobStatus>(`/analysis/${jobId}`)
}

export function cancelAnalysis(jobId: string): Promise<JobStatus> {
  return api.post<JobStatus>(`/analysis/${jobId}/cancel`)
}

export interface CacheClearResult {
  /** 删除的断点文件数量 */
  removed: number
  directory: string
}

/** 清空断点续算缓存；之后启动的任务会从第 1 帧重新识别。 */
export function clearAnalysisCache(): Promise<CacheClearResult> {
  return api.delete<CacheClearResult>('/analysis/cache')
}

export function fetchFrames(
  jobId: string,
  offset = 0,
  limit = 2000,
): Promise<FramesPayload> {
  return api.get<FramesPayload>(`/analysis/${jobId}/frames?offset=${offset}&limit=${limit}`)
}

export function exportAnalysis(jobId: string, options: ExportOptions): Promise<ExportResponse> {
  return api.post<ExportResponse>(`/analysis/${jobId}/export`, options)
}

export function exportDownloadUrl(filename: string): string {
  return `${API_BASE}/exports/${encodeURIComponent(filename)}`
}

export function schemaUrl(): string {
  return `${API_BASE}/exports/schema`
}

export const FRAME_PAGE_LIMIT = 4000
// 长视频（实测 2 万帧以上）不能只加载前半段，否则后半段视频没有骨架叠加。
export const MAX_PREVIEW_FRAMES = 100000
