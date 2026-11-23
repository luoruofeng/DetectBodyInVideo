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

/** 深度方向（每骨段 1 个比特）先验的执行结果。 */
export interface DepthSignSummary {
  applied: boolean
  mode: 'temporal' | 'model' | 'off' | string
  /** 实际下发的 (帧 × 骨段) 符号数量 */
  segments: number
  /** 原始符号序列的帧间翻转次数 */
  flipsBefore: number
  /** 时序平滑后下发符号的帧间翻转次数 */
  flipsAfter: number
  disagreements: number
  /** 实际使用的抬升器（videopose3d / motionbert / agformer）；mode=temporal 时为空或"尝试过哪个" */
  lifter?: string
  /** 自校准一致率 0~1；null/undefined = 未做（temporal 或样本不足） */
  autoCalibrateRatio?: number | null
  /** 自校准结论：通过 / 深度轴镜像已取反 / 判定无效 / 未做 */
  calibration?: 'inactive' | 'skipped' | 'passed' | 'inverted' | 'discarded' | string
  /** §4 方向向量先验生效的 (帧 × 骨段) 数量；0 = 只有符号（hinge） */
  directionSegments?: number
  /** §5.2 摆动腿识别结果 */
  swingLeg?: 'left' | 'right' | 'none' | string
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
  /** 深度方向先验（后端字段 depthSign）；默认关闭时 mode 为 'off' */
  depthSign?: DepthSignSummary
  /** 摆动腿与相位（§8.2）；旧后端没有这个字段 */
  kick?: KickSummary
  /** §5.1 条件式重跑：说明这次为什么跑了两遍 */
  retriedWithHeavy?: boolean
  retryReason?: string
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
  error_details: Record<string, unknown> | null
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
  /** 髋到最低着地点的竖直距离（米，后端已做零相位平滑）；预览直接用它把脚底放到地面 */
  hipHeight?: number | null
  /**
   * 髋相对双脚水平基准点的偏移（米，后端已做零相位平滑）。
   * 关节是髋局部坐标，不加上这个偏移，原地站立时脚会跟着髋在地面上滑动。
   */
  hipOffsetX?: number
  hipOffsetZ?: number
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

/** 深度方向先验：单目下每根骨段只剩"Δz 取正还是取负"一个比特。 */
export interface DepthSignOptions {
  enabled?: boolean
  /** temporal = 用世界坐标 z 做时序一致性（零新增依赖）；model = 需下载 ONNX 权重 */
  mode?: 'temporal' | 'model'
  window?: number
  min_agreement?: number
  /** model 后端的抬升器权重路径；留空自动搜 models/<lifter>-h36m.onnx */
  modelPath?: string
  /** 用同段视频的 world z 符号一致率判断模型是否镜像/无效（默认开） */
  autoCalibrate?: boolean
  /** 模型的左右约定与 BlazePose 相反时置 true */
  swapLeftRight?: boolean
}

/** 直立基准校准：正对静止片段的躯干 z 偏置扣除（门控不满足时自动跳过）。 */
export interface UprightBaselineOptions {
  enabled?: boolean
  /** 只看开头多少帧来估计基准 */
  window?: number
  /** 至少多少帧满足正对静止才应用 */
  min_frames?: number
  /** 观测宽度比高于该值视为正对 */
  frontal_ratio?: number
  /** 相邻帧髋中点的最大 2D 位移（归一化画面高度） */
  max_motion?: number
  /** 偏置绝对值上限（米），超过则跳过 */
  max_bias_meters?: number
}

/** 引擎侧检测阈值与条件式重跑（§5.1）。 */
export interface EngineOptions {
  minDetectionConfidence?: number
  minTrackingConfidence?: number
  /** MediaPipe 专用：下肢可见度低时裁剪下半身再推理 */
  lowerBodyRefine?: boolean
  /** 条件式重跑使用的档位；留空为 heavy */
  modelTier?: 'lite' | 'full' | 'heavy' | null
  /** 下肢可见度不足时是否重跑整段；留空跟随服务级配置 */
  retryWithHeavy?: boolean | null
}

/** 摆动腿 / 踢腿相位（§8.2）：只读 2D 轨迹，不改变坐标。 */
export interface KickSummary {
  swingLeg: 'left' | 'right' | 'none' | string
  swingConfidence: number
  swingApplied: boolean
  energy: Record<string, number>
  detected: boolean
  side: 'left' | 'right' | 'none' | string
  /** start / acceleration / peak / return → 帧号 */
  phases: Record<string, number>
  notes: string[]
}

export interface PostprocessOptions {
  depth_sign?: DepthSignOptions
  upright_baseline?: UprightBaselineOptions
}

export interface AnalysisOptions {
  engine?: EngineKind
  /** 引擎阈值 / 下肢裁剪重解 / 条件式重跑（后端字段 engineOptions） */
  engineOptions?: EngineOptions
  sampling?: SamplingOptions
  skeleton?: SkeletonOptions
  refinement?: RefinementOptions
  perf?: PerfOptions
  /** 骨长约束拟合（后端字段 ik_fit，HTTP 别名为 ikFit） */
  ikFit?: FitOptions
  postprocess?: PostprocessOptions
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
