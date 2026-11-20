import { api } from './client'

export interface CpuInfo {
  model: string
  physical_cores: number
  logical_cores: number
  avx2: boolean | null
  avx512: boolean | null
  features_known: boolean
}

export interface GpuInfo {
  vendor: string
  name: string
  memory_mb: number | null
  driver: string | null
  cuda_version: string | null
  api: string
  unified_memory: boolean
}

export interface MemoryInfo {
  total_mb: number | null
  available_mb: number | null
}

export interface SystemInfo {
  platform: string
  platform_release: string
  platform_version: string
  architecture: string
  machine: string
  glibc: string | null
  is_apple_silicon: boolean
  is_wsl: boolean
  cpu: CpuInfo
  memory: MemoryInfo
  gpus: GpuInfo[]
  onnx_providers: string[]
  directml_available: boolean
  notes: string[]
}

export type ModelTier = 'lite' | 'full' | 'heavy'

export interface AccelerationPlan {
  preferred_provider: string
  device_name: string | null
  gpu_accelerated: boolean
  execution_providers: string[]
  fallback_chain: string[]
  threads: number
  model_tier: ModelTier
  model_tier_reason: string
  confidence: 'high' | 'medium' | 'low'
  reasons: string[]
}

export interface HardwareReport {
  generated_at: string
  system: SystemInfo
  acceleration: AccelerationPlan
}

export function fetchHardware(refresh = false): Promise<HardwareReport> {
  return api.get<HardwareReport>(`/hardware${refresh ? '?refresh=true' : ''}`)
}

/** "CUDAExecutionProvider" → "CUDA" */
export function providerLabel(provider: string): string {
  const stripped = provider.replace(/ExecutionProvider$/i, '')
  if (stripped.toLowerCase() === 'tensorrt') return 'TensorRT'
  if (stripped.toLowerCase() === 'directml') return 'DirectML'
  if (stripped.toLowerCase() === 'coreml') return 'CoreML'
  return stripped || 'CPU'
}
