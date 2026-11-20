import { api } from './client'

export type AcceleratorVariant = 'cuda' | 'directml'

export interface AcceleratorOption {
  variant: AcceleratorVariant
  label: string
  description: string
  packages: string[]
  estimatedBytes: number
  recommended: boolean
  blockedReason: string | null
  installed: boolean
}

export type InstallState = 'queued' | 'running' | 'done' | 'error' | 'cancelled'
export type InstallPhase = 'preparing' | 'downloading' | 'installing' | 'verifying' | 'done'

export interface InstallTask {
  id: string
  variant: AcceleratorVariant
  state: InstallState
  phase: InstallPhase
  percent: number
  message: string
  logTail: string[]
  error: string | null
  restartRequired: boolean
  startedAt: number
  finishedAt: number | null
}

export interface InstalledRuntime {
  variant: AcceleratorVariant
  path: string
  sizeBytes: number
  active: boolean
}

export interface AcceleratorPayload {
  options: AcceleratorOption[]
  installed: InstalledRuntime[]
  active: AcceleratorVariant | null
  currentProviders: string[]
  gpuAccelerated: boolean
  restartRequired: boolean
  canInstall: boolean
  blockedReason: string | null
  restartSupported: boolean
  restartBlockedReason: string | null
  frozen: boolean
  runtimeDir: string
  task: InstallTask | null
}

export function fetchAccelerators(): Promise<AcceleratorPayload> {
  return api.get<AcceleratorPayload>('/runtime/accelerators')
}

export function installAccelerator(variant: AcceleratorVariant): Promise<InstallTask> {
  return api.post<InstallTask>('/runtime/accelerators/install', { variant })
}

export function cancelAcceleratorInstall(): Promise<InstallTask | null> {
  return api.post<InstallTask | null>('/runtime/accelerators/install/cancel')
}

export function removeAccelerator(variant: AcceleratorVariant): Promise<void> {
  return api.delete<void>(`/runtime/accelerators/${variant}`)
}

export function activateAccelerator(variant: AcceleratorVariant): Promise<{ ok: boolean }> {
  return api.post<{ ok: boolean }>(`/runtime/accelerators/${variant}/activate`)
}

export function restartApp(): Promise<{ restarting: boolean }> {
  return api.post<{ restarting: boolean }>('/runtime/restart')
}
