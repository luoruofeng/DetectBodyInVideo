interface IconProps {
  className?: string
}

const base = 'h-4 w-4'

export function PoseIcon({ className = 'h-5 w-5' }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <circle cx="12" cy="4.5" r="2.2" />
      <path d="M12 6.8v6.4M12 8.6 7.6 12M12 8.6l4.4 3.4M12 13.2l-3 6.4M12 13.2l3 6.4" strokeLinecap="round" />
    </svg>
  )
}

export function UploadIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
    </svg>
  )
}

export function CpuIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M10 3v2m4-2v2m-4 16v-2m4 2v-2M3 10h2m-2 4h2m16-4h-2m2 4h-2" strokeLinecap="round" />
    </svg>
  )
}

export function CheckIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className={className} aria-hidden>
      <path d="m5 12.5 4.2 4.2L19 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function AlertIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className={className} aria-hidden>
      <path d="M12 8v5" strokeLinecap="round" />
      <circle cx="12" cy="16.6" r="0.9" fill="currentColor" stroke="none" />
      <path d="M10.3 4.3 3.6 16a2 2 0 0 0 1.7 3h13.4a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0Z" strokeLinejoin="round" />
    </svg>
  )
}

export function CloseIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className={className} aria-hidden>
      <path d="m6 6 12 12M18 6 6 18" strokeLinecap="round" />
    </svg>
  )
}

export function FilmIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
      <path d="M8 4.5v15M16 4.5v15M3 12h18" strokeLinecap="round" />
    </svg>
  )
}

export function PlayIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <path d="M8 5.5v13a1 1 0 0 0 1.53.85l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5Z" />
    </svg>
  )
}

export function PauseIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <rect x="7" y="5" width="3.6" height="14" rx="1.2" />
      <rect x="13.4" y="5" width="3.6" height="14" rx="1.2" />
    </svg>
  )
}

export function StepBackIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} className={className} aria-hidden>
      <path d="M18 6v12L10 12l8-6Z" fill="currentColor" stroke="none" />
      <path d="M6 5v14" strokeLinecap="round" />
    </svg>
  )
}

export function StepForwardIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} className={className} aria-hidden>
      <path d="M6 6v12l8-6-8-6Z" fill="currentColor" stroke="none" />
      <path d="M18 5v14" strokeLinecap="round" />
    </svg>
  )
}

export function DownloadIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <path d="M12 4v11m0 0 4-4m-4 4-4-4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 19h14" strokeLinecap="round" />
    </svg>
  )
}

export function SunIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" strokeLinecap="round" />
    </svg>
  )
}

export function MoonIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z" strokeLinejoin="round" />
    </svg>
  )
}

export function BoneIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <circle cx="6" cy="6" r="2.2" />
      <circle cx="18" cy="18" r="2.2" />
      <path d="m7.6 7.6 8.8 8.8" strokeLinecap="round" />
      <circle cx="18" cy="6" r="1.6" />
      <circle cx="6" cy="18" r="1.6" />
    </svg>
  )
}

export function SparkIcon({ className = base }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className={className} aria-hidden>
      <path d="M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M18 6l-2.5 2.5m-7 7L6 18" strokeLinecap="round" />
    </svg>
  )
}
