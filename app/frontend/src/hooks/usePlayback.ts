import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

export interface LoopRange {
  a: number
  b: number
}

export interface UsePlayback {
  currentTime: number
  duration: number
  playing: boolean
  rate: number
  loop: LoopRange | null
  toggle: () => void
  play: () => void
  pause: () => void
  seek: (time: number) => void
  step: (frames: number) => void
  setRate: (rate: number) => void
  setLoop: (range: LoopRange | null) => void
  setLoopStart: () => void
  setLoopEnd: () => void
  clearLoop: () => void
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** 播放控制：真实驱动 <video>，并与后端返回的采样帧时间轴对齐。 */
export function usePlayback(
  videoRef: RefObject<HTMLVideoElement>,
  options: { duration: number; fps: number; fallbackDuration: number; mediaKey: string | null },
): UsePlayback {
  const { fps, mediaKey } = options
  const duration = options.duration > 0 ? options.duration : options.fallbackDuration
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRateState] = useState(1)
  const [loop, setLoop] = useState<LoopRange | null>(null)
  const loopRef = useRef<LoopRange | null>(null)
  const raf = useRef<number | null>(null)

  useEffect(() => {
    loopRef.current = loop
  }, [loop])

  const seek = useCallback((time: number) => {
    const video = videoRef.current
    const target = clamp(time, 0, duration || 0)
    if (video) {
      try {
        video.currentTime = target
      } catch {
        // 元数据未就绪时忽略
      }
    }
    setCurrentTime(target)
  }, [duration, videoRef])

  const play = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const started = video.play()
    if (started && typeof started.catch === 'function') {
      started.catch(() => setPlaying(false))
    }
  }, [videoRef])

  const pause = useCallback(() => {
    videoRef.current?.pause()
  }, [videoRef])

  const toggle = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) play()
    else pause()
  }, [pause, play, videoRef])

  const step = useCallback(
    (frames: number) => {
      pause()
      const stepSeconds = 1 / Math.max(1, fps)
      seek(currentTime + frames * stepSeconds)
    },
    [currentTime, fps, pause, seek],
  )

  const setRate = useCallback(
    (value: number) => {
      setRateState(value)
      if (videoRef.current) videoRef.current.playbackRate = value
    },
    [videoRef],
  )

  const setLoopStart = useCallback(() => {
    setLoop((previous) => {
      const fallbackEnd = Math.min(duration, currentTime + 2)
      const end = previous && previous.b > currentTime + 0.05 ? previous.b : fallbackEnd
      return { a: currentTime, b: Math.max(end, currentTime + 0.1) }
    })
  }, [currentTime, duration])

  const setLoopEnd = useCallback(() => {
    setLoop((previous) => {
      const fallbackStart = Math.max(0, currentTime - 2)
      const start = previous && previous.a < currentTime - 0.05 ? previous.a : fallbackStart
      return { a: Math.min(start, currentTime - 0.1), b: currentTime }
    })
  }, [currentTime])

  const clearLoop = useCallback(() => setLoop(null), [])

  // 同步 <video> 事件；video 元素是条件渲染的，mediaKey 变化时重新绑定
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    setCurrentTime(video.currentTime)
    setPlaying(!video.paused)
    video.playbackRate = rate
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    const onTime = () => setCurrentTime(video.currentTime)
    const onEnded = () => setPlaying(false)
    const onSeeked = () => setCurrentTime(video.currentTime)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('seeked', onSeeked)
    video.addEventListener('ended', onEnded)
    return () => {
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('ended', onEnded)
    }
  }, [mediaKey, rate, videoRef])

  // 播放时的平滑时间轴 + 循环区间
  useEffect(() => {
    if (!playing) {
      if (raf.current !== null) cancelAnimationFrame(raf.current)
      raf.current = null
      return
    }
    const tick = () => {
      const video = videoRef.current
      if (video) {
        const active = loopRef.current
        if (active && active.b - active.a > 0.02 && video.currentTime >= active.b - 1e-3) {
          video.currentTime = active.a
        }
        setCurrentTime(video.currentTime)
      }
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current)
      raf.current = null
    }
  }, [playing, videoRef])

  // 循环区间变化时立刻校正
  useEffect(() => {
    if (!loop) return
    const video = videoRef.current
    if (video && video.currentTime > loop.b) seek(loop.a)
  }, [loop, seek, videoRef])

  return {
    currentTime,
    duration,
    playing,
    rate,
    loop,
    toggle,
    play,
    pause,
    seek,
    step,
    setRate,
    setLoop,
    setLoopStart,
    setLoopEnd,
    clearLoop,
  }
}
