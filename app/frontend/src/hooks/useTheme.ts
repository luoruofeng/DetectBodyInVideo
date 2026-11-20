import { useCallback, useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'detect-body-theme'

function readTheme(): Theme {
  if (typeof window === 'undefined') return 'dark'
  const stored = window.localStorage.getItem(STORAGE_KEY)
  return stored === 'light' ? 'light' : 'dark'
}

/** 主题切换：深色为默认，选择写入 localStorage（任务 8.2 顶部主题切换）。 */
export function useTheme(): { theme: Theme; toggle: () => void; setTheme: (theme: Theme) => void } {
  const [theme, setThemeState] = useState<Theme>(readTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  const setTheme = useCallback((value: Theme) => setThemeState(value), [])
  const toggle = useCallback(
    () => setThemeState((previous) => (previous === 'dark' ? 'light' : 'dark')),
    [],
  )

  return { theme, toggle, setTheme }
}
