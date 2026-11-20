/**
 * 导出落点选择：优先用桌面壳（pywebview）的原生「另存为」对话框，
 * 让用户自己改文件名、换目录；默认目录由桌面壳打开系统桌面。
 *
 * 浏览器模式下没有可用的原生对话框，返回 ``undefined`` 让调用方沿用
 * 「后端默认目录 + 下载链接」的老逻辑。
 */

export interface ExportSaveTarget {
  /** 目标目录（绝对路径），对应后端 output_dir */
  directory: string
  /** 目标文件名，对应后端 filename */
  filename: string
}

interface PywebviewBridge {
  choose_export_path?: (defaultName: string) => Promise<string | null> | string | null
}

declare global {
  interface Window {
    /** pywebview 注入的 JS 桥；窗口就绪且配置了 js_api 时才存在。 */
    pywebview?: { api?: PywebviewBridge }
  }
}

const UNSAFE_NAME = /[\\/:*?"<>|\s]+/g

/** 与后端 ``safe_stem`` 对齐：把源文件名压成可安全落盘的干名（保留中文）。 */
export function safeStem(original: string): string {
  const name = String(original ?? '').split(/[\\/]/).pop() ?? ''
  const stem = name.replace(/\.[^.]*$/, '')
  return stem.replace(UNSAFE_NAME, '_').replace(/^[._]+|[._]+$/g, '') || 'video'
}

function timestampStamp(now = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, '0')
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  )
}

/** 对话框里的建议文件名，与后端默认命名一致：`{原名}_{时间戳}_pose{粒度}.json`。 */
export function suggestExportFilename(
  sourceName: string,
  format: 'pose' | 'motion',
  granularity: 'keypoints' | 'full' | 'compact',
): string {
  const stem = safeStem(sourceName)
  const stamp = timestampStamp()
  return format === 'motion'
    ? `${stem}_${stamp}_motion.json`
    : `${stem}_${stamp}_pose${granularity}.json`
}

/** 把绝对路径拆成「目录 + 文件名」，Windows 与 POSIX 分隔符都处理。 */
export function splitSavePath(path: string): ExportSaveTarget | null {
  const trimmed = String(path ?? '').trim()
  if (!trimmed) return null
  const index = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'))
  const filename = trimmed.slice(index + 1)
  if (!filename) return null
  let directory = index > 0 ? trimmed.slice(0, index) : trimmed.slice(0, index + 1)
  // "C:\foo.json" 拆出来的目录是 "C:"，补上分隔符才是盘符根目录
  if (/^[A-Za-z]:$/.test(directory)) directory += '\\'
  return { directory, filename }
}

/**
 * 弹系统「另存为」对话框。
 *
 * - 返回 ``ExportSaveTarget``：用户选好了位置
 * - 返回 ``null``：用户取消（调用方应中止导出）
 * - 返回 ``undefined``：当前环境没有原生对话框（浏览器降级）
 */
export async function pickExportTarget(
  suggestedName: string,
): Promise<ExportSaveTarget | null | undefined> {
  if (typeof window === 'undefined') return undefined
  const bridge = window.pywebview?.api
  if (!bridge?.choose_export_path) return undefined
  try {
    const picked = await bridge.choose_export_path(suggestedName)
    if (!picked) return null
    return splitSavePath(String(picked))
  } catch {
    // 桥调用异常时退回默认目录，至少让导出能完成
    return undefined
  }
}
