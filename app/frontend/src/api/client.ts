const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

export interface ApiErrorBody {
  error?: {
    code?: string
    message?: string
    details?: Record<string, unknown>
  }
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details?: Record<string, unknown>

  constructor(
    message: string,
    options: { status: number; code: string; details?: Record<string, unknown> },
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status
    this.code = options.code
    this.details = options.details
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: ApiErrorBody | null = null
  try {
    body = (await response.json()) as ApiErrorBody
  } catch {
    // 非 JSON 错误体（如代理 502 HTML），走兜底文案
  }
  return new ApiError(body?.error?.message ?? `请求失败（HTTP ${response.status}）`, {
    status: response.status,
    code: body?.error?.code ?? 'http_error',
    details: body?.error?.details,
  })
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...(init?.headers ?? {}),
      },
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') {
      throw cause
    }
    throw new ApiError('无法连接本地服务，请确认后端已启动', {
      status: 0,
      code: 'network_error',
    })
  }

  if (!response.ok) {
    throw await toApiError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function encodeBody(body: unknown): BodyInit | undefined {
  if (body === undefined) return undefined
  if (body instanceof FormData || body instanceof Blob || typeof body === 'string') return body
  return JSON.stringify(body)
}

export const api = {
  get: <T>(path: string, init?: RequestInit) => request<T>(path, { ...init, method: 'GET' }),
  post: <T>(path: string, body?: unknown, init?: RequestInit) =>
    request<T>(path, { ...init, method: 'POST', body: encodeBody(body) }),
  put: <T>(path: string, body?: unknown, init?: RequestInit) =>
    request<T>(path, { ...init, method: 'PUT', body: encodeBody(body) }),
  delete: <T>(path: string, init?: RequestInit) => request<T>(path, { ...init, method: 'DELETE' }),
}

export { API_BASE }
