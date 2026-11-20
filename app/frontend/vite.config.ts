import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发：前端 5173，API 通过 proxy 打到本地 FastAPI。
// 构建：产物直接落到后端静态目录，生产模式由 FastAPI 单端口托管。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // three.js 体积大且只在 3D 预览用到，单独切块便于缓存
        manualChunks: {
          three: ['three', 'three/examples/jsm/controls/OrbitControls.js'],
        },
      },
    },
    chunkSizeWarningLimit: 700,
  },
})
