/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // 主题色走 CSS 变量：深色为默认，[data-theme='light'] 覆盖（任务 8.1 / 8.2）
        canvas: 'rgb(var(--surface-canvas) / <alpha-value>)',
        panel: 'rgb(var(--surface-panel) / <alpha-value>)',
        elevated: 'rgb(var(--surface-elevated) / <alpha-value>)',
        line: 'rgb(var(--surface-line) / var(--line-alpha))',
        accent: {
          DEFAULT: '#22D3EE',
          strong: '#06B6D4',
          soft: 'rgba(34,211,238,0.12)',
        },
        ok: '#34D399',
        warn: '#FBBF24',
        danger: '#F87171',
      },
      fontFamily: {
        sans: [
          'Inter',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'PingFang SC',
          'Microsoft YaHei',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      fontSize: {
        // 固定字号层级：12 / 14 / 16 / 20 / 28（任务 8.1）
        '2xs': ['12px', '16px'],
        xs: ['12px', '18px'],
        sm: ['14px', '20px'],
        base: ['16px', '24px'],
        lg: ['20px', '28px'],
        xl: ['28px', '36px'],
      },
      borderRadius: {
        xl: '12px',
        '2xl': '16px',
      },
      boxShadow: {
        panel: 'inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px -24px rgba(0,0,0,0.8)',
        glow: '0 0 0 1px rgba(34,211,238,0.35), 0 12px 32px -12px rgba(34,211,238,0.35)',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        'fade-in': 'fade-in 160ms ease-out',
        'pulse-soft': 'pulse-soft 1.6s ease-in-out infinite',
        shimmer: 'shimmer 1.6s linear infinite',
      },
    },
  },
  plugins: [],
}
