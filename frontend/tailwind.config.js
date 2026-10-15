/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#000000',
          900: '#09090b',
          800: '#121215',
          700: '#18181b',
          600: '#27272a',
          500: '#3f3f46',
        },
        brand: {
          blue: '#2563eb',
          sky: '#0284c7',
          gray: '#71717a',
          emerald: '#10b981',
          amber: '#f59e0b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
