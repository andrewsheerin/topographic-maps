import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Keep base relative so the built assets work when FastAPI serves dist/ from
// any path. Dev server proxies /api to the FastAPI backend so relative fetch
// paths work in both dev and production.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
