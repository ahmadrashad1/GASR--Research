import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwind from '@tailwindcss/vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// The build has one job beyond bundling: produce a page that opens by double-click.
//
// A page on the file:// origin cannot fetch local data and cannot load a module
// script, so the output must be ONE file containing its own CSS and JS, and that JS
// must be a classic script rather than an ES module. Hence format: 'iife' and
// inlineDynamicImports -- Vite emits ESM by default, which would look fine over a
// dev server and fail silently on the examiner's laptop.
//
// demo_data.js and frames/ stay OUTSIDE the bundle. They are the measured export,
// they are large, and they are not committed; the page loads them at runtime by the
// two mechanisms file:// still permits.
export default defineConfig({
  base: './',
  plugins: [react(), tailwind(), viteSingleFile()],
  build: {
    target: 'es2020',
    cssCodeSplit: false,
    assetsInlineLimit: 100_000_000,
    rollupOptions: {
      output: { format: 'iife', inlineDynamicImports: true },
    },
  },
  test: {
    environment: 'happy-dom',
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['src/test-setup.ts'],
  },
});
