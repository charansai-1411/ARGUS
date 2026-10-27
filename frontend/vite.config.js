import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` on :5173 proxies /api to the FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1000 },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
