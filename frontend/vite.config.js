import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (`npm run dev`), proxy V2 API calls to the local inference server so
// the browser sees a single origin. In production the FastAPI app serves both
// the built assets and /v2, so no proxy is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v2": { target: "http://localhost:8080", changeOrigin: true },
      "/metrics": { target: "http://localhost:8080", changeOrigin: true },
    },
  },
  build: { outDir: "dist" },
});
