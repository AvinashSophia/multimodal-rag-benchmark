import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/query": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/image": "http://localhost:8000",
      "/config": "http://localhost:8000",
      "/feedback": "http://localhost:8000",
      "/upload-query-image": "http://localhost:8000",
      "/heatmap": "http://localhost:8000",
      "/storage": "http://localhost:8000",
    },
  },
});
