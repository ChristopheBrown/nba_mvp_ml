import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/predict": "http://localhost:5000",
      "/candidate_pool": "http://localhost:5000",
      "/pipeline": "http://localhost:5000",
      "/monitoring": "http://localhost:5000"
    }
  }
});
