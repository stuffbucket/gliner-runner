import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The page calls /v1/infer on its own address. Vite forwards /v1 to the local
// runner, so the browser never makes a cross-origin request and no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/v1": "http://127.0.0.1:8090" },
  },
});
