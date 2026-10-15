import { defineConfig } from "vite";
import { cpSync, createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import { join } from "node:path";

// onnxruntime-web loads its WASM runtime at run time; ship the WebGPU-capable build alongside the app
const ORT = ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"];
// copied when the config loads, before Vite indexes the public folder
mkdirSync("public/ort", { recursive: true });
for (const f of ORT) cpSync(`node_modules/onnxruntime-web/dist/${f}`, `public/ort/${f}`);

// development only: serve a local model folder at /model/ (set FLUID_MODEL_DIR and VITE_MODEL_BASE=/model/)
function localModel() {
  const dir = process.env.FLUID_MODEL_DIR;
  return {
    name: "local-model",
    configureServer(server: any) {
      if (!dir) return;
      server.middlewares.use("/model/", (req: any, res: any, next: any) => {
        const f = join(dir, decodeURIComponent(req.url.split("?")[0]));
        if (!existsSync(f)) return next();
        res.setHeader("Content-Length", statSync(f).size);
        res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
        if (req.method === "HEAD") return res.end();
        createReadStream(f).pipe(res);
      });
    },
  };
}

// cross-origin isolation enables multi-threaded WASM (SharedArrayBuffer)
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  clearScreen: false,
  plugins: [localModel()],
  server: { port: 5173, strictPort: true, headers: isolation },
  preview: { port: 4173, headers: isolation },
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
