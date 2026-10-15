// Runs the whole analysis off the UI thread: speech recognition, pitch and loudness, then the coaching analysis.
import * as ort from "onnxruntime-web/webgpu";
import { FluidSpeechMini, type ModelFile, type Word } from "./engine/fluid";
import { analyse as prosody } from "./engine/prosody";
import { analyse } from "./engine/analysis";

ort.env.wasm.wasmPaths = new URL("ort/", self.location.origin + import.meta.env.BASE_URL).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
ort.env.logLevel = "error";

let engine: FluidSpeechMini | null = null;
let backend = "wasm";

export type Request =
  | { type: "load"; files: Record<ModelFile, ArrayBuffer>; backend?: "auto" | "wasm" | "webgpu" }
  | { type: "analyse"; id: number; wav: Float32Array; scenario: string; target: number }
  | { type: "analyseWords"; id: number; words: Word[]; asr_s: number; wav: Float32Array; scenario: string; target: number };

async function load(files: Record<ModelFile, ArrayBuffer>, prefer = "wasm") {
  const read = async (f: ModelFile) => new Uint8Array(files[f]);
  // Multi-threaded WASM by default (measured: ~3 s for a 30 s take). WebGPU is opt-in until it's measured on real
  // GPUs; if it fails to start, fall back to WASM.
  if (prefer === "webgpu" && "gpu" in navigator && (await (navigator as any).gpu.requestAdapter().catch(() => null))) {
    try {
      engine = await FluidSpeechMini.load(ort, read, { executionProviders: ["webgpu", "wasm"] });
      backend = "webgpu";
      return;
    } catch (e) {
      console.warn("WebGPU unavailable for this model, using WASM", e);
    }
  }
  engine = await FluidSpeechMini.load(ort, read, { executionProviders: ["wasm"] });
  backend = `wasm x${ort.env.wasm.numThreads}`;
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.files, msg.backend);
      self.postMessage({ type: "loaded", backend });
    } else if (msg.type === "analyse") {
      if (!engine) throw new Error("speech model not loaded");
      const t = performance.now();
      const words = await engine.transcribe(msg.wav, (p) => self.postMessage({ type: "progress", id: msg.id, stage: "transcribe", p }));
      const asr = (performance.now() - t) / 1000;
      self.postMessage({ type: "progress", id: msg.id, stage: "analyse", p: 1 });
      const report = analyse(words, prosody(msg.wav), msg.wav.length / 16000, msg.scenario, msg.target);
      self.postMessage({ type: "result", id: msg.id, report, timing: { asr_s: Math.round(asr * 100) / 100, total_s: Math.round((performance.now() - t) / 10) / 100 }, backend });
    } else if (msg.type === "analyseWords") {
      // speech recognition already ran natively (Core ML); measure the audio and coach here
      const t = performance.now();
      const report = analyse(msg.words, prosody(msg.wav), msg.wav.length / 16000, msg.scenario, msg.target);
      self.postMessage({ type: "result", id: msg.id, report, timing: { asr_s: msg.asr_s, total_s: Math.round((msg.asr_s + (performance.now() - t) / 1000) * 100) / 100 }, backend: "coreml" });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: "id" in msg ? msg.id : -1, message: (err as Error).message ?? String(err) });
  }
};
