// What differs between the Tauri app and a plain browser: HTTP (Tauri's native client has no CORS limits) and
// where the 290 MB speech model is kept (a file in the app's data folder vs the browser's Cache Storage).
import { MODEL_FILES, MODEL_REPO, groupPieces, type ModelFile, type Word } from "./engine/fluid";

export const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

let httpFetch: typeof fetch | null = null;
export async function appFetch(): Promise<typeof fetch> {
  if (httpFetch) return httpFetch;
  if (isTauri) {
    const { fetch: tauriFetch } = await import("@tauri-apps/plugin-http");
    httpFetch = tauriFetch as unknown as typeof fetch;
  } else {
    httpFetch = globalThis.fetch.bind(globalThis);
  }
  return httpFetch;
}

// VITE_MODEL_BASE lets development builds load a local copy of the model instead of downloading it
const MODEL_BASE = import.meta.env.VITE_MODEL_BASE || `https://huggingface.co/${MODEL_REPO}/resolve/main/`;
const MODEL_URL = (f: ModelFile) => new URL(f, new URL(MODEL_BASE, location.href)).href;
const CACHE = "fluid-speech-mini-v1";

export type Progress = (loaded: number, total: number) => void;

/** Make sure every model file is stored locally (downloading what's missing), reporting overall progress. */
export async function ensureModel(onProgress: Progress): Promise<void> {
  if (isTauri) {
    const { invoke, Channel } = await import("@tauri-apps/api/core");
    const channel = new Channel<{ loaded: number; total: number }>();
    channel.onmessage = (m) => onProgress(m.loaded, m.total);
    await invoke("ensure_model", { repo: MODEL_REPO, files: [...MODEL_FILES], onProgress: channel });
    return;
  }
  const cache = await caches.open(CACHE);
  const missing: ModelFile[] = [];
  for (const f of MODEL_FILES) if (!(await cache.match(MODEL_URL(f)))) missing.push(f);
  if (!missing.length) return onProgress(1, 1);
  const sizes = await Promise.all(missing.map(async (f) => Number((await fetch(MODEL_URL(f), { method: "HEAD" })).headers.get("content-length") ?? 0)));
  const total = sizes.reduce((a, b) => a + b, 0);
  let loaded = 0;
  for (const f of missing) {
    const res = await fetch(MODEL_URL(f));
    if (!res.ok || !res.body) throw new Error(`couldn't download ${f} (${res.status})`);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total);
    }
    await cache.put(MODEL_URL(f), new Response(new Blob(chunks as BlobPart[])));
  }
}

/** Read the stored model files as transferable buffers for the analysis worker. */
export async function readModel(): Promise<Record<ModelFile, ArrayBuffer>> {
  const out = {} as Record<ModelFile, ArrayBuffer>;
  if (isTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    for (const f of MODEL_FILES) out[f] = await invoke<ArrayBuffer>("read_model_file", { repo: MODEL_REPO, name: f });
    return out;
  }
  const cache = await caches.open(CACHE);
  for (const f of MODEL_FILES) {
    const res = await cache.match(MODEL_URL(f));
    if (!res) throw new Error(`model file missing: ${f}`);
    out[f] = await res.arrayBuffer();
  }
  return out;
}

// ---------------------------------------------------------------- Core ML on macOS (Neural Engine)

const COREML_REPO = "FluidVoice-Labs/fluid-speech-mini-coreml";
const COREML_FILES = ["config.json", "parakeet_vocab.json", ...["Preprocessor", "Encoder", "Decoder", "JointDecision"].flatMap((m) =>
  ["analytics/coremldata.bin", "coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"].map((f) => `${m}.mlmodelc/${f}`))];
export const isMac = /Mac/.test(navigator.userAgent);

/** Download the Core ML model and start the native helper. Throws where Core ML isn't available. */
export async function startNative(onProgress: Progress): Promise<void> {
  if (!isTauri || !isMac) throw new Error("Core ML needs the macOS app");
  const { invoke, Channel } = await import("@tauri-apps/api/core");
  const channel = new Channel<{ loaded: number; total: number }>();
  channel.onmessage = (m) => onProgress(m.loaded, m.total);
  await invoke("ensure_model", { repo: COREML_REPO, files: COREML_FILES, onProgress: channel });
  await invoke("native_asr_start", { repo: COREML_REPO });
}

export async function transcribeNative(wav: Float32Array): Promise<{ words: Word[]; ms: number }> {
  const { invoke } = await import("@tauri-apps/api/core");
  const bytes = new Uint8Array(wav.buffer, wav.byteOffset, wav.byteLength);
  const res = await invoke<{ pieces: [string, number, number][]; ms: number }>("native_asr_transcribe", bytes);
  return { words: groupPieces(res.pieces), ms: res.ms };
}

/** Decode any recording the platform can play, as 16 kHz mono float samples. */
export async function decodeAudio(blob: Blob, rate = 16000): Promise<Float32Array> {
  const buf = await blob.arrayBuffer();
  const ctx = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(buf);
  } catch {
    throw new Error("This file's audio format can't be decoded here. Try MP3, M4A, WAV or WebM.");
  } finally {
    ctx.close();
  }
  const frames = Math.ceil(decoded.duration * rate);
  const off = new OfflineAudioContext(1, frames, rate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice();
}
