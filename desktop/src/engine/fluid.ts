// Fluid Speech Mini by FluidVoice (CC BY 4.0), ONNX build, run with onnxruntime-web.
// Port of src/intro_coach/asr/fluid.py: log-mel features, chunked conv front end + encoder, greedy TDT decoding
// that keeps the encoder frame of every token for word timestamps.
import type * as OrtNS from "onnxruntime-web";
import { logMel, readNpyF32 } from "./mel";

export const MODEL_REPO = "FluidVoice-Labs/fluid-speech-mini-onnx";
export const MODEL_FILES = [
  "config.json", "tokenizer.json", "mel_filters.npy",
  "frontend.onnx", "encoder.onnx", "encoder.onnx.data", "decoder.onnx", "joint.onnx",
] as const;
export type ModelFile = (typeof MODEL_FILES)[number];

export interface Word { w: string; s: number; e: number }

interface FluidConfig {
  vocab_size: number; blank_token_id: number; pad_token_id: number; eos_token_id?: number;
  durations: number[]; max_symbols_per_step: number; decoder_layers: number; decoder_hidden_size: number;
  sampling_rate: number; n_fft: number; hop_length: number; win_length: number; preemphasis: number;
  num_mel_bins: number; max_single_pass_seconds: number; subsampling_kernel: number; subsampling_stride: number;
  subsampling_layers: number; frontend_chunk_frames: number;
}

const SPACE = "▁";

export class FluidSpeechMini {
  private constructor(
    private ort: typeof OrtNS,
    private cfg: FluidConfig,
    private filters: Float32Array,
    private pieces: Map<number, string>,
    private skip: Set<number>,
    private frontend: OrtNS.InferenceSession,
    private encoder: OrtNS.InferenceSession,
    private decoder: OrtNS.InferenceSession,
    private joint: OrtNS.InferenceSession,
  ) {}

  /** `read` returns the bytes of one model file (from the network, a cache or disk). */
  static async load(ort: typeof OrtNS, read: (f: ModelFile) => Promise<Uint8Array>,
                    opts: OrtNS.InferenceSession.SessionOptions = {}): Promise<FluidSpeechMini> {
    const text = async (f: ModelFile) => new TextDecoder().decode(await read(f));
    const cfg = JSON.parse(await text("config.json")) as FluidConfig;
    const filters = readNpyF32(await read("mel_filters.npy")).data;
    const tok = JSON.parse(await text("tokenizer.json"));
    const pieces = new Map<number, string>();
    for (const [s, i] of Object.entries(tok.model.vocab as Record<string, number>)) pieces.set(i, s);
    const skip = new Set([cfg.blank_token_id, cfg.pad_token_id]);
    for (const a of tok.added_tokens ?? []) {
      pieces.set(a.id, a.content);
      if (a.special) skip.add(a.id);
    }
    const base: OrtNS.InferenceSession.SessionOptions = { executionProviders: ["wasm"], ...opts };
    const frontend = await ort.InferenceSession.create(await read("frontend.onnx"), base);
    const encoder = await ort.InferenceSession.create(await read("encoder.onnx"), {
      ...base, externalData: [{ path: "encoder.onnx.data", data: await read("encoder.onnx.data") }],
    });
    // the decoder and joint are tiny and run once per token: on a GPU every call would wait on a round trip, so
    // they always run on the CPU (WASM)
    const cpu: OrtNS.InferenceSession.SessionOptions = { ...opts, executionProviders: ["wasm"] };
    const decoder = await ort.InferenceSession.create(await read("decoder.onnx"), cpu);
    const joint = await ort.InferenceSession.create(await read("joint.onnx"), cpu);
    return new FluidSpeechMini(ort, cfg, filters, pieces, skip, frontend, encoder, decoder, joint);
  }

  get frameSeconds(): number {
    const c = this.cfg;
    return (c.hop_length / c.sampling_rate) * c.subsampling_stride ** c.subsampling_layers;
  }

  private subsampled(frames: number): number {
    const c = this.cfg;
    const pad = Math.floor((c.subsampling_kernel - 1) / 2) * 2 - c.subsampling_kernel;
    for (let i = 0; i < c.subsampling_layers; i++) frames = Math.floor((frames + pad) / c.subsampling_stride + 1);
    return frames;
  }

  private async encode(features: Float32Array, frames: number, valid: number) {
    const { ort, cfg } = this;
    const mels = cfg.num_mel_bins, hidden = 1024;
    const factor = cfg.subsampling_stride ** cfg.subsampling_layers;
    const total = this.subsampled(frames), step = cfg.frontend_chunk_frames;
    const out = new Float32Array(total * hidden);
    // the conv front end runs in time chunks with one output frame of context on each side (same result, less memory)
    for (let a = 0; a < total; a += step) {
      const b = Math.min(a + step, total);
      const s = a ? factor * (a - 1) : 0;
      const e = Math.min(frames, factor * (b + 1));
      const n = e - s;
      const mask = new BigInt64Array(n);
      for (let i = 0; i < n; i++) mask[i] = s + i < valid ? 1n : 0n;
      const res = await this.frontend.run({
        input_features: new ort.Tensor("float32", features.subarray(s * mels, e * mels), [1, n, mels]),
        attention_mask: new ort.Tensor("int64", mask, [1, n]),
      });
      const y = res.hidden_states.data as Float32Array;
      const skip = a ? 1 : 0;
      out.set(y.subarray(skip * hidden, (skip + (b - a)) * hidden), a * hidden);
    }
    const validOut = this.subsampled(valid);
    const res = await this.encoder.run({
      hidden_states: new ort.Tensor("float32", out, [1, total, hidden]),
      length: new ort.Tensor("int64", BigInt64Array.from([BigInt(validOut)]), [1]),
    });
    return { enc: res.encoder_out.data as Float32Array, frames: total, valid: validOut, hidden };
  }

  private async greedy(enc: Float32Array, frames: number, valid: number, hidden: number) {
    const { ort, cfg } = this;
    const vocab = cfg.vocab_size, blank = cfg.blank_token_id, eos = cfg.eos_token_id;
    const L = cfg.decoder_layers, H = cfg.decoder_hidden_size;
    let h = new ort.Tensor("float32", new Float32Array(L * H), [L, 1, H]);
    let c = new ort.Tensor("float32", new Float32Array(L * H), [L, 1, H]);
    const runDecoder = async (token: number) => {
      const r = await this.decoder.run({ token: new ort.Tensor("int64", BigInt64Array.from([BigInt(token)]), [1, 1]), h_in: h, c_in: c });
      h = r.h_out as typeof h; c = r.c_out as typeof c;
      return r.decoder_out;
    };
    let decOut = await runDecoder(blank);
    const budget = cfg.max_symbols_per_step * frames - 1;
    const out: [number, number, number][] = [];
    let t = 0, steps = 0;
    for (;;) {
      const frame = Math.min(t, frames - 1);
      const r = await this.joint.run({
        encoder_frame: new ort.Tensor("float32", enc.subarray(frame * hidden, (frame + 1) * hidden), [1, hidden]),
        decoder_out: decOut,
      });
      const logits = r.logits.data as Float32Array;
      let token = 0;
      for (let i = 1; i < vocab; i++) if (logits[i] > logits[token]) token = i;
      let d = 0;
      for (let i = 1; i < cfg.durations.length; i++) if (logits[vocab + i] > logits[vocab + d]) d = i;
      let step = cfg.durations[d];
      if (token === blank && step === 0) step = 1;
      if (token !== blank) out.push([token, t, step]);
      t += step;
      steps++;
      if (t >= valid || steps >= budget || token === eos) return out;
      if (token !== blank) decOut = await runDecoder(token);
    }
  }

  private async pass(wav: Float32Array, offset: number): Promise<[string, number, number][]> {
    const { features, frames, valid } = logMel(wav, this.cfg, this.filters, this.cfg.num_mel_bins);
    const enc = await this.encode(features, frames, valid);
    const fs = this.frameSeconds;
    const pieces: [string, number, number][] = [];
    for (const [token, frame, dur] of await this.greedy(enc.enc, enc.frames, enc.valid, enc.hidden)) {
      if (this.skip.has(token)) continue;
      const s = offset + frame * fs;
      pieces.push([this.pieces.get(token) ?? "", s, s + Math.max(dur, 1) * fs]);
    }
    return pieces;
  }

  /** Cut long audio at the quietest point near every `target` seconds. */
  private split(wav: Float32Array, target = 30, search = 10): [number, number][] {
    const rate = this.cfg.sampling_rate, hop = rate / 50;
    const energy = new Float64Array(Math.ceil(wav.length / hop));
    for (let i = 0; i < energy.length; i++) {
      let s = 0;
      for (let j = i * hop; j < Math.min((i + 1) * hop, wav.length); j++) s += wav[j] * wav[j];
      energy[i] = Math.sqrt(s / hop);
    }
    const smooth = energy.map((_, i) => {
      let s = 0, n = 0;
      for (let k = i - 5; k < i + 5; k++) if (k >= 0 && k < energy.length) { s += energy[k]; n++; }
      return s / 10;
    });
    const cuts: [number, number][] = [];
    let start = 0;
    while (wav.length - start > this.cfg.max_single_pass_seconds * rate) {
      const lo = Math.floor((start + (target - search / 2) * rate) / hop);
      const hi = Math.floor((start + (target + search / 2) * rate) / hop);
      let best = lo;
      for (let i = lo; i < hi; i++) if (smooth[i] < smooth[best]) best = i;
      cuts.push([start, best * hop]);
      start = best * hop;
    }
    cuts.push([start, wav.length]);
    return cuts;
  }

  async transcribe(wav: Float32Array, onProgress?: (done: number) => void): Promise<Word[]> {
    const rate = this.cfg.sampling_rate;
    const pieces: [string, number, number][] = [];
    const cuts = this.split(wav);
    for (const [k, [a, b]] of cuts.entries()) {
      pieces.push(...(await this.pass(wav.subarray(a, b), a / rate)));
      onProgress?.((k + 1) / cuts.length);
    }
    return groupPieces(pieces);
  }
}

/** Join subword pieces into words; a piece starting with the space marker opens a new word. Numbers arrive as a
 * bare space piece then the digits ("▁", "5"), so a bare space must also open a word. */
export function groupPieces(pieces: [string, number, number][]): Word[] {
  const words: Word[] = [];
  let fresh = true;
  for (let [text, s, e] of pieces) {
    if (!text) continue;
    if (text.startsWith(SPACE)) {
      fresh = true;
      text = text.replace(/^▁+/, "");
      if (!text) continue;
    }
    if (fresh || !words.length) {
      words.push({ w: text, s, e });
      fresh = false;
    } else {
      const last = words[words.length - 1];
      last.w += text;
      last.e = Math.max(last.e, e);
    }
  }
  for (const w of words) {
    w.s = Math.round(w.s * 1000) / 1000;
    w.e = Math.round(Math.max(w.e, w.s) * 1000) / 1000;
  }
  return words;
}
