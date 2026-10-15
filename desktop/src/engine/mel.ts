// Log-mel features, matching the Python reference (and Hugging Face's ParakeetFeatureExtractor).

export interface MelConfig {
  n_fft: number;
  hop_length: number;
  win_length: number;
  preemphasis?: number;
}

const LOG_ZERO_GUARD = 2 ** -24;
const NORM_EPS = 1e-5;

/** In-place iterative radix-2 FFT on (re, im) of power-of-two length. */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Returns features [frames x mels] (row-major), the frame count and the number of valid frames. */
export function logMel(wav: Float32Array, cfg: MelConfig, filters: Float32Array, mels: number) {
  const { n_fft, hop_length: hop, win_length: win } = cfg;
  const x = new Float32Array(wav.length);
  x[0] = wav[0];
  const pre = cfg.preemphasis ?? 0;
  for (let i = 1; i < wav.length; i++) x[i] = pre ? wav[i] - pre * wav[i - 1] : wav[i];

  const window = new Float64Array(n_fft);
  const left = (n_fft - win) >> 1;
  for (let n = 0; n < win; n++) window[left + n] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (win - 1)));

  const pad = n_fft >> 1;
  const frames = 1 + Math.floor(x.length / hop);
  const bins = (n_fft >> 1) + 1;
  const out = new Float32Array(frames * mels);
  const re = new Float64Array(n_fft), im = new Float64Array(n_fft);
  const power = new Float32Array(bins);
  for (let f = 0; f < frames; f++) {
    const start = f * hop - pad;
    for (let k = 0; k < n_fft; k++) {
      const s = start + k;
      re[k] = (s >= 0 && s < x.length ? x[s] : 0) * window[k];
      im[k] = 0;
    }
    fft(re, im);
    for (let b = 0; b < bins; b++) power[b] = Math.fround(re[b] * re[b] + im[b] * im[b]);
    for (let m = 0; m < mels; m++) {
      let acc = 0;
      const row = m * bins;
      for (let b = 0; b < bins; b++) acc += power[b] * filters[row + b];
      out[f * mels + m] = Math.log(acc + LOG_ZERO_GUARD);
    }
  }

  // per-feature normalisation over the valid frames (unbiased variance), padding frames zeroed
  const valid = Math.min(Math.floor(x.length / hop), frames);
  for (let m = 0; m < mels; m++) {
    let mean = 0;
    for (let f = 0; f < valid; f++) mean += out[f * mels + m];
    mean /= valid;
    let v = 0;
    for (let f = 0; f < valid; f++) v += (out[f * mels + m] - mean) ** 2;
    const sd = Math.sqrt(v / Math.max(valid - 1, 1)) + NORM_EPS;
    for (let f = 0; f < frames; f++) out[f * mels + m] = f < valid ? (out[f * mels + m] - mean) / sd : 0;
  }
  return { features: out, frames, valid };
}

/** Minimal .npy reader for little-endian float32 arrays. */
export function readNpyF32(buf: Uint8Array): { data: Float32Array; shape: number[] } {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const major = buf[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const offset = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(buf.subarray(offset, offset + headerLen));
  if (!header.includes("'<f4'")) throw new Error(`unsupported npy dtype: ${header}`);
  const shape = (header.match(/\(([^)]*)\)/)?.[1] ?? "").split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const start = offset + headerLen;
  const data = new Float32Array(buf.slice(start).buffer);
  return { data, shape };
}
