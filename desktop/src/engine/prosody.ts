// Pitch, loudness and silence, porting src/intro_coach/prosody.py. Python uses Praat through Parselmouth; this
// reimplements Praat's methods (Boersma 1993 autocorrelation pitch with a Viterbi path, and its intensity
// analysis) so the numbers line up with the reference implementation.

export const STEP = 0.01;

export interface Prosody {
  t0: number; // time of frame 0
  n: number;
  f0: Float32Array; // Hz, 0 where unvoiced
  db: Float32Array;
  silent: Uint8Array;
  f0Median: number;
  floor: number;
  ceiling: number;
}

// ---------------------------------------------------------------- small numeric helpers

export function percentile(values: ArrayLike<number>, p: number): number {
  const a = Array.from(values).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const k = (a.length - 1) * (p / 100);
  const lo = Math.floor(k), hi = Math.ceil(k);
  return a[lo] + (a[hi] - a[lo]) * (k - lo);
}
export const median = (v: ArrayLike<number>) => percentile(v, 50);
export const mean = (v: ArrayLike<number>) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s / v.length; };
export function std(v: ArrayLike<number>): number {
  const m = mean(v);
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] - m) ** 2;
  return Math.sqrt(s / v.length);
}

function fftInPlace(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc;
      }
    }
  }
}

/** Autocorrelation of x (length n) for lags 0..n-1, via FFT. */
function autocorr(x: Float64Array, size: number): Float64Array {
  const re = new Float64Array(size), im = new Float64Array(size);
  re.set(x);
  fftInPlace(re, im);
  for (let i = 0; i < size; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
  fftInPlace(re, im, true);
  for (let i = 0; i < size; i++) re[i] /= size;
  return re;
}

// ---------------------------------------------------------------- pitch (Praat "To Pitch (ac)")

interface Candidate { f: number; r: number }

function pitchAC(wav: Float32Array, rate: number, floor: number, ceiling: number) {
  const silenceThreshold = 0.03, voicingThreshold = 0.45, octaveCost = 0.01;
  const octaveJumpCost = 0.35, voicedUnvoicedCost = 0.14, maxCandidates = 15;
  const duration = wav.length / rate;
  const windowLen = 3 / floor; // three periods of the lowest pitch
  const nSamp = Math.floor(windowLen * rate) | 1;
  const half = nSamp >> 1;
  const nFrames = Math.max(1, Math.floor((duration - windowLen) / STEP) + 1);
  const t0 = (duration - (nFrames - 1) * STEP) / 2;
  let size = 1;
  while (size < nSamp * 2) size <<= 1;

  const win = new Float64Array(nSamp);
  for (let i = 0; i < nSamp; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / (nSamp + 1));
  const rw = autocorr(win, size);

  let globalPeak = 0;
  for (let i = 0; i < wav.length; i++) globalPeak = Math.max(globalPeak, Math.abs(wav[i]));
  const minLag = Math.max(2, Math.floor(rate / ceiling));
  const maxLag = Math.min(nSamp - 1, Math.ceil(rate / floor));

  const frames: Candidate[][] = [];
  const seg = new Float64Array(nSamp);
  for (let f = 0; f < nFrames; f++) {
    const centre = Math.round((t0 + f * STEP) * rate);
    let m = 0, localPeak = 0;
    for (let i = 0; i < nSamp; i++) {
      const s = centre - half + i;
      seg[i] = s >= 0 && s < wav.length ? wav[s] : 0;
      m += seg[i];
    }
    m /= nSamp;
    for (let i = 0; i < nSamp; i++) {
      seg[i] -= m;
      localPeak = Math.max(localPeak, Math.abs(seg[i]));
      seg[i] *= win[i];
    }
    const cands: Candidate[] = [];
    // the unvoiced candidate gets stronger as the frame gets quieter relative to the loudest part of the recording
    const intensity = globalPeak > 0 ? localPeak / globalPeak : 0;
    const unvoiced = voicingThreshold + Math.max(0, 2 - intensity / (silenceThreshold / (1 + voicingThreshold)));
    cands.push({ f: 0, r: unvoiced });
    if (localPeak > 0) {
      const ra = autocorr(seg, size);
      const r = (lag: number) => (ra[lag] / ra[0]) / (rw[lag] / rw[0]);
      const peaks: Candidate[] = [];
      for (let lag = minLag; lag <= maxLag; lag++) {
        const a = r(lag - 1), b = r(lag), c = r(lag + 1);
        if (b > 0.5 * voicingThreshold && b >= a && b >= c) {
          const denom = a - 2 * b + c;
          const shift = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
          const exact = lag + Math.max(-0.5, Math.min(0.5, shift));
          const peak = Math.min(1, b - 0.25 * (a - c) * shift);
          const freq = rate / exact;
          if (freq >= floor && freq <= ceiling) {
            peaks.push({ f: freq, r: peak });
          }
        }
      }
      // Praat's octave cost: strength = R - octaveCost * log2(floor / f), a slight preference for higher candidates
      for (const p of peaks) p.r -= octaveCost * Math.log2(floor / p.f);
      peaks.sort((x, y) => y.r - x.r);
      cands.push(...peaks.slice(0, maxCandidates - 1));
    }
    frames.push(cands);
  }

  // Viterbi over candidates: penalise octave jumps and switching between voiced and unvoiced
  const cost = (a: Candidate, b: Candidate) => {
    if (a.f === 0 && b.f === 0) return 0;
    if (a.f === 0 || b.f === 0) return voicedUnvoicedCost;
    return octaveJumpCost * Math.abs(Math.log2(a.f / b.f));
  };
  let delta = frames[0].map((c) => c.r);
  const back: Int32Array[] = [];
  for (let f = 1; f < nFrames; f++) {
    const cur = frames[f], prev = frames[f - 1];
    const nd = new Array<number>(cur.length);
    const bp = new Int32Array(cur.length);
    for (let j = 0; j < cur.length; j++) {
      let best = -Infinity, arg = 0;
      for (let i = 0; i < prev.length; i++) {
        const v = delta[i] - cost(prev[i], cur[j]);
        if (v > best) { best = v; arg = i; }
      }
      nd[j] = best + cur[j].r;
      bp[j] = arg;
    }
    back.push(bp);
    delta = nd;
  }
  const f0 = new Float32Array(nFrames);
  let k = delta.indexOf(Math.max(...delta));
  for (let f = nFrames - 1; f >= 0; f--) {
    f0[f] = frames[f][k].f;
    if (f > 0) k = back[f - 1][k];
  }
  return { t0, f0 };
}

// ---------------------------------------------------------------- intensity (Praat "To Intensity")

function besselI0(x: number): number {
  let s = 1, term = 1;
  for (let k = 1; k < 50; k++) { term *= (x / (2 * k)) ** 2; s += term; if (term < 1e-12 * s) break; }
  return s;
}

function intensity(wav: Float32Array, rate: number, minPitch: number, times: (i: number) => number, n: number) {
  const windowDur = 3.2 / minPitch;
  const half = Math.floor((windowDur / 2) * rate);
  const len = 2 * half + 1;
  // Praat uses a Kaiser-20 window (bessel β = 2π·sqrt(5²-0.5²) ≈ 20)
  const beta = 2 * Math.PI * Math.sqrt(5 * 5 - 0.25);
  const w = new Float64Array(len);
  const i0b = besselI0(beta);
  for (let i = 0; i < len; i++) {
    const x = (i - half) / half;
    w[i] = besselI0(beta * Math.sqrt(Math.max(0, 1 - x * x))) / i0b;
  }
  const db = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    const c = Math.round(times(f) * rate);
    let sw = 0, sx = 0;
    for (let i = 0; i < len; i++) {
      const s = c - half + i;
      if (s < 0 || s >= wav.length) continue;
      sw += w[i]; sx += w[i] * wav[s];
    }
    const m = sw ? sx / sw : 0;
    let e = 0;
    sw = 0;
    for (let i = 0; i < len; i++) {
      const s = c - half + i;
      if (s < 0 || s >= wav.length) continue;
      e += w[i] * (wav[s] - m) ** 2; sw += w[i];
    }
    const meanSq = sw ? e / sw : 0;
    db[f] = meanSq > 0 ? 10 * Math.log10(meanSq / 4e-10) : -300;
  }
  return db;
}

// ---------------------------------------------------------------- analysis entry point

/** Halve the sample rate with a short low-pass FIR (pitch lives far below 4 kHz; Praat also downsamples). */
function decimate2(x: Float32Array): Float32Array {
  const h = [-0.0106, 0, 0.0329, 0, -0.0890, 0, 0.3126, 0.5, 0.3126, 0, -0.0890, 0, 0.0329, 0, -0.0106];
  const half = h.length >> 1;
  const out = new Float32Array(x.length >> 1);
  for (let i = 0; i < out.length; i++) {
    let acc = 0;
    const c = 2 * i;
    for (let k = 0; k < h.length; k++) {
      const j = c + k - half;
      if (j >= 0 && j < x.length) acc += h[k] * x[j];
    }
    out[i] = acc;
  }
  return out;
}

export function analyse(wav: Float32Array, rate = 16000): Prosody {
  // pitch is tracked at half the rate: four times less autocorrelation work, same contour
  const low = rate >= 16000 ? decimate2(wav) : wav;
  const lowRate = rate >= 16000 ? rate / 2 : rate;
  // Two-pass pitch range (De Looze & Hirst): a wide first pass, then a range fitted to this speaker
  const first = pitchAC(low, lowRate, 60, 600).f0;
  const v = Array.from(first).filter((x) => x > 0);
  let floor = 60, ceiling = 600;
  if (v.length > 20) {
    floor = Math.max(50, 0.75 * percentile(v, 15));
    ceiling = Math.min(700, 1.5 * percentile(v, 85));
  }
  const { t0, f0 } = pitchAC(low, lowRate, floor, ceiling);
  const n = f0.length;
  const db = intensity(wav, rate, Math.max(floor, 75), (i) => t0 + i * STEP, n);

  const finite = Array.from(db).filter((x) => x > -300);
  const loud = percentile(finite, 95), quiet = percentile(finite, 10);
  const threshold = Math.max(loud - 25, quiet + 4);
  const silent = new Uint8Array(n);
  for (let i = 0; i < n; i++) silent[i] = db[i] < threshold && f0[i] === 0 ? 1 : 0;
  const voiced = Array.from(f0).filter((x) => x > 0);
  return { t0, n, f0, db, silent, f0Median: voiced.length ? median(voiced) : 120, floor, ceiling };
}

// ---------------------------------------------------------------- queries used by the analysis

export const timeOf = (P: Prosody, i: number) => P.t0 + i * STEP;

/** Frame index range [i0, i1) covering times a <= t < b. */
export function range(P: Prosody, a: number, b: number): [number, number] {
  const i0 = Math.max(0, Math.ceil((a - P.t0) / STEP - 1e-9));
  const i1 = Math.min(P.n, Math.ceil((b - P.t0) / STEP - 1e-9));
  return [i0, Math.max(i0, i1)];
}

export function voiced(P: Prosody, a: number, b: number): number[] {
  const [i0, i1] = range(P, a, b);
  const out: number[] = [];
  for (let i = i0; i < i1; i++) if (P.f0[i] > 0) out.push(P.f0[i]);
  return out;
}

export function loudFrames(P: Prosody, a: number, b: number): number[] {
  const [i0, i1] = range(P, a, b);
  const out: number[] = [];
  for (let i = i0; i < i1; i++) if (!P.silent[i]) out.push(P.db[i]);
  return out;
}

export const semitones = (P: Prosody, hz: number) => 12 * Math.log2(hz / P.f0Median);

/** [start, end] times of runs where mask(i) holds for at least minLen seconds. */
export function runs(P: Prosody, mask: (i: number) => boolean, minLen: number, i0 = 0, i1 = P.n): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (let i = i0; i < i1; i++) {
    if (mask(i) && start < 0) start = i;
    else if (!mask(i) && start >= 0) {
      if ((i - start) * STEP >= minLen) out.push([timeOf(P, start), timeOf(P, i - 1) + STEP]);
      start = -1;
    }
  }
  if (start >= 0 && (i1 - start) * STEP >= minLen) out.push([timeOf(P, start), timeOf(P, i1 - 1) + STEP]);
  return out;
}
