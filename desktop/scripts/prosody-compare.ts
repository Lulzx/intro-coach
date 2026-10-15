// Compare the TS pitch/intensity tracker with Praat (exported by the Python reference) frame by frame.
import { readFileSync } from "node:fs";
import { analyse, percentile, std } from "../src/engine/prosody";
for (const pcm of process.argv.slice(2)) {
  const buf = readFileSync(pcm + ".f32");
  const wav = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const ref = JSON.parse(readFileSync(pcm + ".praat.json", "utf8"));
  const t = performance.now(); const P = analyse(wav); const ms = performance.now() - t;
  // align on time
  const at = (time: number) => Math.round((time - P.t0) / 0.01);
  let both = 0, agreeV = 0, n = 0, oct = 0, dbd: number[] = [], silAgree = 0; const ratios: number[] = [];
  ref.t.forEach((time: number, k: number) => {
    const i = at(time); if (i < 0 || i >= P.n) return; n++;
    const a = ref.f0[k] > 0, b = P.f0[i] > 0; if (a === b) agreeV++;
    if (a && b) { both++; const r = 12 * Math.log2(P.f0[i] / ref.f0[k]); ratios.push(r); if (Math.abs(r) > 5) oct++; }
    if (ref.db[k] > 0 && P.db[i] > 0) dbd.push(P.db[i] - ref.db[k]);
    if (!!ref.silent[k] === !!P.silent[i]) silAgree++;
  });
  const st = (f: ArrayLike<number>, m: number) => Array.from(f).filter((x) => x > 0).map((x) => 12 * Math.log2(x / m));
  const rs = st(ref.f0, ref.median), ts = st(P.f0, P.f0Median);
  console.log(`${pcm.split("/").pop()}: ${ms.toFixed(0)} ms | voicing agree ${(100 * agreeV / n).toFixed(1)}% | pitch diff p50 ${percentile(ratios.map(Math.abs), 50).toFixed(2)} st, octave errors ${(100 * oct / both).toFixed(1)}% | median Hz ${ref.median.toFixed(0)} vs ${P.f0Median.toFixed(0)} | range st ${(percentile(rs, 95) - percentile(rs, 5)).toFixed(1)} vs ${(percentile(ts, 95) - percentile(ts, 5)).toFixed(1)} | sd ${std(rs).toFixed(2)} vs ${std(ts).toFixed(2)} | dB diff p50 ${percentile(dbd, 50).toFixed(2)} | silence agree ${(100 * silAgree / n).toFixed(1)}%`);
}
