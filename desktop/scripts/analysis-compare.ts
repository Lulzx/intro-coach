// Run the full TS analysis on the same transcript + audio as Python and compare the reports.
import { readFileSync } from "node:fs";
import { analyse as prosody } from "../src/engine/prosody";
import { analyse } from "../src/engine/analysis";
for (const base of process.argv.slice(2)) {
  const buf = readFileSync(base + ".f32"); const wav = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const words = JSON.parse(readFileSync(base + ".py.json", "utf8"));
  const py = JSON.parse(readFileSync(base + ".report.py.json", "utf8"));
  const t = performance.now();
  const ts: any = analyse(words, prosody(wav), wav.length / 16000, "interview", 75);
  const ms = performance.now() - t;
  console.log(`\n== ${base.split("/").pop()} (${ms.toFixed(0)} ms)  overall ${py.overall} vs ${ts.overall}`);
  console.log("  scores py", JSON.stringify(py.scores), "\n         ts", JSON.stringify(ts.scores));
  const keys = ["wpm", "articulation_wpm", "silence_pct", "pauses", "hesitations", "fillers", "hidden_fillers", "stretched_words", "pitch_median_hz", "pitch_range_st", "pitch_sd_st", "monotone_secs", "uptalk", "trailing_off", "i_have_been", "long_sentences"];
  console.log("  metrics (py/ts):", keys.map((k) => `${k} ${py.metrics[k]}/${ts.metrics[k]}`).join(", "));
  console.log("  structure py", py.structure.elements.map((e: any) => e.found ? "✓" : "✗").join(""), "ts", ts.structure.elements.map((e: any) => e.found ? "✓" : "✗").join(""));
  console.log("  tips py:", py.tips.map((t: any) => t.title).join(" | "));
  console.log("  tips ts:", ts.tips.map((t: any) => t.title).join(" | "));
  console.log("  insights py concrete", py.insights.concrete.map((c: any) => c.text), "vague", py.insights.vague, "I/we", py.insights.i_did, py.insights.we_did, "action", py.insights.action_share);
  console.log("  insights ts concrete", ts.insights.concrete.map((c: any) => c.text), "vague", ts.insights.vague, "I/we", ts.insights.i_did, ts.insights.we_did, "action", ts.insights.action_share);
  console.log("  numbers py", py.insights.numbers.map((n: any) => n.word), "ts", ts.insights.numbers.map((n: any) => n.word), "| overused py", py.metrics.overused, "ts", ts.metrics.overused);
}
