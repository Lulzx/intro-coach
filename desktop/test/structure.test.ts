// Same 72 labelled intros as the Python version (tests/corpus.py + tests/holdout.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { detect } from "../src/engine/nlp";

interface Item { scenario: string; note: string; text: string; expect: Record<string, boolean> }
const { corpus, holdout, fresh } = JSON.parse(readFileSync(new URL("./intros.json", import.meta.url), "utf8")) as Record<string, Item[]>;
const sentences = (t: string) => t.trim().split(/(?<=[.!?])\s+/).filter(Boolean);

export function report(items: Item[]) {
  const tally: Record<string, [number, number]> = {};
  const misses: string[] = [];
  for (const it of items) {
    const hits = detect(sentences(it.text), Object.keys(it.expect));
    for (const [k, want] of Object.entries(it.expect)) {
      const got = hits[k].found !== null;
      tally[k] ??= [0, 0];
      tally[k][0] += Number(got === want);
      tally[k][1]++;
      if (got !== want) misses.push(`${it.scenario}/${k}: expected ${want ? "present" : "absent"} (${it.note})\n     ${it.text}`);
    }
  }
  return { tally, misses };
}

test("structure checks match the labelled intros", () => {
  const { misses } = report([...corpus, ...holdout, ...fresh]);
  assert.deepEqual(misses, []);
});

if (process.argv.includes("--report")) {
  for (const [name, items] of [["corpus", corpus], ["holdout", holdout], ["fresh", fresh]] as const) {
    const { tally, misses } = report(items);
    const ok = Object.values(tally).reduce((a, [g]) => a + g, 0), n = Object.values(tally).reduce((a, [, c]) => a + c, 0);
    console.log(`${name}: ${ok}/${n} = ${Math.round((100 * ok) / n)}%`);
    for (const [k, [g, c]] of Object.entries(tally).sort()) if (g < c) console.log(`  ${k} ${g}/${c}`);
    for (const m of misses) console.log("MISS", m);
  }
}
