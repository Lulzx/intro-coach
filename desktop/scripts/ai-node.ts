// Live check of the AI path: models.dev catalogue -> AI SDK provider -> structured feedback.
import { readFileSync } from "node:fs";
import { loadCatalog, model, feedback, check, supported, DEFAULT_PROVIDER, DEFAULT_MODEL } from "../src/ai";
const cat = await loadCatalog(fetch);
console.log("providers usable:", Object.keys(cat).length, "| openrouter models:", Object.keys(cat.openrouter?.models ?? {}).length);
const lm = model(cat[DEFAULT_PROVIDER], DEFAULT_MODEL, process.env.OPENROUTER_API_KEY!, fetch);
await check(lm); console.log("key check ok");
try { await check(model(cat.openrouter, DEFAULT_MODEL, "sk-or-v1-bogus", fetch)); } catch (e) { console.log("bad key ->", (e as Error).message); }
const report = JSON.parse(readFileSync(process.argv[2], "utf8"));
const t = performance.now();
const fb = await feedback(lm, report, "");
console.log(`feedback in ${((performance.now() - t) / 1000).toFixed(1)}s from ${fb.model}`);
console.log(fb.summary, "\n", fb.fixes.slice(0, 3), "\n rewrite:", fb.rewrite.slice(0, 300));
