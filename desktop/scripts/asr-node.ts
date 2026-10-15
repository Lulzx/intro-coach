// Node check: run the TS engine with onnxruntime-web (WASM) and compare with the Python transcript.
import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { FluidSpeechMini, type ModelFile } from "../src/engine/fluid";

const dir = process.env.FLUID_MODEL_DIR ?? join(homedir(), ".intro-coach/models/fluid-speech-mini-onnx");
const [pcm, ref] = process.argv.slice(2);
ort.env.wasm.numThreads = Number(process.env.THREADS ?? 1);
let t = performance.now();
const engine = await FluidSpeechMini.load(ort, async (f: ModelFile) => new Uint8Array(readFileSync(join(dir, f))));
console.log("load", ((performance.now() - t) / 1000).toFixed(1), "s");
const buf = readFileSync(pcm);
const wav = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
t = performance.now();
const words = await engine.transcribe(wav);
console.log("transcribe", ((performance.now() - t) / 1000).toFixed(1), "s for", (wav.length / 16000).toFixed(0), "s audio, threads", ort.env.wasm.numThreads);
const py = JSON.parse(readFileSync(ref, "utf8"));
const ts = words.map((w) => w.w).join(" "), pt = py.map((w: any) => w.w).join(" ");
console.log("same text:", ts === pt);
if (ts !== pt) console.log("TS:", ts, "\nPY:", pt);
let dmax = 0;
if (words.length === py.length) for (let i = 0; i < words.length; i++) dmax = Math.max(dmax, Math.abs(words[i].s - py[i].s), Math.abs(words[i].e - py[i].e));
console.log("max timestamp diff", dmax.toFixed(3), "s");
