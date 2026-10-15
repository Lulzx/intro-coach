// Intro Coach desktop UI. Same screens as the Python web app; everything that went through the server now runs
// here: decoding, analysis (in a worker), storage (IndexedDB) and the AI coach (AI SDK + models.dev).
import "./style.css";
import { SCENARIOS } from "./engine/scenarios";
import type { Report } from "./engine/analysis";
import * as store from "./store";
import * as ai from "./ai";
import { appFetch, decodeAudio, ensureModel, isMac, isTauri, readModel, startNative, transcribeNative } from "./platform";
import AnalysisWorker from "./worker?worker";

/* eslint-disable @typescript-eslint/no-explicit-any */
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
type Kid = Node | string | number | null | undefined | false | Kid[];
function el(tag: string, attrs: Record<string, any> = {}, ...kids: Kid[]): HTMLElement {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, String(v));
  }
  const add = (k: Kid) => {
    if (Array.isArray(k)) k.forEach(add);
    else if (k !== null && k !== undefined && k !== false) n.append(typeof k === "object" ? k : String(k));
  };
  kids.forEach(add);
  return n;
}
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const local = {
  get<T>(k: string, d: T): T { try { const v = localStorage.getItem("ic:" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k: string, v: unknown) { try { localStorage.setItem("ic:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const ENGINE = "Fluid Speech Mini by FluidVoice";
let scenario = local.get("scenario", "interview");
let target = 75;
let report: any = null;
let takes: store.TakeSummary[] = [];
let audioUrl: string | null = null;

// ---------------------------------------------------------------- analysis worker + speech model

const worker = new AnalysisWorker();
let nextId = 1;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; onProgress?: (stage: string, p: number) => void }>();
let backend = "";
let mode: "coreml" | "onnx" = "onnx";
let modelReady: Promise<void>;

worker.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "loaded") { backend = m.backend; return; }
  const job = pending.get(m.id);
  if (!job) return;
  if (m.type === "progress") job.onProgress?.(m.stage, m.p);
  else if (m.type === "result") { pending.delete(m.id); job.resolve(m); }
  else if (m.type === "error") { pending.delete(m.id); job.reject(new Error(m.message)); }
};

function startModel(): Promise<void> {
  const chip = $("modelChip");
  const show = (text: string, p?: number) => {
    chip.hidden = false;
    chip.replaceChildren(el("span", {}, text), ...(p === undefined ? [] : [el("div", { class: "bar" }, el("div", { style: `width:${Math.round(p * 100)}%` }))]));
  };
  const progress = (label: string) => (loaded: number, total: number) => {
    if (total > 1 && loaded < total) show(`Downloading the speech model (${label}, first run): ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`, loaded / total);
  };
  return (async () => {
    if (isTauri && isMac) {
      try {
        await startNative(progress("Core ML"));
        mode = "coreml";
        chip.hidden = true;
        $("modelInfo").textContent = "Fluid Speech Mini by FluidVoice, running on the Neural Engine (Core ML).";
        return;
      } catch (e) {
        console.warn("Core ML unavailable, using ONNX:", e);
      }
    }
    let shown = false;
    await ensureModel((loaded, total) => {
      if (total > 1 && loaded < total) { shown = true; show(`Downloading the speech model (first run): ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`, loaded / total); }
    });
    if (shown) show("Loading the speech model...");
    const files = await readModel();
    await new Promise<void>((resolve, reject) => {
      const onMsg = (e: MessageEvent) => {
        if (e.data.type === "loaded") { worker.removeEventListener("message", onMsg); resolve(); }
        if (e.data.type === "error" && e.data.id === -1) { worker.removeEventListener("message", onMsg); reject(new Error(e.data.message)); }
      };
      worker.addEventListener("message", onMsg);
      // ?backend=wasm|webgpu overrides the automatic choice (for testing)
      const prefer = new URLSearchParams(location.search).get("backend") ?? (local.get("gpu", false) ? "webgpu" : "wasm");
      worker.postMessage({ type: "load", files, backend: prefer }, Object.values(files));
    });
    chip.hidden = true;
    $("modelInfo").textContent = `Fluid Speech Mini by FluidVoice, running on ${backend}.`;
  })().catch((e) => {
    show(`Couldn't load the speech model: ${(e as Error).message}`);
    throw e;
  });
}

async function analyseInWorker(wav: Float32Array, onProgress: (stage: string, p: number) => void): Promise<any> {
  const id = nextId++;
  // on macOS the Core ML helper transcribes (Neural Engine); the worker measures the audio and coaches
  const native = mode === "coreml" ? await transcribeNative(wav) : null;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    if (native) worker.postMessage({ type: "analyseWords", id, words: native.words, asr_s: native.ms / 1000, wav, scenario, target }, [wav.buffer]);
    else worker.postMessage({ type: "analyse", id, wav, scenario, target }, [wav.buffer]);
  });
}

// ---------------------------------------------------------------- setup

async function init() {
  const box = $("scenarios");
  for (const [id, s] of Object.entries(SCENARIOS)) {
    box.append(el("button", { class: "chip", role: "radio", "data-id": id, onclick: () => pickScenario(id) }, s.label));
  }
  pickScenario(SCENARIOS[scenario] ? scenario : "interview", true);
  $<HTMLInputElement>("target").oninput = (e) => setTarget(+(e.target as HTMLInputElement).value);
  $<HTMLTextAreaElement>("notes").value = local.get("notes", "");
  $("notes").oninput = (e) => local.set("notes", (e.target as HTMLTextAreaElement).value);
  $("recBtn").onclick = toggleRecording;
  $("uploadBtn").onclick = () => $("file").click();
  $("file").onchange = (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) submitFile(f); };
  setupDrop();
  $("historyBtn").onclick = () => toggleHistory(true);
  $("historyClose").onclick = () => toggleHistory(false);
  $("settingsBtn").onclick = () => toggleSettings(true);
  $("settingsClose").onclick = () => toggleSettings(false);
  $("keyForm").onsubmit = saveKey;
  $("keyShow").onclick = () => {
    const i = $<HTMLInputElement>("keyInput"); i.type = i.type === "password" ? "text" : "password";
    $("keyShow").textContent = i.type === "password" ? "Show" : "Hide";
  };
  $("providerSel").onchange = () => { renderModels(); renderKeyState(); persistChoice(); };
  $("modelSel").onchange = () => { persistChoice(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { toggleHistory(false); toggleSettings(false); }
    if (e.code === "Space" && !/TEXTAREA|INPUT|SELECT|BUTTON|AUDIO/.test((document.activeElement as HTMLElement)?.tagName ?? "")) {
      e.preventDefault(); toggleRecording();
    }
  });
  $("aiBtn").onclick = getFeedback;
  const gpu = $<HTMLInputElement>("gpuToggle");
  gpu.checked = local.get("gpu", false);
  gpu.onchange = () => { local.set("gpu", gpu.checked); $("modelInfo").textContent = "Restart the app to switch."; };
  window.addEventListener("resize", () => report && !report.empty && drawChart());
  modelReady = startModel();
  modelReady.catch(() => { /* shown in the chip; retried on the next take */ });
  await loadTakes();
  loadCatalog();
  if (isTauri) selfTest();
}

/** INTRO_COACH_SELFTEST=<file>: analyse it through the real pipeline, report and quit (see lib.rs). */
async function selfTest() {
  const { invoke } = await import("@tauri-apps/api/core");
  let bytes: ArrayBuffer;
  try { bytes = await invoke<ArrayBuffer>("selftest_file"); } catch { return; }
  const t = performance.now();
  let summary: Record<string, unknown>;
  try {
    await modelReady;
    await submit(new Blob([bytes]), "self-test");
    summary = { ok: !!report && !report.empty, mode, backend, isolated: crossOriginIsolated, mic: !!navigator.mediaDevices?.getUserMedia,
      seconds: Math.round(performance.now() - t) / 1000, timing: report?.timing, overall: report?.overall, transcript: report?.transcript?.slice(0, 120),
      status: $("status").textContent };
  } catch (e) {
    summary = { ok: false, error: (e as Error).message };
  }
  await invoke("selftest_done", { summary: JSON.stringify(summary) });
}

function pickScenario(id: string, initial = false) {
  scenario = id; local.set("scenario", id);
  document.querySelectorAll<HTMLElement>("#scenarios .chip").forEach((c) => c.setAttribute("aria-checked", String(c.dataset.id === id)));
  const sc = SCENARIOS[id];
  $("prompt").textContent = sc.prompt;
  $("shapeTitle").textContent = sc.shape.title;
  $("shapeSteps").replaceChildren(...sc.shape.steps.map((s) =>
    el("li", {}, el("div", {}, el("b", {}, s.name), el("div", {}, s.say), el("span", { class: "ex" }, `“${s.example}”`)))));
  $("shapeLand").textContent = sc.shape.land;
  setTarget(initial ? local.get("target:" + id, sc.target) : sc.target);
}
function setTarget(v: number) {
  target = v; $<HTMLInputElement>("target").value = String(v); $("targetVal").textContent = `${v}s`;
  $("timerTarget").textContent = `/ ${fmt(v)}`; local.set("target:" + scenario, v);
}

// ---------------------------------------------------------------- recording

// Raw PCM capture through an AudioWorklet. A microphone that has just been opened (or just been granted on macOS)
// delivers pure digital silence for a few seconds; the take only starts once real sound arrives, so nothing the
// speaker says is lost and the timer doesn't run on dead air.
const CAPTURE_WORKLET = `registerProcessor("capture", class extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
});`;

interface Rec { stop: () => void }
let rec: Rec | null = null;

async function toggleRecording() {
  if (rec) return rec.stop();
  if (mode === "coreml") return nativeRecording();
  // created inside the click so the platform lets it start running
  const ctx = new AudioContext();
  ctx.resume();
  let stream: MediaStream;
  try {
    // no voice processing: it changes levels while it adapts and can restart the input after it starts
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  } catch (e) {
    ctx.close();
    return status(`Microphone unavailable: ${(e as Error).message}`, true);
  }
  const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "text/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  await ctx.resume();
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "capture");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(node).connect(mute).connect(ctx.destination); // keeps the graph pulling audio without playing it

  const chunks: Float32Array[] = [];
  let live = false, started = 0, level = 0, raf = 0;
  const opened = performance.now();
  status("Starting the microphone...");
  $("recBtn").classList.add("on"); $("recBtn").setAttribute("aria-label", "Stop recording");
  ($("progressBar").parentElement!.querySelector(".tick") as HTMLElement).style.left = `${100 / 1.5}%`;

  node.port.onmessage = (e: MessageEvent<Float32Array>) => {
    const ch = e.data;
    let peak = 0, sum = 0;
    for (let i = 0; i < ch.length; i++) { const a = Math.abs(ch[i]); if (a > peak) peak = a; sum += ch[i] * ch[i]; }
    level = Math.sqrt(sum / ch.length);
    if (!live) {
      // a live microphone always has some noise; exact zeros mean the device isn't delivering yet
      if (peak < 1e-6) return;
      live = true;
      started = performance.now();
      status(`Recording. Answer: “${SCENARIOS[scenario].prompt}”`);
    }
    chunks.push(ch);
  };

  const tick = () => {
    if (live) {
      const s = (performance.now() - started) / 1000;
      $("timer").textContent = fmt(s);
      // the bar spans 1.5x the target so going over is visible; the tick marks the target itself
      $("progressBar").style.width = Math.min(100, (s / (target * 1.5)) * 100) + "%";
      $("progressBar").parentElement!.classList.toggle("over", s > target);
    } else if (performance.now() - opened > 10000) {
      finish(false);
      return status("The microphone didn't send any sound. Check the input device in your system settings.", true);
    }
    const db = 20 * Math.log10(level + 1e-9);
    $("meterBar").style.width = Math.max(0, Math.min(100, (db + 60) * 2)) + "%";
    raf = requestAnimationFrame(tick);
  };

  const finish = (keep: boolean) => {
    cancelAnimationFrame(raf);
    node.port.onmessage = null;
    source.disconnect(); node.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const rate = ctx.sampleRate;
    ctx.close();
    rec = null; $("recBtn").classList.remove("on"); $("recBtn").setAttribute("aria-label", "Start recording");
    $("meterBar").style.width = "0";
    if (!keep) return;
    const n = chunks.reduce((a, c) => a + c.length, 0);
    if (n < rate * 1.5) return status("That was too short. Try again.", true);
    const pcm = new Float32Array(n);
    let o = 0;
    for (const c of chunks) { pcm.set(c, o); o += c.length; }
    submitPcm(pcm, rate);
  };
  rec = { stop: () => finish(true) };
  tick();
}

/** macOS: record through the native helper (AVAudioEngine, plain input). WKWebView's getUserMedia opens the Mac
 * microphone with voice processing and restarts it a few seconds later, leaving the start of the first take far
 * too quiet. The timer shows the audio actually captured. */
async function nativeRecording() {
  const { invoke } = await import("@tauri-apps/api/core");
  try {
    await invoke("native_rec_start");
  } catch (e) {
    return status(`Microphone unavailable: ${String(e)}`, true);
  }
  status("Starting the microphone...");
  $("recBtn").classList.add("on"); $("recBtn").setAttribute("aria-label", "Stop recording");
  ($("progressBar").parentElement!.querySelector(".tick") as HTMLElement).style.left = `${100 / 1.5}%`;
  let stopped = false, live = false;
  const opened = performance.now();
  const poll = async () => {
    if (stopped) return;
    try {
      const l = await invoke<{ live: boolean; seconds: number; rms: number }>("native_rec_level");
      if (l.live && !live) { live = true; status(`Recording. Answer: “${SCENARIOS[scenario].prompt}”`); }
      if (live) {
        $("timer").textContent = fmt(l.seconds);
        $("progressBar").style.width = Math.min(100, (l.seconds / (target * 1.5)) * 100) + "%";
        $("progressBar").parentElement!.classList.toggle("over", l.seconds > target);
      } else if (performance.now() - opened > 10000) {
        rec?.stop();
        return status("The microphone didn't send any sound. Check the input device in System Settings.", true);
      }
      $("meterBar").style.width = Math.max(0, Math.min(100, (20 * Math.log10(l.rms + 1e-9) + 60) * 2)) + "%";
    } catch { /* keep polling */ }
    setTimeout(poll, 80);
  };
  rec = {
    stop: async () => {
      stopped = true;
      rec = null; $("recBtn").classList.remove("on"); $("recBtn").setAttribute("aria-label", "Start recording");
      $("meterBar").style.width = "0";
      try {
        const bytes = await invoke<ArrayBuffer>("native_rec_stop");
        const wav = new Float32Array(bytes);
        if (wav.length < 16000 * 1.5) return status("That was too short. Try again.", true);
        await submit(wavBlob(wav, 16000), undefined, wav);
      } catch (e) {
        status(`Couldn't finish the recording: ${String(e)}`, true);
      }
    },
  };
  poll();
}

/** 16 kHz mono samples from any rate, through the platform's resampler. */
async function resample(pcm: Float32Array, rate: number, to = 16000): Promise<Float32Array> {
  if (rate === to) return pcm;
  const off = new OfflineAudioContext(1, Math.ceil((pcm.length * to) / rate), to);
  const buf = off.createBuffer(1, pcm.length, rate);
  buf.copyToChannel(pcm as Float32Array<ArrayBuffer>, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}

/** 16-bit mono WAV, for playback and history. */
function wavBlob(pcm: Float32Array, rate: number): Blob {
  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const text = (o: number, s: string) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  text(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true); out.setUint32(24, rate, true);
  out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true); text(36, "data");
  out.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  return new Blob([out.buffer], { type: "audio/wav" });
}

async function submitPcm(pcm: Float32Array, rate: number) {
  const wav = await resample(pcm, rate);
  await submit(wavBlob(wav, 16000), undefined, wav);
}

function status(msg: string, error = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", error);
}

const MEDIA_EXT = /\.(m4a|mp3|wav|webm|ogg|oga|opus|flac|aac|mp4|mov|mkv|caf|aiff?)$/i;
function submitFile(f: File) {
  if (rec) return status("Stop the recording first.", true);
  if (!(/^(audio|video)\//.test(f.type) || MEDIA_EXT.test(f.name))) return status(`${f.name} doesn't look like an audio or video file.`, true);
  if (f.size > 500 * 1024 * 1024) return status("That file is over 500 MB. Trim it to your intro first.", true);
  submit(f, f.name);
}

function setupDrop() {
  const card = document.querySelector(".practice")!, drop = $("drop");
  let depth = 0;
  const hasFiles = (e: DragEvent) => [...(e.dataTransfer?.types || [])].includes("Files");
  card.addEventListener("dragenter", (e) => { if (!hasFiles(e as DragEvent)) return; e.preventDefault(); depth++; drop.hidden = false; });
  card.addEventListener("dragover", (e) => { if (hasFiles(e as DragEvent)) e.preventDefault(); });
  card.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; drop.hidden = true; } });
  card.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; drop.hidden = true;
    const f = (e as DragEvent).dataTransfer?.files[0];
    if (f) submitFile(f);
  });
  // a file dropped outside the card shouldn't make the app navigate away
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}

async function submit(blob: Blob, fileLabel?: string, decoded?: Float32Array) {
  $<HTMLButtonElement>("recBtn").disabled = true; $<HTMLButtonElement>("uploadBtn").disabled = true;
  try {
    status(fileLabel ? `Reading ${fileLabel}...` : "Reading your take...");
    const wav = decoded ?? (await decodeAudio(blob));
    if (wav.length < 8000) throw new Error("the recording is shorter than half a second");
    status("Waiting for the speech model...");
    await (modelReady = modelReady.catch(() => startModel()));
    const res = await analyseInWorker(wav, (stage, p) =>
      status(stage === "transcribe" ? `Listening back with ${ENGINE}... ${Math.round(p * 100)}%` : "Coaching..."));
    const r: Report = res.report;
    const now = new Date();
    r.id = now.toISOString().replace(/[-:T]/g, "").slice(0, 14) + "-" + Math.random().toString(16).slice(2, 6);
    r.created = now.toISOString();
    r.engine = "fluid-mini";
    r.timing = res.timing;
    await store.saveTake(r, blob);
    status("");
    await loadTakes();
    await showReport(r);
    $("results").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    status(`Couldn't analyse that take: ${(e as Error).message}`, true);
  } finally {
    $<HTMLButtonElement>("recBtn").disabled = false; $<HTMLButtonElement>("uploadBtn").disabled = false;
    $<HTMLInputElement>("file").value = "";
  }
}

// ---------------------------------------------------------------- report

const LABELS: Record<string, string> = { pace: "Pace", fluency: "Fluency", voice: "Voice", structure: "Structure", language: "Language" };
const verdict = (s: number) => (s >= 80 ? "Strong intro" : s >= 60 ? "Getting there" : s >= 40 ? "Needs work" : "Early draft");

async function showReport(r: any) {
  report = r;
  $("results").hidden = false;
  if (r.empty) {
    $("overall").textContent = "-"; $("overallLabel").textContent = "No speech detected";
    return;
  }
  $("overall").textContent = r.overall;
  $("overallLabel").textContent = verdict(r.overall);
  const prev = takes.find((t) => t.id < r.id && t.scenario === r.scenario);
  $("overallDelta").textContent = prev
    ? `${r.overall - prev.overall >= 0 ? "+" : ""}${r.overall - prev.overall} vs your previous ${SCENARIOS[r.scenario]?.label.toLowerCase() || ""} take`
    : `${SCENARIOS[r.scenario]?.label || r.scenario} · ${new Date(r.created).toLocaleString()}`;
  $("subscores").replaceChildren(...Object.entries(r.scores as Record<string, number>).map(([k, v]) =>
    el("div", { class: "sub" }, el("span", {}, LABELS[k]), el("div", { class: "bar" }, el("div", { style: `width:${v}%` })), el("span", { class: "val" }, v))));

  const m = r.metrics;
  const flag = (ok: boolean, warn: boolean, text: string) => el("span", { class: `flag ${ok ? "good" : warn ? "warn" : "bad"}` }, text);
  const ratio = m.speaking_time / r.target_seconds;
  const tiles: [string, string | number, string, string, HTMLElement][] = [
    ["Length", `${Math.round(m.speaking_time)}`, "s", `target ${r.target_seconds}s`,
      flag(ratio >= 0.8 && ratio <= 1.2, ratio >= 0.6 && ratio <= 1.5, ratio > 1.2 ? "too long" : ratio < 0.8 ? "short" : "on target")],
    ["Pace", m.wpm, "wpm", `${m.articulation_wpm} while talking · aim 130-165`,
      flag(m.wpm >= 130 && m.wpm <= 165, m.wpm >= 115 && m.wpm <= 180, m.wpm < 130 ? "slow" : m.wpm > 165 ? "fast" : "good")],
    ["Silence", m.silence_pct, "%", "aim 15-20%", flag(m.silence_pct <= 22, m.silence_pct <= 28, m.silence_pct > 22 ? "too much" : "good")],
    ["Mid-thought pauses", m.hesitations, "", "pauses ≥0.5s inside a clause", flag(m.hesitations <= 1, m.hesitations <= 4, m.hesitations > 1 ? "work on" : "good")],
    ["Fillers", m.fillers + m.hidden_fillers, "", `${m.fillers_per_min}/min`, flag(m.fillers_per_min <= 1, m.fillers_per_min <= 3, m.fillers_per_min > 1 ? "work on" : "good")],
    ["Pitch range", m.pitch_range_st, "st", `median ${m.pitch_median_hz} Hz · aim 8-12`, flag(m.pitch_range_st >= 8, m.pitch_range_st >= 6, m.pitch_range_st < 8 ? "flat" : "lively")],
  ];
  $("tiles").replaceChildren(...tiles.map(([k, v, u, n, f]) =>
    el("div", { class: "tile" }, el("div", { class: "top-row" }, el("span", { class: "k" }, k), f),
      el("div", { class: "v" }, v, u && el("small", {}, " " + u)), el("div", { class: "n" }, n))));

  const player = $<HTMLAudioElement>("player");
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  const blob = await store.loadAudio(r.id);
  audioUrl = blob ? URL.createObjectURL(blob) : null;
  player.src = audioUrl ?? "";
  player.ontimeupdate = () => follow(player.currentTime);
  drawChart();
  renderTranscript();
  renderTips();
  renderStructure();
  renderInsights();
  updateAiCard();
  $("foot").textContent = `Transcribed by ${r.engine === "parakeet" ? "NVIDIA Parakeet v3" : ENGINE} in ${r.timing?.asr_s ?? "?"}s. ` +
    "Pitch and loudness measured with a Praat-style tracker.";
}

function seek(t: number) {
  const p = $<HTMLAudioElement>("player");
  p.currentTime = Math.max(0, t - 0.15);
  p.play();
}

// ---------------------------------------------------------------- timeline chart

const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag: string, attrs: Record<string, string | number>) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  return n;
};
let chartGeom: { x: (t: number) => number; head: Element } | null = null;

function drawChart() {
  const r = report, svg = $("chart") as unknown as SVGSVGElement;
  const W = svg.clientWidth || 800, H = svg.clientHeight || 220, padL = 34, padR = 12, padT = 14, padB = 24;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const T: number = r.duration;
  const pts: [number, number | null][] = r.timeline.pitch;
  // smooth to show the intonation contour, not frame jitter: a 9-point median drops pitch-tracker spikes,
  // then a 5-point mean softens the steps; unvoiced frames stay gaps
  const med = pts.map((p, i) => {
    if (p[1] === null) return null;
    const w = pts.slice(Math.max(0, i - 4), i + 5).map((q) => q[1]).filter((v): v is number => v !== null).sort((a, b) => a - b);
    return w.length >= 3 ? w[Math.floor(w.length / 2)] : null;
  });
  const vals = med.map((v, i) => {
    if (v === null) return null;
    const w = med.slice(Math.max(0, i - 2), i + 3).filter((u): u is number => u !== null);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });
  const finite = vals.filter((v): v is number => v !== null).sort((a, b) => a - b);
  const lo = Math.min(-4, Math.floor(finite[Math.floor(finite.length * 0.01)] ?? -4));
  const hi = Math.max(6, Math.ceil(finite[Math.floor(finite.length * 0.99)] ?? 6));
  const x = (t: number) => padL + (t / T) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - padT - padB);

  for (const p of r.pauses) {
    if (p.dur < 0.3) continue;
    svg.append(svgEl("rect", { class: p.kind === "hesitation" && p.dur >= 0.5 ? "band-bad" : "band-ok", x: x(p.s), y: padT, width: Math.max(1, x(p.e) - x(p.s)), height: H - padT - padB }));
  }
  for (let v = Math.ceil(lo / 4) * 4; v <= hi; v += 4) {
    svg.append(svgEl("line", { class: v === 0 ? "zero" : "grid", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }));
    const lab = svgEl("text", { class: "axis-label", x: padL - 6, y: y(v) + 4, "text-anchor": "end" });
    lab.textContent = v > 0 ? `+${v}` : String(v);
    svg.append(lab);
  }
  // roughly one time label per 80px, on a round step
  const step = [5, 10, 15, 30, 60, 120].find((st) => (T / st) * 80 <= W - padL - padR) || 120;
  for (let t = 0; t <= T; t += step) {
    const anchor = x(t) < padL + 12 ? "start" : x(t) > W - padR - 16 ? "end" : "middle";
    const lab = svgEl("text", { class: "axis-label", x: x(t), y: H - 6, "text-anchor": anchor });
    lab.textContent = fmt(t);
    svg.append(lab);
  }
  // bridge short unvoiced gaps (consonants) so a phrase reads as one contour; break at real pauses
  let d = "", last = -Infinity;
  pts.forEach((p, i) => {
    const v = vals[i];
    if (v === null) return;
    d += `${p[0] - last <= 0.15 ? "L" : "M"}${x(p[0]).toFixed(1)},${y(v).toFixed(1)}`;
    last = p[0];
  });
  svg.append(svgEl("path", { class: "pitch", d }));
  for (const e of r.endings.filter((e: any) => e.shape === "rising" && !e.question)) {
    const cx = x(e.t), cy = padT + 2;
    svg.append(svgEl("path", { class: "rise", d: `M${cx - 6},${cy + 9}L${cx + 6},${cy + 9}L${cx},${cy - 1}Z` }));
  }
  const cross = svgEl("line", { class: "cross", y1: padT, y2: H - padB, x1: -10, x2: -10 });
  const head = svgEl("line", { class: "playhead", y1: padT, y2: H - padB, x1: -10, x2: -10 });
  svg.append(cross, head);
  chartGeom = { x, head };

  const tip = $("tooltip");
  const tAt = (ev: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    return Math.max(0, Math.min(T, ((px - padL) / (W - padL - padR)) * T));
  };
  svg.onmousemove = (ev) => {
    const t = tAt(ev);
    cross.setAttribute("x1", String(x(t))); cross.setAttribute("x2", String(x(t)));
    const w = r.words.find((w: any) => w.s <= t && t < w.e + 0.05);
    const pz = r.pauses.find((p: any) => p.s <= t && t < p.e);
    const i = Math.min(pts.length - 1, Math.round(t / ((pts[1]?.[0] ?? 0.02) - (pts[0]?.[0] ?? 0) || 0.02)));
    const v = vals[i];
    tip.hidden = false;
    tip.style.left = `${(x(t) / W) * 100}%`;
    tip.style.top = `${padT}px`;
    tip.textContent = `${t.toFixed(1)}s · ` + (pz ? `${pz.kind === "hesitation" ? "mid-thought pause" : "pause"} ${pz.dur}s`
      : w ? `“${w.w}”${v !== null && v !== undefined ? ` · ${v > 0 ? "+" : ""}${v.toFixed(1)} st` : ""}` : "silence");
  };
  svg.onmouseleave = () => { tip.hidden = true; cross.setAttribute("x1", "-10"); cross.setAttribute("x2", "-10"); };
  svg.onclick = (ev) => seek(tAt(ev));
}

function follow(t: number) {
  if (chartGeom) { chartGeom.head.setAttribute("x1", String(chartGeom.x(t))); chartGeom.head.setAttribute("x2", String(chartGeom.x(t))); }
  const i = report.words.findIndex((w: any) => w.s <= t && t < w.e);
  document.querySelectorAll(".transcript .w.current").forEach((n) => n.classList.remove("current"));
  if (i >= 0) document.querySelector(`.transcript .w[data-i="${i}"]`)?.classList.add("current");
}

// ---------------------------------------------------------------- transcript, tips, structure, insights

function renderTranscript() {
  const r = report, out: Kid[] = [];
  const after = new Map<number, any>(r.pauses.filter((p: any) => p.dur >= 0.5).map((p: any) => [p.after, p]));
  r.words.forEach((w: any, i: number) => {
    out.push(el("span", { class: ["w", ...w.f].join(" "), "data-i": i, onclick: () => seek(w.s),
      title: w.f.includes("filler") ? "filler" : w.f.includes("stretched") ? "drawn out" : w.f.includes("vague") ? "vague word" : w.f.includes("overused") ? "repeated word" : null }, w.w), " ");
    const p = after.get(i);
    if (p) out.push(el("span", { class: `p ${p.kind === "hesitation" ? "bad" : "ok"}`, onclick: () => seek(p.s),
      title: p.kind === "hesitation" ? "pause mid-thought" : "pause" }, `${p.dur.toFixed(1)}s`), " ");
  });
  $("transcript").replaceChildren(...(out.flat() as (Node | string)[]));
}

const SEV: Record<string, string> = { critical: "Fix first", serious: "Important", warning: "Polish" };
function renderTips() {
  const tips = report.tips;
  $("tips").replaceChildren(...(tips.length ? tips.slice(0, 8).map((t: any) => el("li", { class: "tip" },
    el("div", { class: "tip-title" }, el("span", { class: `sev ${t.severity}` }, SEV[t.severity]), t.title),
    el("p", {}, t.detail),
    t.evidence.length ? el("ul", { class: "evidence" }, t.evidence.map((e: any) =>
      el("li", {}, el("button", { onclick: () => seek(e.t), title: "play from here" }, fmt(e.t)), el("span", {}, e.quote)))) : null,
    t.drill ? el("div", { class: "drill" }, el("b", {}, "Drill: "), t.drill) : null,
  )) : [el("li", {}, "Nothing major. Polish with another take.")]));
}

function renderStructure() {
  const s = report.structure;
  $("structure").replaceChildren(...s.elements.map((e: any) => el("li", {},
    el("span", { class: `mark ${e.found ? "yes" : e.partial ? "part" : "no"}`, "aria-label": e.found ? "present" : e.partial ? "partly" : "missing" }, e.found ? "✓" : e.partial ? "~" : "✗"),
    el("div", {}, e.label, e.found && e.evidence ? el("div", { class: "ev" }, `“${e.evidence}”`) : null))));
  $("strengths").replaceChildren(...(report.strengths.length ? report.strengths.map((x: string) => el("li", {}, x)) : [el("li", {}, "Keep practising; strengths will show up here.")]));
}

function renderInsights() {
  const i = report.insights, m = report.metrics;
  $("insightsCard").hidden = !i;
  if (!i) return;
  const box = (label: string, body: Kid, note: string | null, extra = "") =>
    el("div", { class: `insight ${extra}` }, el("div", { class: "k" }, label), body, note ? el("div", { class: "n" }, note) : null);
  const items: HTMLElement[] = [];
  items.push(box("Specific details",
    i.concrete.length ? el("div", { class: "chips-out" }, i.concrete.slice(0, 10).map((c: any) => el("span", { class: "tag", title: c.kind }, c.text)))
      : el("div", { class: "big" }, "None"),
    i.concrete.length ? "Companies, places, numbers and durations people can remember." : "Add a company, product, place or number.", "wide"));
  if (i.vague.length) {
    items.push(box("Vague words", el("div", { class: "chips-out" }, i.vague.slice(0, 8).map(([w, c]: [string, number]) => el("span", { class: "tag vague" }, w, c > 1 ? el("b", {}, ` ×${c}`) : null))),
      "Each one could be the specific thing you mean.", "wide"));
  }
  const total = i.i_did + i.we_did;
  items.push(box("Who did it", el("div", {}, el("span", { class: "big" }, `I ${i.i_did}`), el("span", { class: "muted" }, `  ·  we ${i.we_did}`),
    total ? el("div", { class: "split", "aria-hidden": "true" }, el("div", { style: `flex:${i.i_did || 0.0001}` }), el("div", { style: `flex:${i.we_did || 0.0001}` })) : null),
    "Actions you claim as yours versus the team's."));
  items.push(box("Action verbs", el("div", { class: "big" }, i.action_share === null ? "-" : `${Math.round(i.action_share * 100)}%`),
    "of sentences built on a verb that shows what you did, not \"I am\" or \"I have\"."));
  items.push(box("Sentence length", el("div", { class: "big" }, `${m.avg_sentence_words ?? "-"} words`),
    `average; longest ${m.longest_sentence_words ?? "-"}. Under 20 is easy to follow.`));
  if (i.numbers.length) {
    items.push(box("Your numbers", el("div", { class: "chips-out" }, i.numbers.map((n: any) => {
      const flat = n.db < 1.5 && (n.st === null || n.st < 1) && n.slow < 1.25;
      return el("span", { class: `tag ${flat ? "flat" : "ok"}`, title: `${n.db > 0 ? "+" : ""}${n.db} dB${n.st !== null ? `, ${n.st > 0 ? "+" : ""}${n.st} st` : ""}` },
        n.word.replace(/[.,]$/, ""), el("b", {}, flat ? " flat" : " stressed"));
    })), "Stress shows as louder, higher or slower than the rest of the sentence."));
  }
  if (i.name) {
    items.push(box("Your name", el("div", { class: "big" }, i.name.rushed ? "Rushed" : "Clear"),
      `${i.name.name.replace(/[.,]$/, "")}${i.name.gap >= 0.2 ? `, with a ${i.name.gap.toFixed(1)}s pause after` : ""}.`));
  }
  $("insights").replaceChildren(...items);
}

// ---------------------------------------------------------------- AI coach

let catalog: ai.Catalog = {};
let aiCfg: store.AiSettings;

async function loadCatalog() {
  aiCfg = await store.aiSettings();
  const cached = await store.get<ai.Catalog>("settings", "catalog");
  catalog = cached ?? (await ai.loadCatalog(async () => new Response("{}")));
  updateAiCard();
  const fresh = await ai.loadCatalog(await appFetch(), cached);
  if (fresh !== cached) { catalog = fresh; await store.put("settings", "catalog", fresh); }
  updateAiCard();
  if (!$("settings").hidden) renderSettings();
}

function currentModel() {
  const p = catalog[aiCfg?.provider];
  const m = p?.models[aiCfg.model];
  const key = aiCfg?.keys[aiCfg.provider];
  return { p, m, key, label: p ? `${m?.name ?? aiCfg.model} (${p.name})` : "AI coach" };
}

function updateAiCard() {
  const r = report;
  if (!r || r.empty) return;
  const { p, key, label } = currentModel();
  const ready = !!(p && key);
  $("aiCard").hidden = false;
  $("aiLabel").textContent = label;
  $<HTMLButtonElement>("aiBtn").disabled = !ready;
  $("aiBtn").textContent = r.ai ? "Ask again" : "Get content feedback and a rewrite";
  if (r.ai) renderAI(r.ai);
  else if (!ready) {
    $("aiOut").replaceChildren(el("p", { class: "muted" }, `Add an API key for ${p?.name ?? "a provider"} in Settings to turn this on. `,
      el("button", { class: "ghost", onclick: () => toggleSettings(true) }, "Open settings")));
  } else $("aiOut").replaceChildren();
}

async function getFeedback() {
  const btn = $<HTMLButtonElement>("aiBtn");
  const { p, key, m } = currentModel();
  if (!p || !key) return toggleSettings(true);
  btn.disabled = true; btn.textContent = `${m?.name ?? aiCfg.model} is reviewing your take...`;
  try {
    const lm = ai.model(p, aiCfg.model, key, await appFetch());
    const fb = await ai.feedback(lm, report, $<HTMLTextAreaElement>("notes").value);
    report.ai = { ...fb, model: fb.model ?? aiCfg.model, provider: p.name };
    await store.put("takes", report.id, report);
    renderAI(report.ai);
    btn.textContent = "Ask again";
    loadTakes();
  } catch (e) {
    $("aiOut").replaceChildren(el("p", { class: "status error" }, (e as Error).message));
    btn.textContent = "Try again";
  } finally {
    btn.disabled = false;
  }
}

function renderAI(f: any) {
  const copy = el("button", { class: "ghost", onclick: () => { navigator.clipboard.writeText(f.rewrite); copy.textContent = "Copied"; } }, "Copy");
  $("aiOut").replaceChildren(
    el("p", { class: "summary" }, f.summary),
    ...(f.strengths?.length ? [el("p", { class: "muted" }, "Working: " + f.strengths.join(" · "))] : []),
    el("h3", {}, "Change what you say"),
    el("ul", { class: "fixes" }, f.fixes.map((x: any) => el("li", {},
      el("div", { class: "said" }, `“${x.said}”`), el("div", {}, x.issue), el("div", { class: "try" }, "Try: " + x.try)))),
    el("div", { class: "card-head" }, el("h3", {}, `Rewrite (~${f.rewrite_seconds}s)`), copy),
    el("div", { class: "rewrite" }, f.rewrite),
    el("p", {}, el("b", {}, "Next take: "), f.next_take_focus),
    el("p", { class: "muted small" }, `Feedback by ${f.model}${f.provider ? ` via ${f.provider}` : ""}`),
  );
}

// ---------------------------------------------------------------- settings

const POPULAR = ["openrouter", "anthropic", "openai", "google", "groq", "mistral", "deepseek", "xai"];

function toggleSettings(open: boolean) {
  if (open) { toggleHistory(false); renderSettings(); }
  $("settings").hidden = !open;
}

function renderSettings() {
  const sel = $<HTMLSelectElement>("providerSel");
  const all = Object.values(catalog).sort((a, b) => a.name.localeCompare(b.name));
  const popular = POPULAR.map((id) => catalog[id]).filter(Boolean);
  const opt = (p: ai.CatalogProvider) => el("option", { value: p.id, selected: p.id === aiCfg.provider || undefined },
    `${p.name}${aiCfg.keys[p.id] ? "  ✓" : ""}`);
  sel.replaceChildren(el("optgroup", { label: "Popular" }, popular.map(opt)), el("optgroup", { label: "All providers" }, all.filter((p) => !POPULAR.includes(p.id)).map(opt)));
  sel.value = aiCfg.provider;
  renderModels();
  renderKeyState();
}

function renderModels() {
  const p = catalog[$<HTMLSelectElement>("providerSel").value];
  const sel = $<HTMLSelectElement>("modelSel");
  if (!p) return sel.replaceChildren();
  const models = Object.values(p.models).sort((a, b) =>
    Number(b.structured_output ?? false) - Number(a.structured_output ?? false) || a.name.localeCompare(b.name));
  const free = (m: ai.CatalogModel) => m.cost && !m.cost.input && !m.cost.output;
  sel.replaceChildren(...models.map((m) => el("option", { value: m.id, selected: m.id === aiCfg.model || undefined },
    `${m.name}${free(m) ? " (free)" : ""}${m.structured_output === false ? " (no structured output)" : ""}`)));
  if (p.models[aiCfg.model]) sel.value = aiCfg.model;
}

function renderKeyState() {
  const pid = $<HTMLSelectElement>("providerSel").value;
  const p = catalog[pid];
  const key = aiCfg.keys[pid];
  const box = $("keyState");
  const env = p?.env?.[0];
  if (key) {
    box.replaceChildren(el("span", {}, "Saved key ", el("code", {}, key.length > 16 ? `${key.slice(0, 8)}…${key.slice(-4)}` : "…" + key.slice(-4))),
      el("button", { class: "ghost", onclick: removeKey }, "Remove"));
  } else {
    box.replaceChildren(el("span", { class: "muted" }, `No key for ${p?.name ?? pid} yet.`, env ? ` (It's the one usually called ${env}.)` : "",
      p?.doc ? el("span", {}, " ", el("a", { href: p.doc, target: "_blank", rel: "noopener" }, "Provider docs")) : null));
  }
  $("keySave").textContent = key ? "Replace key" : "Check and save";
}

async function persistChoice() {
  aiCfg.provider = $<HTMLSelectElement>("providerSel").value;
  aiCfg.model = $<HTMLSelectElement>("modelSel").value;
  await store.saveAiSettings(aiCfg);
  updateAiCard();
}

function keyMsg(text: string, kind = "") {
  $("keyMsg").textContent = text;
  $("keyMsg").className = `status ${kind}`;
}

async function saveKey(e: Event) {
  e.preventDefault();
  const key = $<HTMLInputElement>("keyInput").value.trim();
  const p = catalog[$<HTMLSelectElement>("providerSel").value];
  const modelId = $<HTMLSelectElement>("modelSel").value;
  if (!key) return keyMsg("Paste your key first.", "error");
  if (!p || !modelId) return keyMsg("Pick a provider and model first.", "error");
  $<HTMLButtonElement>("keySave").disabled = true; keyMsg(`Checking the key with ${p.name}...`);
  try {
    await ai.check(ai.model(p, modelId, key, await appFetch()));
    aiCfg.keys[p.id] = key;
    aiCfg.provider = p.id;
    aiCfg.model = modelId;
    await store.saveAiSettings(aiCfg);
    $<HTMLInputElement>("keyInput").value = "";
    renderSettings();
    keyMsg("Key saved and working.", "ok");
    updateAiCard();
  } catch (err) {
    keyMsg((err as Error).message, "error");
  } finally {
    $<HTMLButtonElement>("keySave").disabled = false;
  }
}

async function removeKey() {
  delete aiCfg.keys[$<HTMLSelectElement>("providerSel").value];
  await store.saveAiSettings(aiCfg);
  renderSettings();
  keyMsg("Key removed.");
  updateAiCard();
}

// ---------------------------------------------------------------- history

async function loadTakes() {
  takes = await store.listTakes();
  const list = $("takes");
  list.replaceChildren(...takes.map((t) => el("li", { onclick: () => openTake(t.id) },
    el("span", { class: "s" }, t.overall ?? "-"),
    el("div", {}, el("div", {}, SCENARIOS[t.scenario]?.label || t.scenario, el("span", { class: "muted" }, ` · ${Math.round(t.duration || 0)}s · ${t.wpm || 0} wpm`)),
      el("div", { class: "muted small" }, new Date(t.created).toLocaleString())),
    el("button", { class: "del", "aria-label": "Delete take", onclick: async (e: Event) => {
      e.stopPropagation();
      if (!confirm("Delete this take?")) return;
      await store.deleteTake(t.id);
      if (report?.id === t.id) { report = null; $("results").hidden = true; }
      loadTakes();
    } }, "✕"))));
  if (!takes.length) list.append(el("li", { class: "muted" }, "No takes yet."));
  drawTrend();
}

function drawTrend() {
  const svg = $("trend") as unknown as SVGSVGElement, pts = takes.filter((t) => t.overall != null).slice().reverse();
  svg.replaceChildren();
  if (pts.length < 2) return;
  const W = 320, H = 70, p = 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (i: number) => p + (i / (pts.length - 1)) * (W - 2 * p);
  const y = (v: number) => p + (1 - v / 100) * (H - 2 * p);
  [0, 50, 100].forEach((v) => svg.append(svgEl("line", { class: "grid", x1: p, x2: W - p, y1: y(v), y2: y(v) })));
  svg.append(svgEl("path", { d: pts.map((t, i) => `${i ? "L" : "M"}${x(i)},${y(t.overall)}`).join("") }));
  pts.forEach((t, i) => {
    const c = svgEl("circle", { cx: x(i), cy: y(t.overall), r: 4 });
    const title = document.createElementNS(NS, "title");
    title.textContent = `${t.overall} · ${new Date(t.created).toLocaleDateString()}`;
    c.append(title);
    svg.append(c);
  });
}

async function openTake(id: string) {
  toggleHistory(false);
  const r = await store.loadTake(id);
  if (!r) { status("That take no longer exists.", true); return loadTakes(); }
  status("");
  await showReport(r);
  $("results").scrollIntoView({ behavior: "smooth" });
}

function toggleHistory(open: boolean) {
  if (open) toggleSettings(false);
  $("history").hidden = !open;
}

if (isTauri) document.documentElement.classList.add("tauri");
init().catch((e) => status(`Couldn't start: ${(e as Error).message}`, true));
