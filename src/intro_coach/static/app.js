"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k.nodeType ? k : String(k));
  return n;
};
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const store = {
  get(k, d) { try { const v = localStorage.getItem("ic:" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("ic:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

let config, scenario = store.get("scenario", "interview"), target, report = null, takes = [];

// ---------------------------------------------------------------- setup
async function init() {
  config = await (await fetch("/api/config")).json();

  const box = $("scenarios");
  for (const [id, s] of Object.entries(config.scenarios)) {
    box.append(el("button", { class: "chip", role: "radio", "data-id": id, onclick: () => pickScenario(id) }, s.label));
  }
  pickScenario(config.scenarios[scenario] ? scenario : "interview", true);
  $("target").oninput = (e) => setTarget(+e.target.value);
  $("notes").value = store.get("notes", "");
  $("notes").oninput = (e) => store.set("notes", e.target.value);
  $("recBtn").onclick = toggleRecording;
  $("uploadBtn").onclick = () => $("file").click();
  $("file").onchange = (e) => e.target.files[0] && submitFile(e.target.files[0]);
  setupDrop();
  $("historyBtn").onclick = () => toggleHistory(true);
  $("historyClose").onclick = () => toggleHistory(false);
  $("settingsBtn").onclick = () => toggleSettings(true);
  $("settingsClose").onclick = () => toggleSettings(false);
  $("keyForm").onsubmit = saveKey;
  $("keyShow").onclick = () => {
    const i = $("keyInput"); i.type = i.type === "password" ? "text" : "password";
    $("keyShow").textContent = i.type === "password" ? "Show" : "Hide";
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { toggleHistory(false); toggleSettings(false); } });
  $("aiBtn").onclick = getFeedback;
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/TEXTAREA|INPUT|SELECT|BUTTON|AUDIO/.test(document.activeElement.tagName)) {
      e.preventDefault(); toggleRecording();
    }
  });
  window.addEventListener("resize", () => report && drawChart());
  await loadTakes();
}

function pickScenario(id, initial) {
  scenario = id; store.set("scenario", id);
  document.querySelectorAll("#scenarios .chip").forEach((c) => c.setAttribute("aria-checked", c.dataset.id === id));
  $("prompt").textContent = config.scenarios[id].prompt;
  const shape = config.scenarios[id].shape;
  $("shapeTitle").textContent = shape.title;
  $("shapeSteps").replaceChildren(...shape.steps.map((s) =>
    el("li", {}, el("div", {}, el("b", {}, s.name), el("div", {}, s.say), el("span", { class: "ex" }, `“${s.example}”`)))));
  $("shapeLand").textContent = shape.land;
  const saved = store.get("target:" + id, config.scenarios[id].target);
  setTarget(initial ? saved : config.scenarios[id].target);
}
function setTarget(v) {
  target = v; $("target").value = v; $("targetVal").textContent = `${v}s`;
  $("timerTarget").textContent = `/ ${fmt(v)}`; store.set("target:" + scenario, v);
}

// ---------------------------------------------------------------- recording
let rec = null;
async function toggleRecording() {
  if (rec) return rec.stop();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: false } });
  } catch (e) {
    return status(`Microphone unavailable: ${e.message}`, true);
  }
  const mime = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
  const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const started = performance.now();
  let raf;
  const tick = () => {
    const s = (performance.now() - started) / 1000;
    $("timer").textContent = fmt(s);
    // the bar spans 1.5x the target so going over is visible; the tick marks the target itself
    const pct = Math.min(100, (s / (target * 1.5)) * 100);
    $("progressBar").style.width = pct + "%";
    $("progressBar").parentElement.classList.toggle("over", s > target);
    analyser.getFloatTimeDomainData(buf);
    let sum = 0; for (const x of buf) sum += x * x;
    const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-9);
    $("meterBar").style.width = Math.max(0, Math.min(100, (db + 60) * 2)) + "%";
    raf = requestAnimationFrame(tick);
  };
  mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mr.onstop = () => {
    cancelAnimationFrame(raf); stream.getTracks().forEach((t) => t.stop()); ctx.close();
    rec = null; $("recBtn").classList.remove("on"); $("recBtn").setAttribute("aria-label", "Start recording");
    $("meterBar").style.width = "0";
    const blob = new Blob(chunks, { type: mr.mimeType });
    if ((performance.now() - started) < 1500) return status("That was too short. Try again.", true);
    submit(blob, mime.includes("mp4") ? "take.m4a" : "take.webm");
  };
  mr.start(250);
  rec = mr;
  $("recBtn").classList.add("on"); $("recBtn").setAttribute("aria-label", "Stop recording");
  $("progressBar").parentElement.querySelector(".tick").style.left = `${100 / 1.5}%`;
  status(`Recording. Answer: “${config.scenarios[scenario].prompt}”`);
  tick();
}

function status(msg, error) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", !!error);
}

const MEDIA_EXT = /\.(m4a|mp3|wav|webm|ogg|oga|opus|flac|aac|mp4|mov|mkv|caf|aiff?)$/i;
function submitFile(f) {
  if (rec) return status("Stop the recording first.", true);
  if (!(/^(audio|video)\//.test(f.type) || MEDIA_EXT.test(f.name))) return status(`${f.name} doesn't look like an audio or video file.`, true);
  if (f.size > 500 * 1024 * 1024) return status("That file is over 500 MB. Trim it to your intro first.", true);
  submit(f, f.name, f.name);
}

function setupDrop() {
  const card = document.querySelector(".practice"), drop = $("drop");
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  card.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; drop.hidden = false; });
  card.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  card.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; drop.hidden = true; } });
  card.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; drop.hidden = true;
    const f = e.dataTransfer.files[0];
    if (f) submitFile(f);
  });
  // a file dropped outside the card shouldn't make the browser navigate away from the app
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}

async function submit(blob, name, fileLabel) {
  const label = config.engine;
  status(fileLabel ? `Analysing ${fileLabel} with ${label}...` : `Listening back with ${label}...`);
  $("recBtn").disabled = true; $("uploadBtn").disabled = true;
  const fd = new FormData();
  fd.append("file", blob, name);
  fd.append("scenario", scenario);
  fd.append("target", target);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    status("");
    await loadTakes();
    showReport(body);
    $("results").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    status(`Couldn't analyse that take: ${e.message}`, true);
  } finally {
    $("recBtn").disabled = false; $("uploadBtn").disabled = false; $("file").value = "";
  }
}

// ---------------------------------------------------------------- report
const LABELS = { pace: "Pace", fluency: "Fluency", voice: "Voice", structure: "Structure", language: "Language" };
function verdict(s) { return s >= 80 ? "Strong intro" : s >= 60 ? "Getting there" : s >= 40 ? "Needs work" : "Early draft"; }

function showReport(r) {
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
    ? `${r.overall - prev.overall >= 0 ? "+" : ""}${r.overall - prev.overall} vs your previous ${config.scenarios[r.scenario]?.label.toLowerCase() || ""} take`
    : `${config.scenarios[r.scenario]?.label || r.scenario} · ${new Date(r.created).toLocaleString()}`;
  $("subscores").replaceChildren(...Object.entries(r.scores).map(([k, v]) =>
    el("div", { class: "sub" }, el("span", {}, LABELS[k]), el("div", { class: "bar" }, el("div", { style: `width:${v}%` })), el("span", { class: "val" }, v))));

  const m = r.metrics;
  const flag = (ok, warn, text) => el("span", { class: `flag ${ok ? "good" : warn ? "warn" : "bad"}` }, text);
  const ratio = m.speaking_time / r.target_seconds;
  const tiles = [
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

  const player = $("player");
  player.src = `/api/takes/${r.id}/audio`;
  player.ontimeupdate = () => follow(player.currentTime);
  drawChart();
  renderTranscript();
  renderTips();
  renderStructure();
  renderInsights();

  updateAiCard();
  $("foot").textContent = `Transcribed by ${r.engine === "parakeet" ? "NVIDIA Parakeet v3" : config.engine} in ${r.timing?.asr_s ?? "?"}s. ` +
    "Pitch and loudness by Praat.";
}

function seek(t) {
  const p = $("player");
  p.currentTime = Math.max(0, t - 0.15);
  p.play();
}

// ---------------------------------------------------------------- timeline chart
const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
let chartGeom = null;

function drawChart() {
  const r = report, svg = $("chart");
  const W = svg.clientWidth || 800, H = svg.clientHeight || 220, padL = 34, padR = 12, padT = 14, padB = 24;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const T = r.duration;
  const pts = r.timeline.pitch;
  // smooth to show the intonation contour, not frame jitter: a 9-point median drops pitch-tracker spikes,
  // then a 5-point mean softens the steps; unvoiced frames stay gaps
  const med = pts.map((p, i) => {
    if (p[1] === null) return null;
    const w = pts.slice(Math.max(0, i - 4), i + 5).map((q) => q[1]).filter((v) => v !== null).sort((a, b) => a - b);
    return w.length >= 3 ? w[Math.floor(w.length / 2)] : null;
  });
  const vals = med.map((v, i) => {
    if (v === null) return null;
    const w = med.slice(Math.max(0, i - 2), i + 3).filter((u) => u !== null);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });
  const finite = vals.filter((v) => v !== null).sort((a, b) => a - b);
  const lo = Math.min(-4, Math.floor(finite[Math.floor(finite.length * 0.01)] ?? -4));
  const hi = Math.max(6, Math.ceil(finite[Math.floor(finite.length * 0.99)] ?? 6));
  const x = (t) => padL + (t / T) * (W - padL - padR);
  const y = (v) => padT + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - padT - padB);
  chartGeom = { x, padL, padR, W, T };

  for (const p of r.pauses) {
    if (p.dur < 0.3) continue;
    svg.append(svgEl("rect", { class: p.kind === "hesitation" && p.dur >= 0.5 ? "band-bad" : "band-ok", x: x(p.s), y: padT, width: Math.max(1, x(p.e) - x(p.s)), height: H - padT - padB }));
  }
  for (let v = Math.ceil(lo / 4) * 4; v <= hi; v += 4) {
    svg.append(svgEl("line", { class: v === 0 ? "zero" : "grid", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }));
    const lab = svgEl("text", { class: "axis-label", x: padL - 6, y: y(v) + 4, "text-anchor": "end" });
    lab.textContent = v > 0 ? `+${v}` : v;
    svg.append(lab);
  }
  // roughly one time label per 80px, on a round step
  const step = [5, 10, 15, 30, 60, 120].find((st) => (T / st) * 80 <= W - padL - padR) || 120;
  for (let t = 0; t <= T; t += step) {
    // keep the first and last labels inside the plot instead of clipping at the edges
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
  for (const e of r.endings.filter((e) => e.shape === "rising" && !e.question)) {
    const cx = x(e.t), cy = padT + 2;
    svg.append(svgEl("path", { class: "rise", d: `M${cx - 6},${cy + 9}L${cx + 6},${cy + 9}L${cx},${cy - 1}Z` }));
  }
  const cross = svgEl("line", { class: "cross", y1: padT, y2: H - padB, x1: -10, x2: -10 });
  const head = svgEl("line", { class: "playhead", y1: padT, y2: H - padB, x1: -10, x2: -10 });
  svg.append(cross, head);
  chartGeom.cross = cross; chartGeom.head = head; chartGeom.y = y; chartGeom.vals = vals;

  const tip = $("tooltip");
  const tAt = (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    return Math.max(0, Math.min(T, ((px - padL) / (W - padL - padR)) * T));
  };
  svg.onmousemove = (ev) => {
    const t = tAt(ev);
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t));
    const w = r.words.find((w) => w.s <= t && t < w.e + 0.05);
    const pz = r.pauses.find((p) => p.s <= t && t < p.e);
    const i = Math.min(pts.length - 1, Math.round(t / (pts[1]?.[0] - pts[0]?.[0] || 0.02)));
    const v = vals[i];
    tip.hidden = false;
    tip.style.left = `${(x(t) / W) * 100}%`;
    tip.style.top = `${padT}px`;
    tip.textContent = `${t.toFixed(1)}s · ` + (pz ? `${pz.kind === "hesitation" ? "mid-thought pause" : "pause"} ${pz.dur}s`
      : w ? `“${w.w}”${v !== null && v !== undefined ? ` · ${v > 0 ? "+" : ""}${v.toFixed(1)} st` : ""}` : "silence");
  };
  svg.onmouseleave = () => { tip.hidden = true; cross.setAttribute("x1", -10); cross.setAttribute("x2", -10); };
  svg.onclick = (ev) => seek(tAt(ev));
}

function follow(t) {
  if (chartGeom) { chartGeom.head.setAttribute("x1", chartGeom.x(t)); chartGeom.head.setAttribute("x2", chartGeom.x(t)); }
  const words = report.words;
  const i = words.findIndex((w) => w.s <= t && t < w.e);
  document.querySelectorAll(".transcript .w.current").forEach((n) => n.classList.remove("current"));
  if (i >= 0) document.querySelector(`.transcript .w[data-i="${i}"]`)?.classList.add("current");
}

// ---------------------------------------------------------------- transcript, tips, structure
function renderTranscript() {
  const r = report, out = [];
  const after = new Map(r.pauses.filter((p) => p.dur >= 0.5).map((p) => [p.after, p]));
  r.words.forEach((w, i) => {
    out.push(el("span", { class: ["w", ...w.f].join(" "), "data-i": i, onclick: () => seek(w.s),
      title: w.f.includes("filler") ? "filler" : w.f.includes("stretched") ? "drawn out" : w.f.includes("vague") ? "vague word" : w.f.includes("overused") ? "repeated word" : null }, w.w), " ");
    const p = after.get(i);
    if (p) out.push(el("span", { class: `p ${p.kind === "hesitation" ? "bad" : "ok"}`, onclick: () => seek(p.s),
      title: p.kind === "hesitation" ? "pause mid-thought" : "pause" }, `${p.dur.toFixed(1)}s`), " ");
  });
  $("transcript").replaceChildren(...out);
}

const SEV = { critical: "Fix first", serious: "Important", warning: "Polish" };
function renderTips() {
  const tips = report.tips;
  $("tips").replaceChildren(...(tips.length ? tips.slice(0, 8).map((t) => el("li", { class: "tip" },
    el("div", { class: "tip-title" }, el("span", { class: `sev ${t.severity}` }, SEV[t.severity]), t.title),
    el("p", {}, t.detail),
    t.evidence.length ? el("ul", { class: "evidence" }, t.evidence.map((e) =>
      el("li", {}, el("button", { onclick: () => seek(e.t), title: "play from here" }, `${fmt(e.t)}`), el("span", {}, e.quote)))) : null,
    t.drill ? el("div", { class: "drill" }, el("b", {}, "Drill: "), t.drill) : null,
  )) : [el("li", {}, "Nothing major. Polish with another take.")]));
}

function renderStructure() {
  const s = report.structure;
  $("structure").replaceChildren(...s.elements.map((e) => el("li", {},
    el("span", { class: `mark ${e.found ? "yes" : e.partial ? "part" : "no"}`, "aria-label": e.found ? "present" : e.partial ? "partly" : "missing" }, e.found ? "✓" : e.partial ? "~" : "✗"),
    el("div", {}, e.label, e.found && e.evidence ? el("div", { class: "ev" }, `“${e.evidence}”`) : null))));
  $("strengths").replaceChildren(...(report.strengths.length ? report.strengths.map((x) => el("li", {}, x)) : [el("li", {}, "Keep practising; strengths will show up here.")]));
}

// ---------------------------------------------------------------- insights
function renderInsights() {
  const i = report.insights, m = report.metrics;
  $("insightsCard").hidden = !i;
  if (!i) return;
  const box = (label, body, note, extra = "") => el("div", { class: `insight ${extra}` }, el("div", { class: "k" }, label), body, note ? el("div", { class: "n" }, note) : null);
  const items = [];
  items.push(box("Specific details",
    i.concrete.length ? el("div", { class: "chips-out" }, i.concrete.slice(0, 10).map((c) => el("span", { class: "tag", title: c.kind }, c.text)))
      : el("div", { class: "big" }, "None"),
    i.concrete.length ? "Companies, places, numbers and durations people can remember." : "Add a company, product, place or number.", "wide"));
  if (i.vague.length) {
    items.push(box("Vague words", el("div", { class: "chips-out" }, i.vague.slice(0, 8).map(([w, c]) => el("span", { class: "tag vague" }, w, c > 1 ? el("b", {}, ` ×${c}`) : null))),
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
    items.push(box("Your numbers", el("div", { class: "chips-out" }, i.numbers.map((n) => {
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

// ---------------------------------------------------------------- Claude feedback
async function getFeedback() {
  const btn = $("aiBtn");
  btn.disabled = true; btn.textContent = `${config.ai.label.split(" (")[0]} is reviewing your take...`;
  try {
    const res = await fetch(`/api/takes/${report.id}/feedback`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ notes: $("notes").value }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    report.ai = body;
    renderAI(body);
    btn.textContent = "Ask again";
  } catch (e) {
    $("aiOut").replaceChildren(el("p", { class: "status error" }, e.message));
    btn.textContent = "Try again";
  } finally {
    btn.disabled = false;
  }
}

function renderAI(f) {
  const copy = el("button", { class: "ghost", onclick: () => { navigator.clipboard.writeText(f.rewrite); copy.textContent = "Copied"; } }, "Copy");
  $("aiOut").replaceChildren(
    el("p", { class: "summary" }, f.summary),
    f.strengths.length ? el("p", { class: "muted" }, "Working: " + f.strengths.join(" · ")) : null,
    el("h3", {}, "Change what you say"),
    el("ul", { class: "fixes" }, f.fixes.map((x) => el("li", {},
      el("div", { class: "said" }, `“${x.said}”`), el("div", {}, x.issue), el("div", { class: "try" }, "Try: " + x.try)))),
    el("div", { class: "card-head" }, el("h3", {}, `Rewrite (~${f.rewrite_seconds}s)`), copy),
    el("div", { class: "rewrite" }, f.rewrite),
    el("p", {}, el("b", {}, "Next take: "), f.next_take_focus),
    el("p", { class: "muted small" }, `Feedback by ${f.model}`),
  );
}

// ---------------------------------------------------------------- history
async function loadTakes() {
  takes = await (await fetch("/api/takes")).json();
  const list = $("takes");
  list.replaceChildren(...takes.map((t) => el("li", { onclick: () => openTake(t.id) },
    el("span", { class: "s" }, t.overall ?? "-"),
    el("div", {}, el("div", {}, config.scenarios[t.scenario]?.label || t.scenario, el("span", { class: "muted" }, ` · ${Math.round(t.duration || 0)}s · ${t.wpm || 0} wpm`)),
      el("div", { class: "muted small" }, new Date(t.created).toLocaleString())),
    el("button", { class: "del", "aria-label": "Delete take", onclick: async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this take?")) return;
      await fetch(`/api/takes/${t.id}`, { method: "DELETE" });
      loadTakes();
    } }, "✕"))));
  if (!takes.length) list.append(el("li", { class: "muted" }, "No takes yet."));
  drawTrend();
}

function drawTrend() {
  const svg = $("trend"), pts = takes.filter((t) => t.overall != null).slice().reverse();
  svg.replaceChildren();
  if (pts.length < 2) return;
  const W = 320, H = 70, p = 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (i) => p + (i / (pts.length - 1)) * (W - 2 * p);
  const y = (v) => p + (1 - v / 100) * (H - 2 * p);
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

async function openTake(id) {
  toggleHistory(false);
  const res = await fetch(`/api/takes/${id}`);
  if (!res.ok) {
    status("That take no longer exists.", true);
    return loadTakes();
  }
  status("");
  showReport(await res.json());
  $("results").scrollIntoView({ behavior: "smooth" });
}

function toggleHistory(open) {
  if (open) toggleSettings(false);
  $("history").hidden = !open;
}

// ---------------------------------------------------------------- AI card + settings
function updateAiCard() {
  const r = report;
  if (!r || r.empty) return;
  $("aiCard").hidden = false;
  $("aiLabel").textContent = config.ai.label;
  $("aiBtn").disabled = !config.ai.ready;
  $("aiBtn").textContent = r.ai ? "Ask again" : "Get content feedback and a rewrite";
  if (r.ai) renderAI(r.ai);
  else if (!config.ai.ready) {
    $("aiOut").replaceChildren(el("p", { class: "muted" }, config.ai.hint, " ",
      config.ai.provider === "openrouter" ? el("button", { class: "ghost", onclick: () => toggleSettings(true) }, "Open settings") : null));
  } else $("aiOut").replaceChildren();
}

function toggleSettings(open) {
  if (open) { toggleHistory(false); loadSettings(); }
  $("settings").hidden = !open;
  if (open) setTimeout(() => $("keyInput").focus(), 50);
}

async function loadSettings() {
  const s = await (await fetch("/api/settings")).json();
  renderKeyState(s.openrouter);
}

function renderKeyState(o) {
  $("orModel").textContent = o.model;
  const box = $("keyState");
  if (o.source === "settings") {
    box.replaceChildren(el("span", {}, "Saved key ", el("code", {}, o.masked)),
      el("button", { class: "ghost", onclick: removeKey }, "Remove"));
  } else if (o.source === "environment") {
    box.replaceChildren(el("span", {}, "Using ", el("code", {}, o.masked), " from the OPENROUTER_API_KEY environment variable. A key saved here takes priority."));
  } else {
    box.replaceChildren(el("span", { class: "muted" }, "No key yet. Content feedback is off."));
  }
  $("keySave").textContent = o.source === "settings" ? "Replace key" : "Save key";
}

function keyMsg(text, kind) {
  $("keyMsg").textContent = text;
  $("keyMsg").className = `status ${kind || ""}`;
}

async function refreshAi() {
  config.ai = (await (await fetch("/api/config")).json()).ai;
  updateAiCard();
}

async function saveKey(e) {
  e.preventDefault();
  const key = $("keyInput").value.trim();
  if (!key) return keyMsg("Paste your key first.", "error");
  $("keySave").disabled = true; keyMsg("Checking the key with OpenRouter...");
  try {
    const res = await fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ openrouter_api_key: key }) });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    $("keyInput").value = "";
    renderKeyState(body.openrouter);
    keyMsg("Key saved and working.", "ok");
    await refreshAi();
  } catch (err) {
    keyMsg(err.message, "error");
  } finally {
    $("keySave").disabled = false;
  }
}

async function removeKey() {
  const res = await fetch("/api/settings/openrouter", { method: "DELETE" });
  renderKeyState((await res.json()).openrouter);
  keyMsg("Key removed.");
  await refreshAi();
}

init().catch((e) => status(`Couldn't start: ${e.message}`, true));
