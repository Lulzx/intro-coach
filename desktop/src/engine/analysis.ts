// Timed words + acoustics -> intro coaching: metrics, scores, structure checks, insights and tips.
// Port of src/intro_coach/analysis.py; the report has the same shape and field names.
import * as nlp from "./nlp";
import type { Word } from "./fluid";
import { SCENARIOS } from "./scenarios";
import {
  STEP, type Prosody, loudFrames, mean, median, percentile, range, runs, semitones, std, timeOf, voiced,
} from "./prosody";

const FILLERS = new Set(["um", "uh", "umm", "uhm", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm"]);
const HEDGES = ["kind of", "sort of", "i think", "i guess", "i feel like", "maybe", "probably", "basically",
  "actually", "literally", "pretty much", "a little bit", "you know", "i mean", "stuff like that",
  "and stuff", "or something", "things like that"];
const STOP = new Set(`a an the and or but so of to in on at for from with by as is are was were be been being have has had do does
did i me my mine we our you your he she it its they them their this that these those there here which who whom what
when where why how not no yes very really just also too than then into over about up out more most some any all can
could would should will shall may might must am im i'm i've ive it's its that's there's let's like one get got make
made things thing lot lots well even quite`.split(/\s+/));

const r1 = (x: number, n = 1) => Math.round(x * 10 ** n) / 10 ** n;
const clamp = (x: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, x));
const ramp = (x: number, bad: number, good: number) => (bad === good ? 100 : clamp(((x - bad) / (good - bad)) * 100));
const norm = (w: string) => w.toLowerCase().replace(/[^\w'-]/g, "");
const letters = (w: string) => w.replace(/\W/g, "").length;
const spokenLen = (w: string) => Math.max(1, w.replace(/[^A-Za-z]/g, "").length
  + 4 * (w.match(/\d/g)?.length ?? 0) + 7 * (w.split("%").length - 1) + 6 * (w.split("$").length - 1));
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const signed = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(1)}`;

export interface Evidence { t: number; quote: string }
export interface Tip { area: string; severity: "critical" | "serious" | "warning"; title: string; detail: string; evidence: Evidence[]; drill: string | null; impact: number }

function refineWords(words: Word[], P: Prosody): Word[] {
  // TDT durations can stretch a word across the silence after it; clip each word to its sounding part
  return words.map((w) => {
    let { s, e } = w;
    const [i0, i1] = range(P, s, e);
    let first = -1, last = -1;
    for (let i = i0; i < i1; i++) if (!P.silent[i]) { if (first < 0) first = i; last = i; }
    if (first >= 0) {
      const ns = timeOf(P, first), ne = timeOf(P, last) + STEP;
      if (ns - s > 0.15) s = ns;
      if (e - ne > 0.15) e = ne;
    }
    return { ...w, s: r1(s, 3), e: r1(Math.max(e, s + 0.05), 3) };
  });
}

function sentences(words: Word[]): number[][] {
  const out: number[][] = [];
  let cur: number[] = [];
  words.forEach((w, i) => {
    cur.push(i);
    if (/[.!?]$/.test(w.w)) { out.push(cur); cur = []; }
  });
  if (cur.length) out.push(cur);
  return out;
}

export function analyse(rawWords: Word[], P: Prosody, duration: number, scenario = "interview", targetIn?: number) {
  const sc = SCENARIOS[scenario] ?? SCENARIOS.interview;
  const target = targetIn || sc.target;
  if (!rawWords.length) {
    return { empty: true, duration, scenario, target_seconds: target,
      tips: [{ area: "Recording", severity: "critical", title: "No speech detected", detail: "Check the microphone and try again.", evidence: [], drill: null, impact: 0 }] };
  }
  const words = refineWords(rawWords, P);
  const toks = words.map((w) => norm(w.w));
  const textOf = (idx: number[]) => idx.map((i) => words[i].w).join(" ");
  const quote = (i: number, before = 3, after = 3) => words.slice(Math.max(0, i - before), Math.min(words.length, i + after + 1)).map((w) => w.w).join(" ");
  const t0 = words[0].s, t1 = words[words.length - 1].e;
  const span = Math.max(t1 - t0, 1e-6);
  const minutes = span / 60;
  const sents = sentences(words);

  // --- fillers ---------------------------------------------------------------------------------------------
  const fillerSet = new Set<number>();
  toks.forEach((t, i) => {
    if (FILLERS.has(t)) fillerSet.add(i);
    // "like," used as a filler, not "I like maps"
    if (t === "like" && (words[i].w.endsWith(",") || (i > 0 && words[i - 1].w.endsWith(",")))) fillerSet.add(i);
  });
  const fillerIdx = [...fillerSet].sort((a, b) => a - b);
  const contentN = words.length - fillerIdx.length;

  // sounds the recogniser left out: voiced, un-transcribed gaps between words (often a drawn-out "uhh")
  const hidden: { t: number; dur: number; after: string }[] = [];
  for (let i = 0; i < words.length - 1; i++) {
    const a = words[i].e, b = words[i + 1].s;
    if (b - a < 0.3) continue;
    const [i0, i1] = range(P, a, b);
    let best = 0, run = 0;
    for (let k = i0; k < i1; k++) { run = P.f0[k] > 0 && !P.silent[k] ? run + 1 : 0; best = Math.max(best, run); }
    if (best * STEP >= 0.25) hidden.push({ t: r1(a, 2), dur: r1(best * STEP, 2), after: words[i].w });
  }

  // stretched words: held far longer than their length needs while still voiced (a drawl, thinking aloud)
  const stretched: { i: number; t: number; dur: number; word: string }[] = [];
  words.forEach((w, i) => {
    const d = w.e - w.s;
    if (!fillerSet.has(i) && d > Math.max(0.75, 0.09 * letters(w.w) + 0.45) && voiced(P, w.s, w.e).length * STEP > 0.5 * d) {
      stretched.push({ i, t: r1(w.s, 2), dur: r1(d, 2), word: w.w });
    }
  });

  // --- pauses ----------------------------------------------------------------------------------------------
  const [s0, s1] = range(P, t0, t1);
  const pauses = runs(P, (i) => !!P.silent[i], 0.25, s0, s1).map(([a, b]) => {
    let prev = 0;
    words.forEach((w, i) => { if (w.s < (a + b) / 2) prev = i; });
    const w = words[prev].w;
    let kind = /[.!?]$/.test(w) ? "sentence" : /[,;:—-]$/.test(w) ? "phrase" : "hesitation";
    if (fillerSet.has(prev)) kind = "hesitation";
    return { s: r1(a, 2), e: r1(b, 2), dur: r1(b - a, 2), kind, after: prev };
  });
  const pauseTotal = pauses.reduce((a, p) => a + p.dur, 0);
  const hesitations = pauses.filter((p) => p.kind === "hesitation" && p.dur >= 0.5);
  const longPauses = pauses.filter((p) => p.dur >= 2);

  // --- pace ------------------------------------------------------------------------------------------------
  const wpm = contentN / minutes;
  const articulation = contentN / Math.max((span - pauseTotal) / 60, 1e-6);
  const sentStats = sents.map((idx) => {
    const a = words[idx[0]].s, b = words[idx[idx.length - 1]].e;
    const n = idx.filter((i) => !fillerSet.has(i)).length;
    return { s: a, e: b, n, wpm: n / Math.max((b - a) / 60, 1e-6), idx };
  });
  const paceWindows: { t: number; wpm: number }[] = [];
  for (let a = t0; a < t1; a += 10) {
    const n = words.filter((w, i) => a <= w.s && w.s < a + 10 && !fillerSet.has(i)).length;
    paceWindows.push({ t: r1(a), wpm: Math.round(n * 6) });
  }

  // --- pitch -----------------------------------------------------------------------------------------------
  const vHz = voiced(P, t0, t1);
  const st = vHz.length ? vHz.map((f) => semitones(P, f)) : [0];
  const pitchRange = percentile(st, 95) - percentile(st, 5);
  const pitchSd = std(st);
  const monotone: [number, number][] = [];
  for (let a = t0; a < Math.max(t0 + 0.01, t1 - 8); a += 2) {
    const v = voiced(P, a, a + 8);
    if (v.length >= 150 && std(v.map((f) => semitones(P, f))) < 1.2) {
      if (monotone.length && a <= monotone[monotone.length - 1][1]) monotone[monotone.length - 1][1] = a + 8;
      else monotone.push([a, a + 8]);
    }
  }
  const monotoneSecs = monotone.reduce((s, [a, b]) => s + b - a, 0);
  const endings: { t: number; delta: number; shape: string; question: boolean; text: string; final: boolean }[] = [];
  sents.forEach((idx, k) => {
    const last = words[idx[idx.length - 1]];
    const tail = voiced(P, Math.max(last.s, last.e - 0.35), last.e);
    const pre = voiced(P, last.e - 1.15, last.e - 0.35);
    if (tail.length >= 5 && pre.length >= 10) {
      const delta = 12 * Math.log2(median(tail) / median(pre));
      endings.push({ t: r1(last.e, 2), delta: r1(delta), shape: delta > 1.5 ? "rising" : delta < -1 ? "falling" : "flat",
        question: last.w.endsWith("?"), text: textOf(idx.slice(-6)), final: k === sents.length - 1 });
    }
  });
  const uptalk = endings.filter((e) => e.shape === "rising" && !e.question);

  // --- volume ----------------------------------------------------------------------------------------------
  const loud = loudFrames(P, t0, t1);
  const volSd = loud.length ? std(loud) : 0;
  const trailing: { t: number; drop: number; text: string }[] = [];
  for (const s of sentStats) {
    if (s.idx.length < 4) continue;
    const sentDb = loudFrames(P, s.s, s.e);
    const lw = words[s.idx[s.idx.length - 1]];
    const lastDb = loudFrames(P, lw.s, lw.e);
    if (sentDb.length && lastDb.length) {
      const drop = mean(lastDb) - mean(sentDb);
      if (drop < -6) trailing.push({ t: r1(lw.s, 2), drop: r1(drop), text: textOf(s.idx.slice(-5)) });
    }
  }
  const thirds = [0, 1, 2].map((k) => loudFrames(P, t0 + (k * span) / 3, t0 + ((k + 1) * span) / 3));
  const energyDrift = thirds.every((x) => x.length) ? mean(thirds[2]) - mean(thirds[0]) : 0;

  // --- language --------------------------------------------------------------------------------------------
  const lowerText = " " + toks.join(" ") + " ";
  const hedges = new Map<string, number>();
  for (const h of HEDGES) {
    const c = lowerText.split(` ${h} `).length - 1;
    if (c) hedges.set(h, c);
  }
  const hedgeTotal = [...hedges.values()].reduce((a, b) => a + b, 0);
  // sentences without fillers parse better; word indices are kept so findings point back at the transcript
  const clean = sents.map((idx) => idx.filter((i) => !fillerSet.has(i)).map((i) => [i, words[i].w] as [number, string]));
  const cleanSents = clean.map((s) => s.map(([, w]) => w).join(" ") || ".");
  const ins = nlp.insights(clean);
  const lemmaOf = new Map([...ins.lemmas].filter(([, l]) => !STOP.has(l)));
  const counts = new Map<string, number>();
  for (const l of lemmaOf.values()) counts.set(l, (counts.get(l) ?? 0) + 1);
  const overused = [...counts].sort((a, b) => b[1] - a[1])
    .filter(([w, c]) => c >= 3 && c / Math.max(contentN, 1) >= 0.015 && !nlp.VAGUE.has(w)).slice(0, 6);

  const perfect = Math.max(ins.haveBeen.length, (lowerText.match(/ (i have been|i've been|ive been) /g) ?? []).length);
  const vague = new Map<string, number>();
  for (const v of ins.vague) vague.set(v.text, (vague.get(v.text) ?? 0) + 1);
  const vagueSorted = [...vague].sort((a, b) => b[1] - a[1]);
  const vagueN = ins.vague.length;
  const concrete = ins.concrete;
  const actionShare = ins.mainVerbs ? ins.actionVerbs / ins.mainVerbs : null;
  const sentOf = new Map<number, number>();
  sents.forEach((idx, k) => idx.forEach((i) => sentOf.set(i, k)));

  // are the numbers stressed? compare each number's pitch, loudness and pace with the rest of its sentence
  const stressed: { i: number; t: number; word: string; st: number | null; db: number; slow: number }[] = [];
  for (const i of ins.numbers) {
    const w = words[i], idx = sents[sentOf.get(i) ?? 0];
    const a = words[idx[0]].s, b = words[idx[idx.length - 1]].e;
    const nv = voiced(P, w.s, w.e), sv = voiced(P, a, b);
    const ndb = loudFrames(P, w.s, w.e), sdb = loudFrames(P, a, b);
    if (!ndb.length || !sdb.length) continue;
    let stv: number | null = nv.length >= 3 && sv.length >= 10 ? 12 * Math.log2(median(nv) / median(sv)) : null;
    if (stv !== null && Math.abs(stv) > 8) stv = null; // an octave jump from the pitch tracker, not intonation
    const typical = Math.max(median(idx.map((j) => (words[j].e - words[j].s) / spokenLen(words[j].w))), 1e-3);
    stressed.push({ i, t: r1(w.s, 2), word: w.w, st: stv === null ? null : r1(stv), db: r1(mean(ndb) - mean(sdb)),
      slow: r1((w.e - w.s) / spokenLen(w.w) / typical, 2) });
  }
  // stress shows as louder, higher or slower than the rest of the sentence; any one of them is enough
  const flatNumbers = stressed.filter((x) => x.db < 1.5 && (x.st === null || x.st < 1) && x.slow < 1.25);

  // was the name rushed? time per letter against the speaker's own typical word, and the gap after it
  let namePace: { t: number; name: string; ratio: number; gap: number; rushed: boolean } | null = null;
  if (ins.nameWords.length) {
    const nw = ins.nameWords.map((i) => words[i]);
    const perLetter = (nw[nw.length - 1].e - nw[0].s) / Math.max(nw.reduce((a, w) => a + letters(w.w), 0), 1);
    const long = words.filter((w) => letters(w.w) >= 4).map((w) => (w.e - w.s) / letters(w.w));
    const typical = long.length ? median(long) : perLetter;
    const nxt = ins.nameWords[ins.nameWords.length - 1] + 1;
    const gap = nxt < words.length ? words[nxt].s - nw[nw.length - 1].e : 1;
    namePace = { t: r1(nw[0].s, 2), name: nw.map((w) => w.w).join(" "), ratio: r1(perLetter / typical, 2), gap: r1(gap, 2),
      rushed: perLetter < 0.8 * typical && gap < 0.2 };
  }
  const andStarts = sents.filter((idx) => ["and", "so", "but"].includes(toks[idx[0]])).length;
  const longSents = sentStats.filter((s) => s.n > 35);

  // --- intro structure -------------------------------------------------------------------------------------
  const text = words.map((w) => w.w).join(" ");
  const sentTexts = sents.map(textOf);
  const clip = (s: string, n = 140) => (s.length <= n ? s : s.slice(0, n).replace(/\s+\S*$/, "") + "...");
  const evidence = (k: number | null) => (k === null ? null : ([clip(sentTexts[k]), words[sents[k][0]].s] as [string, number]));
  const hits = nlp.detect(cleanSents, sc.checks.map((c) => c.key));
  const elements = sc.checks.map((c) => {
    const h = hits[c.key];
    return { ...c, found: h.found !== null, partial: h.partial !== null, evidence: evidence(h.found ?? h.partial) };
  });
  const weakOpen = ["so", "um", "uh", "okay", "ok", "well", "like", "yeah", "alright"].includes(toks[0]);
  const lastSentence = sentTexts[sentTexts.length - 1] ?? "";
  const weakEnd = /(\byeah\W*$|that's (it|about it|all)|that is (it|all)|\bi guess\W*$|\banyway\W*$|or something\W*$)/i.test(lastSentence);
  const finalEnding = endings.find((e) => e.final);
  const lengthRatio = span / target;

  // --- scores ----------------------------------------------------------------------------------------------
  let paceScore = Math.min(ramp(wpm, 90, 130), ramp(wpm, 210, 165));
  if (articulation > 200) paceScore -= Math.min(25, (articulation - 200) * 0.8);
  const silencePct = (pauseTotal / span) * 100;
  const fluency = 100 - Math.max(0, silencePct - 20) * 2 - (4 * hesitations.length) / minutes
    - (5 * (fillerIdx.length + hidden.length)) / minutes - (3 * stretched.length) / minutes - 5 * longPauses.length;
  const voice = 0.5 * ramp(pitchRange, 2.5, 8) + 0.5 * ramp(pitchSd, 0.8, 2.2)
    - Math.min(24, 8 * uptalk.length) - (20 * monotoneSecs) / span - Math.min(15, 5 * trailing.length);
  let structure = elements.reduce((a, e) => a + e.weight * (e.found ? 1 : e.partial ? 0.5 : 0), 0);
  structure += 10 * (lengthRatio >= 0.8 && lengthRatio <= 1.2 ? 1 : lengthRatio >= 0.6 && lengthRatio <= 1.5 ? 0.5 : 0);
  structure -= 5 * Number(weakOpen) + 5 * Number(weakEnd);
  const language = 100 - (4 * hedgeTotal) / minutes - 5 * overused.length - 5 * longSents.length
    - 3 * Math.max(0, perfect - 2) - 2 * Math.max(0, andStarts - 2) - Math.min(15, 3 * Math.max(0, vagueN - 2))
    - (!concrete.length && contentN > 30 ? 10 : 0) - (actionShare !== null && actionShare < 0.4 ? 8 : 0);
  const scores = { pace: clamp(paceScore), fluency: clamp(fluency), voice: clamp(voice), structure: clamp(structure), language: clamp(language) };
  const weights = { pace: 0.2, fluency: 0.25, voice: 0.2, structure: 0.25, language: 0.1 };
  const overall = Math.round(Object.entries(weights).reduce((a, [k, w]) => a + scores[k as keyof typeof scores] * w, 0));

  // --- tips ------------------------------------------------------------------------------------------------
  const tips: Tip[] = [];
  const strengths: string[] = [];
  const ev = (t: number, q: string): Evidence => ({ t: r1(t, 2), quote: q });
  const tip = (area: string, severity: Tip["severity"], title: string, detail: string, evidence: Evidence[] = [], drill: string | null = null, impact = 0) =>
    tips.push({ area, severity, title, detail, evidence: evidence.slice(0, 4), drill, impact });
  const shape = sc.shape.title;

  if (lengthRatio > 1.25) {
    tip("Structure", "serious", `Too long: ${span.toFixed(0)}s for a ${target.toFixed(0)}s slot`,
      `Cut about ${(span - target).toFixed(0)}s. Keep one sentence per step of ${shape} and drop the rest.`, [],
      `Write one sentence for each step (${shape}), then say it with a ${target.toFixed(0)}s timer.`, 30 * (lengthRatio - 1));
  } else if (lengthRatio < 0.7) {
    tip("Structure", "warning", `Short: ${span.toFixed(0)}s for a ${target.toFixed(0)}s slot`, `There's room to fill out a step of ${shape}.`, [], null, 10);
  } else strengths.push(`Length fits the slot (${span.toFixed(0)}s of ${target.toFixed(0)}s).`);

  if (hesitations.length) {
    const worst = [...hesitations].sort((a, b) => b.dur - a.dur);
    tip("Fluency", hesitations.length / minutes > 3 ? "critical" : "serious",
      `${plural(hesitations.length, "pause")} in the middle of a thought`,
      "Pausing mid-clause sounds like searching for words. Pause after a complete idea instead, at a full stop.",
      worst.map((p) => ev(p.s, `${quote(p.after, 3, 0)} [${p.dur.toFixed(1)}s] ${p.after + 1 < words.length ? quote(p.after + 1, 0, 3) : ""}`)),
      "Read your intro aloud with the full stops marked; pause only there. Repeat the words you stalled on 5 times first.",
      (8 * hesitations.length) / minutes + 10);
  }
  if (silencePct > 25) {
    tip("Pace", "serious", `${silencePct.toFixed(0)}% of your time is silence`,
      "Engaging speakers sit around 15-20%. Most of the gain comes from removing the mid-thought gaps.", [], null, silencePct - 20);
  }
  if (wpm < 115) {
    tip("Pace", wpm < 100 ? "serious" : "warning", `Slow overall pace: ${wpm.toFixed(0)} wpm`,
      articulation >= 130 ? `Aim for 130-165 wpm. Your speed while actually talking is ${articulation.toFixed(0)} wpm, so the pauses are what slow you down` : "Aim for 130-165 wpm.",
      [], null, (130 - wpm) / 2);
  } else if (wpm > 180 || articulation > 210) {
    const fast = [...sentStats].sort((a, b) => b.wpm - a.wpm).slice(0, 2);
    tip("Pace", "serious", `Rushing: ${wpm.toFixed(0)} wpm (${articulation.toFixed(0)} while talking)`,
      "Slow down on the parts that matter: your name, your role and your result.",
      fast.map((s) => ev(s.s, textOf(s.idx.slice(0, 10)) + "...")), "Breathe at every full stop. Say your name, then a beat of silence.", (wpm - 165) / 2);
  } else strengths.push(`Comfortable pace (${wpm.toFixed(0)} wpm).`);

  const fillersTotal = fillerIdx.length + hidden.length;
  if (fillersTotal / minutes > 2) {
    tip("Fluency", fillersTotal / minutes > 5 ? "serious" : "warning", `${plural(fillersTotal, "filler sound")} (${(fillersTotal / minutes).toFixed(1)}/min)`,
      "Swap 'um' and 'uh' for a silent breath. Closing your mouth while you think stops the sound.",
      [...fillerIdx.slice(0, 3).map((i) => ev(words[i].s, quote(i))), ...hidden.slice(0, 2).map((h) => ev(h.t, `${h.after} [voiced ${h.dur}s]`))],
      "Record one take where every 'um' gets replaced by a full one-second silence.", (4 * fillersTotal) / minutes);
  } else if (fillersTotal <= 1) strengths.push("Almost no filler sounds.");
  if (stretched.length >= 3) {
    tip("Fluency", "warning", `${stretched.length} drawn-out words`,
      "Holding a word while you think ('lea-ding...', 'so-o') is a hidden filler. Finish the word, then pause.",
      stretched.slice(0, 4).map((s) => ev(s.t, `${s.word} (${s.dur}s)`)), null, 2 * stretched.length);
  }

  if (pitchRange < 6 || pitchSd < 1.6) {
    tip("Voice", pitchRange < 4.5 ? "serious" : "warning", `Flat pitch: ${pitchRange.toFixed(1)} semitone range`,
      "Engaging speakers use 8-12 semitones. Lift the one word per sentence that carries the meaning.",
      monotone.slice(0, 3).map(([a, b]) => ev(a, `monotone ${(a - t0).toFixed(0)}s-${(b - t0).toFixed(0)}s`)),
      "Underline one word per sentence in your script and say it higher and slightly louder. Exaggerate on take one.", (8 - pitchRange) * 4);
  } else if (pitchRange >= 8) strengths.push(`Lively pitch (${pitchRange.toFixed(0)} semitone range).`);
  if (uptalk.length) {
    const finalRises = uptalk.some((e) => e.final);
    tip("Voice", finalRises ? "serious" : "warning", `${plural(uptalk.length, "statement")} ending on a rise`,
      "A rising end makes a statement sound like a question and undercuts confidence" + (finalRises ? ", and your final line rises." : "."),
      uptalk.map((e) => ev(e.t, `...${e.text} (+${e.delta} st)`)), "Say your last sentence three times, landing each one lower than the last.", 6 * uptalk.length);
  } else if (finalEnding?.shape === "falling") strengths.push("Your final line lands with a confident downward pitch.");
  if (trailing.length) {
    tip("Voice", "warning", `Volume drops at the end of ${plural(trailing.length, "sentence")}`,
      "The last word of a sentence is often the important one; keep breath for it.", trailing.map((x) => ev(x.t, `...${x.text} (${x.drop} dB)`)), null, 3 * trailing.length);
  }
  if (energyDrift < -4) tip("Voice", "warning", `Energy fades: ${Math.abs(energyDrift).toFixed(0)} dB quieter by the end`, "Your close should be at least as strong as your opening.", [], null, 5);

  for (const e of elements.filter((x) => !x.found)) {
    tip("Structure", e.weight >= 20 ? "serious" : "warning", e.fix_title, e.fix,
      e.evidence ? [ev(e.evidence[1], "closest: " + e.evidence[0])] : [], null, e.weight * (e.partial ? 0.5 : 1));
  }
  if (weakOpen) tip("Structure", "warning", `Opening with "${words[0].w.replace(/[,.]+$/, "")}"`, "Start straight with a greeting and your name.", [ev(t0, quote(0, 0, 5))], null, 5);
  if (weakEnd) tip("Structure", "serious", "Weak final line", "Ending on 'so yeah' or 'that's it' drains the intro. " + sc.shape.land, [ev(words[sents[sents.length - 1][0]].s, lastSentence)], null, 10);

  if (overused.length) tip("Language", "warning", "Repeated words: " + overused.slice(0, 4).map(([w, c]) => `"${w}" x${c}`).join(", "),
    "Repetition makes a short intro sound vague. Replace with the specific thing you mean.", [], null, 3 * overused.length);
  if (perfect >= 3) tip("Language", "warning", `"I have been..." used ${perfect} times`, 'Switch to active verbs: "I led", "I built", "I shipped".', [], null, 2 * perfect);
  if (hedgeTotal / minutes > 2) tip("Language", "warning", "Hedging words: " + [...hedges].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([h, c]) => `"${h}" x${c}`).join(", "),
    "Hedges soften every claim. Cut them and state it.", [], null, 2 * hedgeTotal);
  if (longSents.length) tip("Language", "warning", `${longSents.length} very long sentence${longSents.length === 1 ? "" : "s"}`,
    "Over 35 words in one breath is hard to follow. Split at the 'and's.", longSents.slice(0, 2).map((s) => ev(s.s, textOf(s.idx.slice(0, 12)) + "...")), null, 4 * longSents.length);
  if (elements.some((e) => e.key === "name" && e.found) && !weakOpen) strengths.push("Clear opening with your name.");

  if (vagueN >= 3 && vagueN / Math.max(contentN, 1) > 0.025) {
    tip("Language", "warning", "Vague words: " + vagueSorted.slice(0, 4).map(([w, c]) => `"${w}" x${c}`).join(", "),
      "Swap each one for the specific thing: a company, a product, a number, a type of customer.",
      ins.vague.slice(0, 3).map((v) => ev(words[v.i].s, quote(v.i, 4, 3))), null, 2 * vagueN);
  }
  if (!concrete.length && contentN > 30) {
    tip("Language", "serious", "No concrete details", "Nothing a listener can hold on to: no company, place, product or number. Add at least one.", [], null, 12);
  } else if (concrete.length >= 3) strengths.push("Specific details: " + concrete.slice(0, 4).map((c) => c.text).join(", ") + ".");
  if (scenario === "interview" && ins.weDid >= 2 && ins.weDid > ins.iDid) {
    tip("Language", "warning", `More "we" (${ins.weDid}) than "I" (${ins.iDid})`,
      'Interviewers are hiring you. Keep the team result, but say what you did: "I designed...", "I led...".', [], null, 6);
  }
  if (actionShare !== null && ins.mainVerbs >= 4 && actionShare < 0.4) {
    tip("Language", "warning", `Few action verbs (${Math.round(actionShare * 100)}% of sentences)`,
      'Most sentences rest on "I am", "I have" or "I like". Verbs like led, built, cut and launched show what you did.', [], null, 6);
  }
  if (stressed.length && flatNumbers.length >= Math.max(1, Math.floor((stressed.length + 1) / 2))) {
    tip("Voice", "warning", "Your numbers don't stand out",
      "Key figures said at the same pitch and volume as the rest of the sentence get lost. Pause just before the number and lift it.",
      flatNumbers.slice(0, 3).map((x) => ev(x.t, `${quote(x.i, 3, 2)} (${signed(x.db)} dB vs the sentence)`)),
      "Say the sentence three times, each time hitting the number a little harder.", 7);
  } else if (stressed.length) strengths.push("You stressed your key numbers.");
  if (namePace?.rushed) {
    tip("Voice", "warning", "Your name went by quickly", "People can't remember a name they didn't catch. Slow down on it and leave a beat after.",
      [ev(namePace.t, `${namePace.name} (gap after: ${namePace.gap.toFixed(1)}s)`)], 'Say just "Hi, I\'m [name].", pause for one second, then carry on.', 6);
  }
  const order = { critical: 0, serious: 1, warning: 2 };
  tips.sort((a, b) => order[a.severity] - order[b.severity] || b.impact - a.impact);

  // --- timeline for the UI ---------------------------------------------------------------------------------
  const pitchLine: [number, number | null][] = [];
  for (let k = 0; k < P.n; k += 2) pitchLine.push([r1(timeOf(P, k), 2), P.f0[k] > 0 ? r1(semitones(P, P.f0[k]), 2) : null]);
  const overusedSet = new Set(overused.map(([w]) => w));
  const vagueIdx = vagueN >= 3 ? new Set(ins.vague.map((v) => v.i)) : new Set<number>();
  const stretchedIdx = new Set(stretched.map((s) => s.i));
  const wordOut = words.map((w, i) => ({
    w: w.w, s: w.s, e: w.e,
    f: [fillerSet.has(i) && "filler", overusedSet.has(lemmaOf.get(i) ?? "") && "overused", stretchedIdx.has(i) && "stretched", vagueIdx.has(i) && "vague"].filter(Boolean) as string[],
  }));
  const avgSentence = sentStats.length ? mean(sentStats.map((x) => x.n)) : 0;

  return {
    scenario, prompt: sc.prompt, target_seconds: target,
    duration: r1(duration), speech_start: r1(t0, 2), speech_end: r1(t1, 2),
    transcript: text, words: wordOut, pauses,
    overall, scores: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, Math.round(v)])),
    metrics: {
      speaking_time: r1(span), words: contentN, wpm: Math.round(wpm), articulation_wpm: Math.round(articulation),
      silence_pct: Math.round(silencePct), pauses: pauses.length, hesitations: hesitations.length,
      long_pauses: longPauses.length, fillers: fillerIdx.length, hidden_fillers: hidden.length,
      stretched_words: stretched.length, fillers_per_min: r1(fillersTotal / minutes),
      pitch_median_hz: Math.round(P.f0Median), pitch_range_st: r1(pitchRange), pitch_sd_st: r1(pitchSd, 2),
      monotone_secs: Math.round(monotoneSecs), uptalk: uptalk.length, volume_sd_db: r1(volSd),
      energy_drift_db: r1(energyDrift), trailing_off: trailing.length,
      hedges: Object.fromEntries(hedges), overused, i_have_been: perfect, long_sentences: longSents.length,
      avg_sentence_words: r1(avgSentence), longest_sentence_words: Math.max(0, ...sentStats.map((x) => x.n)),
    },
    insights: {
      concrete: concrete.map((c) => ({ text: c.text, kind: c.kind, t: r1(words[c.i].s, 2) })),
      vague: vagueSorted, i_did: ins.iDid, we_did: ins.weDid,
      action_share: actionShare === null ? null : r1(actionShare, 2),
      numbers: stressed, flat_numbers: flatNumbers.length, name: namePace,
    },
    structure: {
      elements: elements.map((e) => ({ key: e.key, label: e.label, found: e.found, partial: e.partial,
        evidence: e.evidence?.[0] ?? null, t: e.evidence ? r1(e.evidence[1], 2) : null })),
      weak_open: weakOpen, weak_end: weakEnd, length_ratio: r1(lengthRatio, 2),
    },
    endings, monotone, hidden_fillers: hidden, pace_windows: paceWindows, tips, strengths,
    timeline: { pitch: pitchLine, median_hz: Math.round(P.f0Median) },
  };
}

export type Report = Extract<ReturnType<typeof analyse>, { overall: number }> & { id?: string; created?: string; engine?: string; timing?: Record<string, number>; ai?: unknown };
