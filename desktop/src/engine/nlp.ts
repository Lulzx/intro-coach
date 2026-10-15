// Intro structure checks and language insights with compromise.js, porting src/intro_coach/nlp.py (spaCy).
// compromise has no dependency tree, so "who did what" comes from each verb's subject, its tense and form, and the
// words around it. Verified against the same 72 labelled intros as the Python version (test/intros.json).
import nlp from "compromise";

type Doc = ReturnType<typeof nlp>;

const ACHIEVE = new Set(["build", "launch", "ship", "grow", "scale", "lead", "reduce", "increase", "save", "win", "found",
  "create", "deliver", "raise", "cut", "improve", "double", "triple", "halve", "migrate", "hire", "mentor", "publish",
  "sell", "drop", "boost", "speed", "automate", "serve", "reach", "generate", "handle", "process", "onboard", "close",
  "lower", "shrink", "bring", "help", "design", "redesign", "rebuild", "run", "manage", "train", "answer", "treat"]);
const ROLE_VERBS = new Set(["work", "lead", "run", "manage", "own", "head", "teach", "study", "build", "design", "develop",
  "join", "found", "advise", "coach", "write", "research", "consult", "freelance", "handle", "oversee", "review", "draw"]);
const FOCUS = new Set(["love", "enjoy", "specialize", "specialise", "focus", "care", "like", "obsess"]);
const FORWARD = new Set(["want", "hope", "plan", "aim", "seek", "explore", "grow", "join", "apply"]);
const VALUE_VERBS = new Set(["help", "enable", "support", "serve", "teach", "coach", "advise", "let", "allow", "save", "make", "give"]);
const REACH = new Set(["ping", "reach", "ask", "message", "find", "come", "grab", "talk", "email", "slack", "dm", "contact", "drop", "send", "tell"]);
const PREVIEW = new Set(["show", "cover", "share", "walk", "talk", "explain", "tell", "look", "go", "dive", "discuss", "teach",
  "demo", "unpack", "learn", "see", "explore", "break", "reveal", "compare", "dig"]);
const LISTENER = new Set(["see", "learn", "leave", "know", "get", "discover", "walk", "find", "understand", "hear"]);
const DESIRE = new Set(["want", "hope", "plan", "aim", "wish", "try", "like", "love", "dream"]);
const NOT_PREVIEW = new Set(["be", "have", "stay", "try", "keep", "begin", "start", "get"]);
export const TRAIT_NOUNS = new Set(["learner", "generalist", "person", "people", "fan", "guy", "girl", "kind", "type", "player", "believer",
  "lover", "thinker", "communicator", "solver", "problem", "worker", "listener", "self", "starter", "fit", "one", "bit"]);
export const JOB_NOUNS = new Set(["engineer", "developer", "designer", "manager", "founder", "cofounder", "co-founder",
  "analyst", "scientist", "architect", "consultant", "counsel", "lawyer", "attorney", "accountant", "nurse", "doctor",
  "teacher", "professor", "student", "intern", "researcher", "recruiter", "marketer", "writer", "editor", "photographer",
  "pilot", "director", "lead", "head", "officer", "specialist", "coordinator", "administrator", "assistant", "associate",
  "executive", "physiotherapist", "therapist", "cardiologist", "physician", "chef", "owner", "producer", "strategist",
  "advisor", "adviser", "freelancer", "contractor", "technician", "operator", "programmer", "ceo", "cto", "cfo", "coo",
  "vp", "president", "partner", "practitioner", "captain", "agent", "coach", "illustrator", "artist", "postdoc",
  "agent", "chemist", "economist", "journalist", "paralegal", "pharmacist", "dentist", "surgeon", "consultant"]);
export const VAGUE = new Set(["thing", "things", "stuff", "different", "various", "everything", "something", "somehow", "lots",
  "whatever", "several", "bunch", "certain", "etc", "nice", "great", "good", "interesting", "really", "very", "basically"]);
const STATE_VERBS = new Set(["be", "have", "do", "get", "go", "like", "want", "think", "feel", "know", "mean", "guess",
  "seem", "say", "thank", "hope", "love"]);

const ME = /^(i|i'm|i've|i'd|i'll|we|we're|we've|we'd|we'll|my|our|me|us)\b/i;

interface Term { text: string; normal: string; root: string; tags: Set<string>; i: number }
interface Verb { inf: string; tense: string; form: string; subject: string; me: boolean; start: number; text: string; aux: string }
export interface Sentence {
  text: string;
  terms: Term[];
  verbs: Verb[];
  people: string[];
  quantities: string[]; // counted things: numbers, percentages, money, short durations
  durations: string[]; // spans of experience: "five years", "the last decade"
  places: string[];
}

function words(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9%$']+/g) ?? [];
}

export function parse(text: string): Sentence {
  const doc: Doc = nlp(text);
  doc.compute("root");
  const terms: Term[] = [];
  for (const s of doc.json() as any[]) {
    for (const t of s.terms) {
      if (!t.text) continue; // compromise expands contractions into silent terms ("I'm" -> I + am)
      terms.push({ text: t.text, normal: t.normal ?? t.text.toLowerCase(), root: t.root ?? t.normal, tags: new Set(t.tags), i: terms.length });
    }
  }
  const verbs: Verb[] = [];
  doc.verbs().forEach((v: any) => {
    const j = v.json()[0]?.verb ?? {};
    let subject = "";
    try { subject = v.subjects().text() || ""; } catch { /* no subject */ }
    const text = v.text();
    const first = v.json()[0]?.terms?.find((t: any) => t.text)?.text;
    verbs.push({
      inf: (j.infinitive || v.text("normal")).toLowerCase(), tense: j.grammar?.tense ?? "", form: j.grammar?.form ?? "",
      subject, me: ME.test(subject.trim()), start: first ? terms.findIndex((t) => t.text === first) : -1, text,
      aux: (j.auxiliary || "").toLowerCase(),
    });
  });
  const people = doc.people().out("array") as string[];
  const quantities: string[] = [];
  const durations: string[] = [];
  const spans = [
    ...doc.match("#Value (or|to) #Value #Duration").out("array") as string[],
    ...doc.match("(#Value|a|an|several|few|many|couple|past|last|the last|the past|over the last|over the past) #Value? (#Duration)").out("array") as string[],
  ];
  spans.forEach((span) => {
    const m = { text: () => span.toLowerCase().replace(/[.,!?]+$/, "") };
    const t = m.text();
    if (durations.some((d) => d.includes(t))) return;
    if (/\b(years|months|decades?)\b/.test(t) && !/^(last|this|next) (year|month)$/.test(t)) durations.push(t);
    else if (/\b(hours?|minutes?|seconds?|days?|weeks?)\b/.test(t) && /\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred/.test(t)) quantities.push(t);
  });
  if (/\bdecade/i.test(text) && !durations.length) durations.push("decade");
  doc.percentages().forEach((m: any) => quantities.push(m.text()));
  doc.money().forEach((m: any) => quantities.push(m.text()));
  doc.numbers().forEach((m: any) => {
    const t = m.text("normal");
    if (/^(19|20)\d\d\.?$/.test(t)) return; // a year, not a count
    if (durations.some((d) => d.split(/\s+/).includes(t.replace(/[.,]$/, "")))) return; // "five or six years": experience
    quantities.push(m.text());
  });
  const places = (doc.places().out("array") as string[]).concat(doc.organizations().out("array") as string[]);
  return { text, terms, verbs, people, quantities: [...new Set(quantities)], durations, places };
}

// ---------------------------------------------------------------- helpers

const lower = (s: Sentence) => s.text.toLowerCase();
const has = (s: Sentence, re: RegExp) => re.test(s.text);
const isJob = (t: Term) => !TRAIT_NOUNS.has(t.normal) && (JOB_NOUNS.has(t.normal) || JOB_NOUNS.has(t.root) || t.tags.has("Actor"));

/** A verb is an aim, not a done thing, when a want/hope/plan verb comes earlier in the same sentence. */
function desire(s: Sentence, v: Verb): boolean {
  return s.verbs.some((u) => u !== v && u.start >= 0 && u.start < v.start && DESIRE.has(u.inf)) || /\b(i'd|would)\s+(love|like)\b/i.test(s.text);
}
const future = (s: Sentence, v: Verb) =>
  /future/i.test(v.tense) || /\b(will|'ll)\b/.test(v.aux) || new RegExp(`\\b(going to|'ll|will)\\s+(\\w+\\s+)?${v.inf}`, "i").test(s.text);
const progressive = (v: Verb) => /progressive/.test(v.form) && !/perfect/.test(v.form);

// ---------------------------------------------------------------- detectors: (sentence, index, count) -> found?

type Detector = (s: Sentence, k: number, n: number) => boolean;

const name: Detector = (s, k) => {
  if (k > 1) return false;
  if (s.people.length) return true;
  if (/\b(?:[Mm]y name is|[Mm]y name's|I'm|I am|[Tt]his is)\s+[A-Z][a-z]/.test(s.text)) return true;
  if (/^\s*(?:\w+[!,.]\s+)?([A-Z][a-z]+)\s+here\b/.test(s.text)) return true; // "Morning! Ahmed here"
  return /^(hi|hello|hey)[,!]?\s+([A-Z][a-z]+)\s*,/i.test(s.text) && !/^(hi|hello|hey)[,!]?\s+(everyone|all|team|folks|there)\b/i.test(s.text);
};

const role: Detector = (s, k) => {
  const t = lower(s);
  // "I'm a backend engineer", "I am the new office manager", "I'm Alex, a product designer"
  const copula = /\b(i'm|i am|we're|we are)\s+(?:[A-Z][\w-]*,?\s+(?:and\s+)?)?(?:a|an|the|our)\s+([\w-]+\s+){0,3}?([\w-]+)\b/i.exec(s.text);
  const plain = /\b(?:I'm|I am)\s+(?:a|an)\s+([a-z-]+)[.,!]?(?:\s|$)/.exec(s.text);
  if (plain && !TRAIT_NOUNS.has(plain[1]) && s.terms.some((x) => x.normal === plain[1] && x.tags.has("Noun"))) return true;
  if (copula) {
    const after = s.terms.filter((x) => x.tags.has("Noun") && t.indexOf(x.normal) > t.indexOf(copula[0].toLowerCase().split(" ")[0]));
    if (after.some(isJob)) return true;
  }
  // a title next to the name, or a title fragment, near the start: "Ahmed here, backend developer" / "Senior product manager."
  if (k <= 2 && s.terms.some(isJob) && (s.people.length || s.verbs.length === 0 || /\b(i'm|i am)\b/i.test(s.text))
      && !s.verbs.some((v) => DESIRE.has(v.inf))) return true;
  // "I'm Felix from finance"
  if (/\b(?:I'm|I am)\s+[A-Z]\w+\s+(from|on|in)\s+(the\s+)?[a-z]/.test(s.text) && !s.places.length) return true;
  for (const v of s.verbs) {
    if (!v.me || desire(s, v)) continue;
    // "I lead the mobile team", "I work at a bank", "I'm working on payments", "I just joined the data team"
    if (ROLE_VERBS.has(v.inf) && (/simple-present/.test(v.form) || progressive(v) || v.inf === "join")
        && !/\b(that|which|who)\s+\w+$/.test(t.slice(0, Math.max(0, t.indexOf(v.text.toLowerCase())) ).trim())) return true;
    // "I have been working as a mobile developer", "I just started as a backend engineer"
    if (new RegExp(`${v.text.toLowerCase().split(" ").pop()}\\s+as\\s+(a|an|the)\\s+`, "i").test(t)
        && (v.form !== "simple-past" || ["start", "join", "begin"].includes(v.inf))) return true;
    // "I recently moved over from sales to customer success"
    if (["move", "switch", "transfer"].includes(v.inf) && /\bto\b/.test(t)) return true;
  }
  return false;
};

const experience: Detector = (s, k) => {
  if (s.durations.length) return true;
  if (/\bsince (19|20)\d\d\b/i.test(s.text)) return true;
  if (/\b(trained|studied|worked|was)\b.*\b(in|at)\s+[A-Z]/.test(s.text) && k > 0) return true;
  return k > 0 && s.places.length > 0 && s.verbs.some((v) => v.me && /past/i.test(v.tense));
};

const proof: Detector = (s) => {
  if (!s.quantities.length) return false;
  for (const v of s.verbs) {
    if (ACHIEVE.has(v.inf) && !desire(s, v) && !future(s, v)) return true;
  }
  // "a 40% drop in churn", "$2M in revenue": a quantified outcome
  return /%|percent|\$|dollars?|euros?|pounds?/i.test(s.quantities.join(" "));
};

const proofPartial: Detector = (s) => s.verbs.some((v) => ACHIEVE.has(v.inf) && /past/i.test(v.tense) && v.me);

const scale: Detector = (s) =>
  s.quantities.some((q) => !/\b(minutes?|hours?|weeks?|days?|seconds?)\b/.test(q))
  && s.verbs.some((v) => v.me && !desire(s, v) && !future(s, v) && !STATE_VERBS.has(v.inf));

const focus: Detector = (s) =>
  s.verbs.some((v) => FOCUS.has(v.inf) && v.me) || /\b(passionate|expertise|expert|best at|good at|known for|strength)\b/i.test(s.text);

const close: Detector = (s, k, n) => {
  if (k < n * 0.5) return false;
  for (const v of s.verbs) {
    if (!v.me) continue;
    if (FORWARD.has(v.inf)) return true;
    if (v.inf === "look" && /\blook(ing)?\s+(for|forward|to)\b/i.test(s.text)) return true;
    if (v.inf === "love" && /\b(i'd|would)\s+love\b/i.test(s.text)) return true;
    if (future(s, v) && v.inf !== "be") return true;
  }
  return /\b(excited|eager|keen|my goal|next|which is why|that's why|why i'm here|here to)\b/i.test(s.text);
};

const value: Detector = (s) => {
  for (const v of s.verbs) {
    if (!v.me || desire(s, v)) continue;
    if (VALUE_VERBS.has(v.inf)) return true;
    // "gives laptops to kids", "draw books for small publishers": who the work is for
    const after = s.text.slice(s.text.toLowerCase().indexOf(v.text.toLowerCase()));
    if (/\b(for|to)\s+(\w+\s+){0,3}?\w+s\b/i.test(after) && s.terms.some((t) => t.tags.has("Plural"))) return true;
  }
  return /\bso (that|they|you|people|customers|clients|teams|users)\b/i.test(s.text) || /\b(that|which|who)\s+(helps?|gives?|serves?|lets?)\b/i.test(s.text);
};

const hook: Detector = (s, k, n) => {
  if (k < n * 0.5) return false;
  if (s.text.trim().endsWith("?")) return true;
  if (s.verbs.some((v) => v.me && progressive(v))) return true;
  return /\b(right now|currently|these days|lately|curious)\b/i.test(s.text);
};

const currentWork: Detector = (s) => {
  if (s.verbs.some((v) => v.me && !desire(s, v) && (progressive(v) || /\b(will|'ll) be \w+ing\b/i.test(s.text)))) return true;
  // "a contractor helping with the security audit"
  if (s.terms.some((t, i) => isJob(t) && s.terms[i + 1]?.tags.has("Gerund"))) return true;
  if (/\b(i'm|i am|we're|we are)\b[^.]*\b\w+ing\b/i.test(s.text) && !/\b(hoping|looking forward|planning)\b/i.test(s.text)) return true;
  return /\b(this (quarter|month|week|sprint|year|half)|right now|currently|these days)\b/i.test(s.text);
};

const reach: Detector = (s) => {
  const first = s.terms[0];
  if (first && REACH.has(first.root) && !/^(i|we)\b/i.test(s.text)) return true; // imperative: "Ping me", "Ask me anything"
  if (/\b(ping|message|ask|find|email|contact|dm|slack|tell)\s+me\b|\breach out\b|\bcome (find|say)\b|\bsend (it|them) my way\b/i.test(s.text)) return true;
  if (/\b(happy to help|feel free|if you('re| are)? ?(need|have|want|ever|signing|stuck|working)|my way|door is (always )?open|drop me a (line|note)|say hi|don't hesitate)\b/i.test(s.text)) return true;
  return /\b(outside (of )?work|free time|spare time|fun fact|weekends?|hobby|hobbies)\b/i.test(s.text);
};

const audience: Detector = (s) => {
  const text = s.text.replace(/\b(thank you|thanks|for having me|nice to meet you|good to see you)\b/gi, "");
  if (/\b(you|your|yours|yourself|you've|you're|you'll|you'd)\b/i.test(text)) return true;
  return /\b(all of us|each of us|many of us|most of us|we've all|we all|everyone here|anyone who|those of us)\b/i.test(text);
};

const preview: Detector = (s, k, n) => {
  if (k < n * 0.4) return false;
  for (const v of s.verbs) {
    // "By the end of this talk, you'll see why..."
    if (LISTENER.has(v.inf) && /\byou('ll| will)\b/i.test(s.text) && future(s, v)) return true;
    if (!v.me && !/\blet'?s\b/i.test(s.text)) continue;
    if (PREVIEW.has(v.inf)) {
      if (v.inf === "go" && !/\bgo (through|over)\b/i.test(s.text)) continue;
      if (future(s, v) || /\blet'?s\b/i.test(s.text) || /\b(going to)\b/i.test(s.text)) return true;
      if (desire(s, v) && /\b(want|like|love|hope) to\b/i.test(s.text)) return true; // "Today I want to talk about..."
    }
    // "We're going to dig into...", "I'll take you through..."
    if (future(s, v) && !NOT_PREVIEW.has(v.inf) && v.me) return true;
  }
  return false;
};

const role_ = role;
export const DETECTORS: Record<string, Detector> = {
  name, role: role_, experience, proof, focus, close, value, hook, current_work: currentWork, reach, audience, preview,
};

export interface Hit { found: number | null; partial: number | null }

export function detect(sentences: string[], keys: string[]): Record<string, Hit> {
  const parsed = sentences.map(parse);
  const n = parsed.length;
  const first = (fn: Detector) => { const k = parsed.findIndex((s, i) => fn(s, i, n)); return k < 0 ? null : k; };
  const out: Record<string, Hit> = {};
  for (const key of keys) {
    if (key === "credibility") {
      const hit = [first(proof), first(experience), first(scale), first(role_)].find((h) => h !== null) ?? null;
      out[key] = { found: hit, partial: hit === null ? first(proofPartial) : null };
    } else if (key === "proof") {
      const hit = first(proof);
      out[key] = { found: hit, partial: hit === null ? first(proofPartial) : null };
    } else {
      out[key] = { found: first(DETECTORS[key]), partial: null };
    }
  }
  return out;
}

// ---------------------------------------------------------------- language insights

export interface Insights {
  concrete: { i: number; text: string; kind: string }[];
  vague: { i: number; text: string }[];
  numbers: number[];
  nameWords: number[];
  iDid: number;
  weDid: number;
  mainVerbs: number;
  actionVerbs: number;
  haveBeen: number[];
  lemmas: Map<number, string>;
}

/** `sentences` holds [word index, word] pairs per sentence; findings point back at transcript word indices. */
export function insights(sentences: [number, string][][]): Insights {
  const out: Insights = { concrete: [], vague: [], numbers: [], nameWords: [], iDid: 0, weDid: 0, mainVerbs: 0, actionVerbs: 0, haveBeen: [], lemmas: new Map() };
  sentences.forEach((sent, k) => {
    const text = sent.map(([, w]) => w).join(" ");
    const s = parse(text);
    const find = (phrase: string) => {
      const first = words(phrase)[0];
      if (!first) return undefined;
      return sent.find(([, w]) => words(w)[0] === first)?.[0];
    };
    if (k <= 1 && !out.nameWords.length && s.people.length) {
      const ws = new Set(words(s.people[0]));
      out.nameWords = sent.filter(([, w]) => ws.has(words(w)[0] ?? "")).map(([i]) => i);
    } else if (k <= 1 && !out.nameWords.length) {
      const m = /\b(?:I'm|I am|[Mm]y name is)\s+([A-Z][a-z]+)/.exec(text);
      if (m) out.nameWords = sent.filter(([, w]) => words(w)[0] === m[1].toLowerCase()).map(([i]) => i);
    }
    const seen = new Set<string>();
    const add = (t: string, kind: string) => {
      const i = find(t);
      if (i === undefined || seen.has(t)) return;
      seen.add(t);
      out.concrete.push({ i, text: t.replace(/[.,!?]+$/, ""), kind });
    };
    s.durations.forEach((d) => { const orig = text.match(new RegExp(d.replace(/\s+/g, "\\s+"), "i"))?.[0] ?? d; add(orig, "DATE"); });
    s.quantities.forEach((q) => add(q, "QUANTITY"));
    s.places.forEach((p) => add(p, "ORG"));
    for (const q of s.quantities) for (const w of words(q)) { const i = sent.find(([, x]) => words(x)[0] === w)?.[0]; if (i !== undefined) out.numbers.push(i); }
    for (const [i, w] of sent) {
      const n = words(w)[0];
      if (n && VAGUE.has(n)) out.vague.push({ i, text: n });
    }
    for (const v of s.verbs) {
      if (v.inf && !STATE_VERBS.has(v.inf)) out.actionVerbs++;
      out.mainVerbs++;
      if (!STATE_VERBS.has(v.inf) && !desire(s, v)) {
        if (/^(i|i'm|i've|i'd|i'll)\b/i.test(v.subject)) out.iDid++;
        else if (/^(we|we're|we've|we'll|our team|my team)\b/i.test(v.subject) || /\bteam\b/i.test(v.subject)) out.weDid++;
      }
    }
    // "I have been working...": once per phrase, pointing at "been"
    for (const m of text.matchAll(/\b(?:have|'ve|has)\s+been\s+\w+ing\b/gi)) {
      const before = text.slice(0, m.index).split(/\s+/).filter(Boolean).length;
      const i = sent[Math.min(before + 1, sent.length - 1)]?.[0];
      if (i !== undefined && !out.haveBeen.includes(i)) out.haveBeen.push(i);
    }
    // base forms for spotting repetition across tenses and plurals
    const doc = nlp(text); doc.compute("root");
    const terms = (doc.json() as any[]).flatMap((x) => x.terms).filter((t: any) => t.text);
    let wi = 0;
    for (const t of terms) {
      while (wi < sent.length && words(sent[wi][1])[0] !== (t.normal ?? "").replace(/[^a-z0-9%$']/g, "")) wi++;
      if (wi >= sent.length) break;
      const tags: string[] = t.tags;
      if ((tags.includes("Noun") || tags.includes("Verb") || tags.includes("Adjective") || tags.includes("Adverb"))
          && !tags.includes("Pronoun") && !tags.includes("Auxiliary") && !tags.includes("Copula") && (t.normal ?? "").length > 2) {
        out.lemmas.set(sent[wi][0], (t.root || t.normal).toLowerCase());
      }
      wi++;
    }
  });
  out.numbers = [...new Set(out.numbers)].sort((a, b) => a - b);
  return out;
}
