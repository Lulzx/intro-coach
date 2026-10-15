// Content coaching with any provider/model, the way opencode does it: the models.dev catalogue says which AI SDK
// package speaks to each provider (and its base URL), and the AI SDK runs the call with a zod schema.
import { generateText, Output, NoObjectGeneratedError, type LanguageModel } from "ai";
import { z } from "zod";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { createMistral } from "@ai-sdk/mistral";
import { createGroq } from "@ai-sdk/groq";
import { createXai } from "@ai-sdk/xai";
import { createDeepSeek } from "@ai-sdk/deepseek";
import { createCerebras } from "@ai-sdk/cerebras";
import { createDeepInfra } from "@ai-sdk/deepinfra";
import { createPerplexity } from "@ai-sdk/perplexity";
import { createCohere } from "@ai-sdk/cohere";
import { SCENARIOS } from "./engine/scenarios";
import type { Report } from "./engine/analysis";

type Fetch = typeof globalThis.fetch;

export interface CatalogModel { id: string; name: string; structured_output?: boolean; tool_call?: boolean; reasoning?: boolean; cost?: { input?: number; output?: number }; limit?: { context?: number } }
export interface CatalogProvider { id: string; name: string; npm?: string; api?: string; env?: string[]; doc?: string; models: Record<string, CatalogModel> }
export type Catalog = Record<string, CatalogProvider>;

export const DEFAULT_PROVIDER = "openrouter";
export const DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free";
const CATALOG_URL = "https://models.dev/api.json";

// AI SDK packages bundled into the app; any other provider with a base URL goes through the OpenAI-compatible adapter
const FACTORIES: Record<string, (o: { apiKey: string; baseURL?: string; fetch: Fetch; name: string }) => (id: string) => LanguageModel> = {
  "@ai-sdk/openai": (o) => { const p = createOpenAI({ apiKey: o.apiKey, baseURL: o.baseURL, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/anthropic": (o) => {
    const p = createAnthropic({ apiKey: o.apiKey, baseURL: o.baseURL, fetch: o.fetch, headers: { "anthropic-dangerous-direct-browser-access": "true" } });
    return (id) => p(id);
  },
  "@ai-sdk/google": (o) => { const p = createGoogleGenerativeAI({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@openrouter/ai-sdk-provider": (o) => {
    const p = createOpenRouter({ apiKey: o.apiKey, fetch: o.fetch, headers: { "HTTP-Referer": "https://github.com/Lulzx/intro-coach", "X-Title": "Intro Coach" } });
    return (id) => p(id) as unknown as LanguageModel;
  },
  "@ai-sdk/mistral": (o) => { const p = createMistral({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/groq": (o) => { const p = createGroq({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/xai": (o) => { const p = createXai({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/deepseek": (o) => { const p = createDeepSeek({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/cerebras": (o) => { const p = createCerebras({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/deepinfra": (o) => { const p = createDeepInfra({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/perplexity": (o) => { const p = createPerplexity({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/cohere": (o) => { const p = createCohere({ apiKey: o.apiKey, fetch: o.fetch }); return (id) => p(id); },
  "@ai-sdk/openai-compatible": (o) => {
    const p = createOpenAICompatible({ name: o.name, apiKey: o.apiKey, baseURL: o.baseURL!, fetch: o.fetch });
    return (id) => p(id);
  },
};

/** Providers the app can call: a bundled AI SDK package, or any provider with an OpenAI-compatible base URL. */
export function supported(p: CatalogProvider): boolean {
  if (p.npm && FACTORIES[p.npm] && (p.npm !== "@ai-sdk/openai-compatible" || p.api)) return true;
  return !!p.api && !p.npm?.includes("azure") && !p.npm?.includes("bedrock") && !p.npm?.includes("vertex");
}

export function model(p: CatalogProvider, modelId: string, apiKey: string, fetch: Fetch): LanguageModel {
  const factory = (p.npm && FACTORIES[p.npm]) || (p.api ? FACTORIES["@ai-sdk/openai-compatible"] : undefined);
  if (!factory) throw new Error(`${p.name} isn't supported yet`);
  return factory({ apiKey, baseURL: p.api, fetch, name: p.id })(modelId);
}

// A small built-in catalogue so the app works offline and before models.dev has loaded
const FALLBACK: Catalog = {
  openrouter: { id: "openrouter", name: "OpenRouter", npm: "@openrouter/ai-sdk-provider", api: "https://openrouter.ai/api/v1", env: ["OPENROUTER_API_KEY"],
    models: { [DEFAULT_MODEL]: { id: DEFAULT_MODEL, name: "Nemotron 3.5 Super VL (free)", structured_output: true, cost: { input: 0, output: 0 } } } },
  anthropic: { id: "anthropic", name: "Anthropic", npm: "@ai-sdk/anthropic", env: ["ANTHROPIC_API_KEY"],
    models: { "claude-opus-5-5": { id: "claude-opus-5-5", name: "Claude Opus 5.5", structured_output: true } } },
  openai: { id: "openai", name: "OpenAI", npm: "@ai-sdk/openai", env: ["OPENAI_API_KEY"], models: {} },
  google: { id: "google", name: "Google", npm: "@ai-sdk/google", env: ["GOOGLE_GENERATIVE_AI_API_KEY"], models: {} },
};

/** Fetch the models.dev catalogue, trimmed to what the settings screen needs. Falls back to the built-in list. */
export async function loadCatalog(fetch: Fetch, cached?: Catalog): Promise<Catalog> {
  try {
    const res = await fetch(CATALOG_URL);
    if (!res.ok) throw new Error(String(res.status));
    const raw = (await res.json()) as Catalog;
    const out: Catalog = {};
    for (const [id, p] of Object.entries(raw)) {
      if (!supported(p)) continue;
      const models: Record<string, CatalogModel> = {};
      for (const [mid, m] of Object.entries(p.models ?? {})) {
        models[mid] = { id: mid, name: m.name, structured_output: m.structured_output, tool_call: m.tool_call, reasoning: m.reasoning, cost: m.cost, limit: m.limit };
      }
      out[id] = { id, name: p.name, npm: p.npm, api: p.api, env: p.env, doc: p.doc, models };
    }
    return out;
  } catch {
    return cached ?? FALLBACK;
  }
}

// ---------------------------------------------------------------- the coaching call

const SYSTEM = `You are a speech coach who specialises in self-introductions: interview "tell me about yourself", networking, meetings and talk openers. You receive an automatic transcript of one practice take, with silent pauses marked like [1.4s], plus acoustic measurements already computed by software.

Coach the CONTENT and WORDING: structure against the shape given for this situation (\`shape\` in the measurements; a networking intro, a team-meeting intro and a talk opener each need different things), specificity, grammar and phrasing slips, and what to cut to fit the time slot. Delivery numbers (pace, pitch, pauses) are shown to the speaker separately; mention them only where they change what to say.

The transcript is machine-generated: a word that makes no sense in context is probably a recognition error for a similar-sounding word (for example "pie thon" for "Python"). Infer the intended word only when the sound and context make it clear, and point out that it came out unclear rather than calling it a vocabulary mistake. When you can't tell what a garbled phrase was, don't guess a specific term: leave it out of the rewrite or use a [placeholder], and name it in a fix so the speaker says it more clearly.

Keep every fact the speaker gave; never invent employers, skills, technologies, numbers or achievements. Where a number would help and none was given, use a placeholder like [X%] for them to fill in. The rewrite must be speakable in the target time at about 150 words per minute, follow the situation's shape step by step, and stay in the speaker's own voice and register, in plain spoken English.`;

export const FeedbackSchema = z.object({
  summary: z.string().describe("Two or three sentences: the overall impression this intro makes."),
  strengths: z.array(z.string()),
  fixes: z.array(z.object({
    said: z.string().describe("Verbatim words copied from the transcript (3-15 words), not a label."),
    issue: z.string().describe("What's wrong with that phrase, in one sentence."),
    try: z.string().describe("The words to say instead."),
  })).describe("3-6 fixes, most important first, each about one specific phrase the speaker said."),
  rewrite: z.string().describe("An improved intro script that fits the target time."),
  rewrite_seconds: z.number().int().describe("Estimated speaking time of the rewrite."),
  next_take_focus: z.string().describe("The single thing to concentrate on in the next take."),
});
export type Feedback = z.infer<typeof FeedbackSchema> & { model?: string };

function markedTranscript(r: Report): string {
  const pauses = new Map(r.pauses.filter((p) => p.dur >= 0.5).map((p) => [p.after, p]));
  return r.words.map((w, i) => (pauses.has(i) ? `${w.w} [${pauses.get(i)!.dur.toFixed(1)}s]` : w.w)).join(" ");
}

export function buildPrompt(r: Report, notes = ""): string {
  const m = r.metrics, shape = (SCENARIOS[r.scenario] ?? SCENARIOS.interview).shape;
  const facts = {
    shape: { steps: shape.steps.map((s) => `${s.name}: ${s.say}`), ending: shape.land },
    scenario: r.scenario, question: r.prompt, target_seconds: r.target_seconds,
    actual_seconds: m.speaking_time, wpm: m.wpm, silence_pct: m.silence_pct,
    mid_thought_pauses: m.hesitations, fillers: m.fillers + m.hidden_fillers,
    pitch_range_semitones: m.pitch_range_st, statements_ending_on_rise: m.uptalk,
    structure_found: Object.fromEntries(r.structure.elements.map((e) => [e.key, e.found])),
  };
  let user = `<measurements>\n${JSON.stringify(facts, null, 1)}\n</measurements>\n\n<transcript>\n${markedTranscript(r)}\n</transcript>`;
  if (notes.trim()) user += `\n\n<speaker_notes>\n${notes.trim()}\n</speaker_notes>\n(Background the speaker wrote about themselves; facts here may be used in the rewrite.)`;
  return user;
}

export async function feedback(lm: LanguageModel, r: Report, notes: string, signal?: AbortSignal): Promise<Feedback> {
  try {
    const res = await generateText({
      model: lm, system: SYSTEM, prompt: buildPrompt(r, notes), maxRetries: 1, abortSignal: signal,
      output: Output.object({ schema: FeedbackSchema }),
    });
    return { ...res.output, model: res.response?.modelId };
  } catch (e) {
    if (NoObjectGeneratedError.isInstance(e)) throw new Error("The model didn't return usable feedback. Try again, or pick a model with structured output.");
    throw friendly(e);
  }
}

/** One tiny call to check that a key works with the chosen model. */
export async function check(lm: LanguageModel, signal?: AbortSignal): Promise<void> {
  try {
    await generateText({ model: lm, prompt: "Reply with the single word OK.", maxRetries: 0, abortSignal: signal, maxOutputTokens: 16 });
  } catch (e) {
    throw friendly(e);
  }
}

function friendly(e: unknown): Error {
  const err = e as { statusCode?: number; message?: string; responseBody?: string };
  const status = err.statusCode;
  if (status === 401 || status === 403) return new Error("The provider rejected this API key.");
  if (status === 429) return new Error("Rate limited by the provider. Wait a minute and try again.");
  if (status === 404) return new Error("The provider doesn't know this model.");
  return new Error(err.message?.slice(0, 300) || "The request failed.");
}
