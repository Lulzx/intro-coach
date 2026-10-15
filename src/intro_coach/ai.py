"""Optional content coaching: what to say, not how it sounded.

Providers (INTRO_COACH_AI_PROVIDER):
  openrouter (default)  NVIDIA Nemotron 3.5 Super VL, free tier, via OpenRouter. Needs a key, entered in the
                        app's Settings or set as OPENROUTER_API_KEY.
  claude                Claude via the Anthropic SDK. Needs `uv sync --extra ai` and ANTHROPIC_API_KEY
                        (or an `ant auth login` profile).
"""
import json
import os
import re
import urllib.error
import urllib.request

from . import settings

PROVIDER = os.environ.get("INTRO_COACH_AI_PROVIDER", "openrouter")
OPENROUTER_MODEL = os.environ.get("INTRO_COACH_OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")
CLAUDE_MODEL = os.environ.get("INTRO_COACH_MODEL", "claude-opus-5-5")

SYSTEM = """You are a speech coach who specialises in self-introductions: interview "tell me about yourself", \
networking, meetings and talk openers. You receive an automatic transcript of one practice take, with silent pauses \
marked like [1.4s], plus acoustic measurements already computed by software.

Coach the CONTENT and WORDING: structure against the shape given for this situation (`shape` in the measurements; \
a networking intro, a team-meeting intro and a talk opener each need different things), specificity, grammar and phrasing slips, and what to cut to fit the time slot. Delivery numbers (pace, pitch, pauses) \
are shown to the speaker separately; mention them only where they change what to say.

The transcript is machine-generated: a word that makes no sense in context is probably a recognition error for a \
similar-sounding word (for example "pie thon" for "Python"). Infer the intended word only when the \
sound and context make it clear, and point out that it came out unclear rather than calling it a vocabulary mistake. \
When you can't tell what a garbled phrase was, don't guess a specific term: leave it out of the rewrite or use a \
[placeholder], and name it in a fix so the speaker says it more clearly.

Keep every fact the speaker gave; never invent employers, skills, technologies, numbers or achievements. Where a number would help and none \
was given, use a placeholder like [X%] for them to fill in. The rewrite must be speakable in the target time at about \
150 words per minute, follow the situation's shape step by step, and stay in the speaker's own voice and register, \
in plain spoken English."""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three sentences: the overall impression this intro makes."},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "fixes": {
            "type": "array",
            "description": "3-6 fixes, most important first, each about one specific phrase the speaker said.",
            "items": {
                "type": "object",
                "properties": {
                    "said": {"type": "string", "description": "Verbatim words copied from the transcript (3-15 words), not a label."},
                    "issue": {"type": "string", "description": "What's wrong with that phrase, in one sentence."},
                    "try": {"type": "string", "description": "The words to say instead."},
                },
                "required": ["said", "issue", "try"],
                "additionalProperties": False,
            },
        },
        "rewrite": {"type": "string", "description": "An improved intro script that fits the target time."},
        "rewrite_seconds": {"type": "integer", "description": "Estimated speaking time of the rewrite."},
        "next_take_focus": {"type": "string", "description": "The single thing to concentrate on in the next take."},
    },
    "required": ["summary", "strengths", "fixes", "rewrite", "rewrite_seconds", "next_take_focus"],
    "additionalProperties": False,
}


def status() -> dict:
    """What the UI shows: which provider and model, and whether it can run right now."""
    if PROVIDER == "claude":
        import importlib.util

        installed = importlib.util.find_spec("anthropic") is not None
        return {"provider": "claude", "label": "Claude", "model": CLAUDE_MODEL, "ready": installed,
                "hint": None if installed else "Install the ai extra: uv sync --extra ai"}
    key = bool(settings.openrouter_key()[0])
    return {"provider": "openrouter", "label": "Nemotron 3.5 Super (OpenRouter)", "model": OPENROUTER_MODEL,
            "ready": key, "hint": None if key else "Add your OpenRouter API key in Settings to turn this on."}


def marked_transcript(report: dict) -> str:
    """Transcript with pauses of 0.5s or more written inline, so the model sees where the speaker stalled."""
    words, pauses = report["words"], {p["after"]: p for p in report["pauses"] if p["dur"] >= 0.5}
    out = []
    for i, w in enumerate(words):
        out.append(w["w"])
        if i in pauses:
            out.append(f"[{pauses[i]['dur']:.1f}s]")
    return " ".join(out)


def build_prompt(report: dict, notes: str = "") -> str:
    from .scenarios import SCENARIOS

    m = report["metrics"]
    shape = SCENARIOS.get(report["scenario"], SCENARIOS["interview"])["shape"]
    facts = {
        "shape": {"steps": [f"{st['name']}: {st['say']}" for st in shape["steps"]], "ending": shape["land"]},
        "scenario": report["scenario"], "question": report["prompt"], "target_seconds": report["target_seconds"],
        "actual_seconds": m["speaking_time"], "wpm": m["wpm"], "silence_pct": m["silence_pct"],
        "mid_thought_pauses": m["hesitations"], "fillers": m["fillers"] + m["hidden_fillers"],
        "pitch_range_semitones": m["pitch_range_st"], "statements_ending_on_rise": m["uptalk"],
        "structure_found": {e["key"]: e["found"] for e in report["structure"]["elements"]},
    }
    user = f"<measurements>\n{json.dumps(facts, indent=1)}\n</measurements>\n\n<transcript>\n{marked_transcript(report)}\n</transcript>"
    if notes.strip():
        user += f"\n\n<speaker_notes>\n{notes.strip()}\n</speaker_notes>\n(Background the speaker wrote about themselves; facts here may be used in the rewrite.)"
    return user


def feedback(report: dict, notes: str = "") -> dict:
    user = build_prompt(report, notes)
    out = _claude(user) if PROVIDER == "claude" else _openrouter(user)
    missing = [k for k in SCHEMA["required"] if k not in out]
    if missing:
        raise RuntimeError(f"feedback was missing {', '.join(missing)}; try again")
    return out


def _parse_json(text: str) -> dict:
    """Structured output should be bare JSON; tolerate a code fence or stray prose around it anyway."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            text = text[a:b + 1]
    return json.loads(text)


def _openrouter(user: str) -> dict:
    key = settings.openrouter_key()[0]
    if not key:
        raise RuntimeError("no OpenRouter API key: add one in Settings")
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "intro_feedback", "strict": True, "schema": SCHEMA}},
        # only route to providers that honour the schema
        "provider": {"require_parameters": True},
        "max_tokens": 16000,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "http://localhost/intro-coach", "X-Title": "Intro Coach"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            data = json.loads(res.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        if e.code == 429:
            raise RuntimeError("OpenRouter free-tier rate limit hit; wait a minute and try again") from e
        raise RuntimeError(f"OpenRouter error {e.code}: {detail}") from e
    if "error" in data:
        raise RuntimeError(f"OpenRouter: {data['error'].get('message', data['error'])}")
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("the feedback was cut off; try again")
    out = _parse_json(choice["message"].get("content") or "")
    out["model"] = data.get("model", OPENROUTER_MODEL)
    return out


def _claude(user: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to review this take.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's feedback was cut off; try again.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    out = json.loads(text)
    out["model"] = response.model
    return out


def check_openrouter_key(key: str) -> str | None:
    """None if OpenRouter accepts the key, otherwise a short reason."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15):
            return None
    except urllib.error.HTTPError as e:
        return "OpenRouter rejected this key" if e.code in (401, 403) else f"OpenRouter returned {e.code}"
    except urllib.error.URLError as e:
        return f"couldn't reach OpenRouter ({e.reason})"
