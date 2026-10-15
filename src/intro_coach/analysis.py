"""Turn timed words + acoustics into intro-specific coaching: metrics, scores, structure checks and tips."""
import re
from collections import Counter

import numpy as np

from . import nlp
from .prosody import STEP, Prosody, runs
from .scenarios import SCENARIOS

FILLERS = {"um", "uh", "umm", "uhm", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm"}
HEDGES = ["kind of", "sort of", "i think", "i guess", "i feel like", "maybe", "probably", "basically",
          "actually", "literally", "pretty much", "a little bit", "you know", "i mean", "stuff like that",
          "and stuff", "or something", "things like that"]
STOP = set("""a an the and or but so of to in on at for from with by as is are was were be been being have has had do does
did i me my mine we our you your he she it its they them their this that these those there here which who whom what
when where why how not no yes very really just also too than then into over about up out more most some any all can
could would should will shall may might must am im i'm i've ive it's its that's there's let's like one get got make
made things thing lot lots well even quite""".split())


def norm(w: str) -> str:
    return re.sub(r"[^\w'-]", "", w.lower())


def refine_words(words, P: Prosody):
    """TDT durations can stretch a word across the silence after it; clip each word to its sounding part."""
    out = []
    for w in words:
        s, e = w["s"], w["e"]
        m = P.window(s, e)
        sounding = np.where(~P.silent[m])[0]
        if len(sounding):
            ts = P.t[m]
            ns, ne = float(ts[sounding[0]]), float(ts[sounding[-1]] + STEP)
            if ns - s > 0.15:
                s = ns
            if e - ne > 0.15:
                e = ne
        out.append({**w, "s": round(s, 3), "e": round(max(e, s + 0.05), 3)})
    return out


def sentences(words):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        if re.search(r"[.!?]$", w["w"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def text_of(words, idx):
    return " ".join(words[i]["w"] for i in idx)


def quote(words, i, before=3, after=3):
    a, b = max(0, i - before), min(len(words), i + after + 1)
    return " ".join(words[j]["w"] for j in range(a, b))


def spoken_len(w: str) -> int:
    """Rough letters-as-spoken: "40%" is said "forty percent", so digits and symbols count for more than one."""
    return max(1, len(re.sub(r"[^A-Za-z]", "", w)) + 4 * sum(c.isdigit() for c in w) + 7 * w.count("%") + 6 * w.count("$"))


def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))


def ramp(x, bad, good):
    """0 at `bad`, 100 at `good`, linear between (works in either direction)."""
    if bad == good:
        return 100.0
    return clamp((x - bad) / (good - bad) * 100)


def analyse(words, P: Prosody, duration: float, scenario: str = "interview", target: float | None = None) -> dict:
    sc = SCENARIOS.get(scenario, SCENARIOS["interview"])
    target = float(target or sc["target"])
    if not words:
        return {"empty": True, "duration": duration, "scenario": scenario, "target_seconds": target,
                "tips": [{"area": "Recording", "severity": "critical", "title": "No speech detected",
                          "detail": "Check the microphone and try again.", "evidence": []}]}

    words = refine_words(words, P)
    toks = [norm(w["w"]) for w in words]
    t0, t1 = words[0]["s"], words[-1]["e"]
    span = max(t1 - t0, 1e-6)
    minutes = span / 60
    sents = sentences(words)

    # --- fillers -------------------------------------------------------------------------------------------
    filler_idx = [i for i, t in enumerate(toks) if t in FILLERS]
    for i, t in enumerate(toks):  # "like," used as a filler, not "I like maps"
        if t == "like" and (words[i]["w"].endswith(",") or (i and words[i - 1]["w"].endswith(","))):
            filler_idx.append(i)
    filler_idx = sorted(set(filler_idx))
    is_filler = set(filler_idx)
    content_n = len(words) - len(filler_idx)

    # Sounds the recogniser left out: voiced, un-transcribed gaps between words (often a drawn-out "uhh")
    hidden = []
    for i in range(len(words) - 1):
        a, b = words[i]["e"], words[i + 1]["s"]
        if b - a < 0.3:
            continue
        m = P.window(a, b)
        v = (P.f0[m] > 0) & ~P.silent[m]
        best = run = 0
        for x in v:
            run = run + 1 if x else 0
            best = max(best, run)
        if best * STEP >= 0.25:
            hidden.append({"t": round(a, 2), "dur": round(best * STEP, 2), "after": words[i]["w"]})

    # Stretched words: held far longer than their length needs while still voiced (a drawl, thinking aloud)
    stretched = []
    for i, w in enumerate(words):
        d = w["e"] - w["s"]
        letters = len(re.sub(r"\W", "", w["w"]))
        if i not in is_filler and d > max(0.75, 0.09 * letters + 0.45):
            if (P.voiced(w["s"], w["e"]).size * STEP) > 0.5 * d:
                stretched.append({"i": i, "t": round(w["s"], 2), "dur": round(d, 2), "word": w["w"]})

    # --- pauses --------------------------------------------------------------------------------------------
    inside = P.window(t0, t1) & P.silent
    pauses = []
    for a, b in runs(inside, P.t, 0.25):
        prev = max((i for i, w in enumerate(words) if w["s"] < (a + b) / 2), default=0)
        w = words[prev]["w"]
        kind = "sentence" if re.search(r"[.!?]$", w) else "phrase" if re.search(r"[,;:—-]$", w) else "hesitation"
        if prev in is_filler:
            kind = "hesitation"
        pauses.append({"s": round(a, 2), "e": round(b, 2), "dur": round(b - a, 2), "kind": kind, "after": prev})
    pause_total = sum(p["dur"] for p in pauses)
    hesitations = [p for p in pauses if p["kind"] == "hesitation" and p["dur"] >= 0.5]
    long_pauses = [p for p in pauses if p["dur"] >= 2.0]

    # --- pace ----------------------------------------------------------------------------------------------
    wpm = content_n / minutes
    articulation = content_n / max((span - pause_total) / 60, 1e-6)
    sent_stats = []
    for idx in sents:
        a, b = words[idx[0]]["s"], words[idx[-1]]["e"]
        n = sum(1 for i in idx if i not in is_filler)
        sent_stats.append({"s": a, "e": b, "n": n, "wpm": n / max((b - a) / 60, 1e-6), "idx": idx})
    pace_windows = []
    for a in np.arange(t0, t1, 10.0):
        n = sum(1 for i, w in enumerate(words) if a <= w["s"] < a + 10 and i not in is_filler)
        pace_windows.append({"t": round(float(a), 1), "wpm": round(n * 6)})

    # --- pitch ---------------------------------------------------------------------------------------------
    speech = P.window(t0, t1)
    voiced_hz = P.f0[speech & (P.f0 > 0)]
    st = P.semitones(voiced_hz) if len(voiced_hz) else np.zeros(1)
    pitch_range = float(np.percentile(st, 95) - np.percentile(st, 5))
    pitch_sd = float(st.std())

    monotone = []
    for a in np.arange(t0, max(t0 + 0.01, t1 - 8), 2.0):
        v = P.voiced(a, a + 8)
        if v.size >= 150 and P.semitones(v).std() < 1.2:
            if monotone and a <= monotone[-1][1]:
                monotone[-1][1] = float(a + 8)
            else:
                monotone.append([float(a), float(a + 8)])
    monotone_secs = sum(b - a for a, b in monotone)

    endings = []
    for k, idx in enumerate(sents):
        last = words[idx[-1]]
        is_q = last["w"].endswith("?")
        tail = P.voiced(max(last["s"], last["e"] - 0.35), last["e"])
        pre = P.voiced(last["e"] - 1.15, last["e"] - 0.35)
        if tail.size >= 5 and pre.size >= 10:
            delta = float(12 * np.log2(np.median(tail) / np.median(pre)))
            shape = "rising" if delta > 1.5 else "falling" if delta < -1.0 else "flat"
            endings.append({"t": round(last["e"], 2), "delta": round(delta, 1), "shape": shape, "question": is_q,
                            "text": text_of(words, idx[-6:]), "final": k == len(sents) - 1})
    uptalk = [e for e in endings if e["shape"] == "rising" and not e["question"]]

    # --- volume --------------------------------------------------------------------------------------------
    loud = P.db[speech & ~P.silent]
    vol_sd = float(loud.std()) if loud.size else 0.0
    trailing = []
    for s in sent_stats:
        idx = s["idx"]
        if len(idx) < 4:
            continue
        sent_db = P.db[P.window(s["s"], s["e"]) & ~P.silent]
        lw = words[idx[-1]]
        last_db = P.db[P.window(lw["s"], lw["e"]) & ~P.silent]
        if sent_db.size and last_db.size:
            drop = float(np.mean(last_db) - np.mean(sent_db))
            if drop < -6:
                trailing.append({"t": round(lw["s"], 2), "drop": round(drop, 1), "text": text_of(words, idx[-5:])})
    thirds = [P.db[P.window(t0 + k * span / 3, t0 + (k + 1) * span / 3) & ~P.silent] for k in range(3)]
    energy_drift = float(np.mean(thirds[2]) - np.mean(thirds[0])) if all(x.size for x in thirds) else 0.0

    # --- language ------------------------------------------------------------------------------------------
    lower = " " + " ".join(toks) + " "
    hedges = Counter()
    for h in HEDGES:
        c = len(re.findall(rf"(?<=\s){re.escape(h)}(?=\s)", lower))
        if c:
            hedges[h] = c
    # sentences without fillers parse better; word indices are kept so findings point back at the transcript
    clean = [[(i, words[i]["w"]) for i in idx if i not in is_filler] for idx in sents]
    clean_sents = [" ".join(w for _, w in sent) or "." for sent in clean]
    lemma_of = {i: lem for i, lem in nlp.word_lemmas(clean).items() if lem not in STOP}
    content_words = Counter(lemma_of.values())
    overused = [(w, c) for w, c in content_words.most_common()
                if c >= 3 and c / max(content_n, 1) >= 0.015 and w not in nlp.VAGUE][:6]

    # --- language insights (spaCy) ---------------------------------------------------------------------------
    ins = nlp.insights(clean)
    perfect = max(len(ins["have_been"]), len(re.findall(r"(?<=\s)(i have been|i've been|ive been)(?=\s)", lower)))
    vague = Counter(v["text"] for v in ins["vague"])
    vague_n = sum(vague.values())
    concrete = ins["concrete"]
    action_share = ins["action_verbs"] / ins["main_verbs"] if ins["main_verbs"] else None
    sent_of = {i: k for k, idx in enumerate(sents) for i in idx}

    # Are the numbers stressed? Compare each number's pitch and loudness with the rest of its sentence.
    stressed = []
    for i in ins["numbers"]:
        w, idx = words[i], sents[sent_of[i]]
        a, b = words[idx[0]]["s"], words[idx[-1]]["e"]
        nv, sv = P.voiced(w["s"], w["e"]), P.voiced(a, b)
        ndb, sdb = P.db[P.window(w["s"], w["e"]) & ~P.silent], P.db[P.window(a, b) & ~P.silent]
        if ndb.size and sdb.size:
            st = float(12 * np.log2(np.median(nv) / np.median(sv))) if nv.size >= 3 and sv.size >= 10 else None
            if st is not None and abs(st) > 8:
                st = None  # an octave jump from the pitch tracker, not real intonation
            slow = (w["e"] - w["s"]) / spoken_len(w["w"]) / max(
                np.median([(words[j]["e"] - words[j]["s"]) / spoken_len(words[j]["w"]) for j in idx]), 1e-3)
            stressed.append({"i": i, "t": round(w["s"], 2), "word": w["w"], "st": None if st is None else round(st, 1),
                             "db": round(float(np.mean(ndb) - np.mean(sdb)), 1), "slow": round(float(slow), 2)})
    # stress shows as louder, higher or slower than the rest of the sentence; any one of them is enough
    flat_numbers = [x for x in stressed if x["db"] < 1.5 and (x["st"] is None or x["st"] < 1.0) and x["slow"] < 1.25]

    # Was the name rushed? Time per letter against the speaker's own typical word, and the gap after it.
    name_pace = None
    if ins["name_words"]:
        nw = [words[i] for i in ins["name_words"]]
        letters = sum(len(re.sub(r"\W", "", w["w"])) for w in nw)
        per_letter = (nw[-1]["e"] - nw[0]["s"]) / max(letters, 1)
        typical = np.median([(w["e"] - w["s"]) / len(re.sub(r"\W", "", w["w"])) for w in words
                             if len(re.sub(r"\W", "", w["w"])) >= 4] or [per_letter])
        nxt = ins["name_words"][-1] + 1
        gap = words[nxt]["s"] - nw[-1]["e"] if nxt < len(words) else 1.0
        name_pace = {"t": round(nw[0]["s"], 2), "name": " ".join(w["w"] for w in nw),
                     "ratio": round(float(per_letter / typical), 2), "gap": round(float(gap), 2),
                     "rushed": bool(per_letter < 0.8 * typical and gap < 0.2)}
    and_starts = sum(1 for idx in sents if toks[idx[0]] in {"and", "so", "but"})
    long_sents = [s for s in sent_stats if s["n"] > 35]

    # --- intro structure (spaCy: entities, lemmas, tense and who-did-what; see nlp.py) ------------------------
    text = " ".join(w["w"] for w in words)
    sent_texts = [text_of(words, idx) for idx in sents]

    def clip(s, n=140):
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."

    def evidence(k):
        return None if k is None else (clip(sent_texts[k]), words[sents[k][0]]["s"])

    hits = nlp.detect(clean_sents, [c["key"] for c in sc["checks"]])
    elements = []
    for c in sc["checks"]:
        h = hits[c["key"]]
        elements.append({**c, "found": h["found"] is not None, "partial": h["partial"] is not None,
                         "evidence": evidence(h["found"] if h["found"] is not None else h["partial"])})
    weak_open = toks[0] in {"so", "um", "uh", "okay", "ok", "well", "like", "yeah", "alright"}
    last_sentence = sent_texts[-1] if sent_texts else ""
    weak_end = bool(re.search(r"(\byeah\W*$|that's (it|about it|all)|that is (it|all)|\bi guess\W*$|\banyway\W*$|or something\W*$)", last_sentence, re.I))
    final_ending = next((e for e in endings if e["final"]), None)
    length_ratio = span / target

    # --- scores --------------------------------------------------------------------------------------------
    pace_score = min(ramp(wpm, 90, 130), ramp(wpm, 210, 165))
    if articulation > 200:
        pace_score -= min(25, (articulation - 200) * 0.8)
    silence_pct = pause_total / span * 100
    fluency = 100 - max(0, silence_pct - 20) * 2 - 4 * len(hesitations) / minutes \
        - 5 * (len(filler_idx) + len(hidden)) / minutes - 3 * len(stretched) / minutes - 5 * len(long_pauses)
    voice = 0.5 * ramp(pitch_range, 2.5, 8) + 0.5 * ramp(pitch_sd, 0.8, 2.2) \
        - min(24, 8 * len(uptalk)) - 20 * monotone_secs / span - min(15, 5 * len(trailing))
    structure = sum(e["weight"] * (1 if e["found"] else 0.5 if e.get("partial") else 0) for e in elements)
    structure += 10 * (1 if 0.8 <= length_ratio <= 1.2 else 0.5 if 0.6 <= length_ratio <= 1.5 else 0)
    structure -= 5 * weak_open + 5 * weak_end
    language = 100 - 4 * sum(hedges.values()) / minutes - 5 * len(overused) - 5 * len(long_sents) \
        - 3 * max(0, perfect - 2) - 2 * max(0, and_starts - 2) - min(15, 3 * max(0, vague_n - 2)) \
        - (10 if not concrete and content_n > 30 else 0) - (8 if action_share is not None and action_share < 0.4 else 0)
    scores = {"pace": clamp(pace_score), "fluency": clamp(fluency), "voice": clamp(voice),
              "structure": clamp(structure), "language": clamp(language)}
    weights = {"pace": 0.2, "fluency": 0.25, "voice": 0.2, "structure": 0.25, "language": 0.1}
    overall = round(sum(scores[k] * w for k, w in weights.items()))

    # --- tips ----------------------------------------------------------------------------------------------
    tips, strengths = [], []

    def ev(t, q):
        return {"t": round(float(t), 2), "quote": q}

    def tip(area, severity, title, detail, evidence=(), drill=None, impact=0.0):
        tips.append({"area": area, "severity": severity, "title": title, "detail": detail,
                     "evidence": list(evidence)[:4], "drill": drill, "impact": impact})

    if length_ratio > 1.25:
        tip("Structure", "serious", f"Too long: {span:.0f}s for a {target:.0f}s slot",
            f"Cut about {span - target:.0f}s. Keep one sentence per step of {sc['shape']['title']} and drop the rest.",
            drill=f"Write one sentence for each step ({sc['shape']['title']}), then say it with a {target:.0f}s timer.", impact=30 * (length_ratio - 1))
    elif length_ratio < 0.7:
        tip("Structure", "warning", f"Short: {span:.0f}s for a {target:.0f}s slot",
            f"There's room to fill out a step of {sc['shape']['title']}.", impact=10)
    else:
        strengths.append(f"Length fits the slot ({span:.0f}s of {target:.0f}s).")

    if hesitations:
        worst = sorted(hesitations, key=lambda p: -p["dur"])
        tip("Fluency", "critical" if len(hesitations) / minutes > 3 else "serious",
            f"{len(hesitations)} pause{'s' if len(hesitations) > 1 else ''} in the middle of a thought",
            "Pausing mid-clause sounds like searching for words. Pause after a complete idea instead, at a full stop.",
            [ev(p["s"], f"{quote(words, p['after'], 3, 0)} [{p['dur']:.1f}s] {quote(words, p['after'] + 1, 0, 3) if p['after'] + 1 < len(words) else ''}") for p in worst],
            "Read your intro aloud with the full stops marked; pause only there. Repeat the words you stalled on 5 times first.",
            impact=8 * len(hesitations) / minutes + 10)
    if silence_pct > 25:
        tip("Pace", "serious", f"{silence_pct:.0f}% of your time is silence",
            "Engaging speakers sit around 15-20%. Most of the gain comes from removing the mid-thought gaps.", impact=silence_pct - 20)
    if wpm < 115:
        tip("Pace", "serious" if wpm < 100 else "warning", f"Slow overall pace: {wpm:.0f} wpm",
            f"Aim for 130-165 wpm. Your speed while actually talking is {articulation:.0f} wpm, so the pauses are what slow you down"
            if articulation >= 130 else "Aim for 130-165 wpm.", impact=(130 - wpm) / 2)
    elif wpm > 180 or articulation > 210:
        fast = sorted(sent_stats, key=lambda s: -s["wpm"])[:2]
        tip("Pace", "serious", f"Rushing: {wpm:.0f} wpm ({articulation:.0f} while talking)",
            "Slow down on the parts that matter: your name, your role and your result.",
            [ev(s["s"], text_of(words, s["idx"][:10]) + "...") for s in fast],
            "Breathe at every full stop. Say your name, then a beat of silence.", impact=(wpm - 165) / 2)
    else:
        strengths.append(f"Comfortable pace ({wpm:.0f} wpm).")

    fillers_total = len(filler_idx) + len(hidden)
    if fillers_total / minutes > 2:
        tip("Fluency", "serious" if fillers_total / minutes > 5 else "warning",
            f"{fillers_total} filler sound{'s' if fillers_total > 1 else ''} ({fillers_total / minutes:.1f}/min)",
            "Swap 'um' and 'uh' for a silent breath. Closing your mouth while you think stops the sound.",
            [ev(words[i]["s"], quote(words, i)) for i in filler_idx[:3]] + [ev(h["t"], f"{h['after']} [voiced {h['dur']}s]") for h in hidden[:2]],
            "Record one take where every 'um' gets replaced by a full one-second silence.", impact=4 * fillers_total / minutes)
    elif fillers_total <= 1:
        strengths.append("Almost no filler sounds.")
    if len(stretched) >= 3:
        tip("Fluency", "warning", f"{len(stretched)} drawn-out words",
            "Holding a word while you think ('lea-ding...', 'so-o') is a hidden filler. Finish the word, then pause.",
            [ev(s["t"], f"{s['word']} ({s['dur']}s)") for s in stretched[:4]], impact=2 * len(stretched))

    if pitch_range < 6 or pitch_sd < 1.6:
        tip("Voice", "serious" if pitch_range < 4.5 else "warning", f"Flat pitch: {pitch_range:.1f} semitone range",
            "Engaging speakers use 8-12 semitones. Lift the one word per sentence that carries the meaning.",
            [ev(a, f"monotone {a - t0:.0f}s-{b - t0:.0f}s") for a, b in monotone[:3]],
            "Underline one word per sentence in your script and say it higher and slightly louder. Exaggerate on take one.",
            impact=(8 - pitch_range) * 4)
    elif pitch_range >= 8:
        strengths.append(f"Lively pitch ({pitch_range:.0f} semitone range).")
    if uptalk:
        sev = "serious" if any(e["final"] for e in uptalk) else "warning"
        tip("Voice", sev, f"{len(uptalk)} statement{'s' if len(uptalk) > 1 else ''} ending on a rise",
            "A rising end makes a statement sound like a question and undercuts confidence"
            + (", and your final line rises." if any(e["final"] for e in uptalk) else "."),
            [ev(e["t"], f"...{e['text']} (+{e['delta']} st)") for e in uptalk],
            "Say your last sentence three times, landing each one lower than the last.", impact=6 * len(uptalk))
    elif final_ending and final_ending["shape"] == "falling":
        strengths.append("Your final line lands with a confident downward pitch.")
    if trailing:
        tip("Voice", "warning", f"Volume drops at the end of {len(trailing)} sentence{'s' if len(trailing) > 1 else ''}",
            "The last word of a sentence is often the important one; keep breath for it.",
            [ev(x["t"], f"...{x['text']} ({x['drop']} dB)") for x in trailing], impact=3 * len(trailing))
    if energy_drift < -4:
        tip("Voice", "warning", f"Energy fades: {abs(energy_drift):.0f} dB quieter by the end",
            "Your close should be at least as strong as your opening.", impact=5)

    for e in (e for e in elements if not e["found"]):
        tip("Structure", "serious" if e["weight"] >= 20 else "warning", e["fix_title"], e["fix"],
            [ev(e["evidence"][1], "closest: " + e["evidence"][0])] if e.get("evidence") else [],
            impact=e["weight"] * (0.5 if e.get("partial") else 1))
    if weak_open:
        tip("Structure", "warning", f"Opening with \"{words[0]['w'].strip(',.')}\"",
            "Start straight with a greeting and your name.", [ev(t0, quote(words, 0, 0, 5))], impact=5)
    if weak_end:
        tip("Structure", "serious", "Weak final line", "Ending on 'so yeah' or 'that's it' drains the intro. " + sc["shape"]["land"],
            [ev(words[sents[-1][0]]["s"], last_sentence)], impact=10)

    if overused:
        tip("Language", "warning", "Repeated words: " + ", ".join(f"\"{w}\" x{c}" for w, c in overused[:4]),
            "Repetition makes a short intro sound vague. Replace with the specific thing you mean.", impact=3 * len(overused))
    if perfect >= 3:
        tip("Language", "warning", f"\"I have been...\" used {perfect} times",
            "Switch to active verbs: \"I led\", \"I built\", \"I shipped\".", impact=2 * perfect)
    if sum(hedges.values()) / minutes > 2:
        tip("Language", "warning", "Hedging words: " + ", ".join(f"\"{h}\" x{c}" for h, c in hedges.most_common(4)),
            "Hedges soften every claim. Cut them and state it.", impact=2 * sum(hedges.values()))
    if long_sents:
        tip("Language", "warning", f"{len(long_sents)} very long sentence{'s' if len(long_sents) > 1 else ''}",
            "Over 35 words in one breath is hard to follow. Split at the 'and's.",
            [ev(s["s"], text_of(words, s["idx"][:12]) + "...") for s in long_sents[:2]], impact=4 * len(long_sents))
    if any(e["found"] for e in elements if e["key"] == "name") and not weak_open:
        strengths.append("Clear opening with your name.")

    # insights from the language model
    if vague_n >= 3 and vague_n / max(content_n, 1) > 0.025:
        tip("Language", "warning", "Vague words: " + ", ".join(f"\"{w}\" x{c}" for w, c in vague.most_common(4)),
            "Swap each one for the specific thing: a company, a product, a number, a type of customer.",
            [ev(words[v["i"]]["s"], quote(words, v["i"], 4, 3)) for v in ins["vague"][:3]], impact=2 * vague_n)
    if not concrete and content_n > 30:
        tip("Language", "serious", "No concrete details",
            "Nothing a listener can hold on to: no company, place, product or number. Add at least one.", impact=12)
    elif len(concrete) >= 3:
        strengths.append("Specific details: " + ", ".join(c["text"] for c in concrete[:4]) + ".")
    if scenario == "interview" and ins["we_did"] >= 2 and ins["we_did"] > ins["i_did"]:
        tip("Language", "warning", f"More \"we\" ({ins['we_did']}) than \"I\" ({ins['i_did']})",
            "Interviewers are hiring you. Keep the team result, but say what you did: \"I designed...\", \"I led...\".",
            impact=6)
    if action_share is not None and ins["main_verbs"] >= 4 and action_share < 0.4:
        tip("Language", "warning", f"Few action verbs ({action_share:.0%} of sentences)",
            "Most sentences rest on \"I am\", \"I have\" or \"I like\". Verbs like led, built, cut and launched show what you did.",
            impact=6)
    if stressed and len(flat_numbers) >= max(1, (len(stressed) + 1) // 2):
        tip("Voice", "warning", "Your numbers don't stand out",
            "Key figures said at the same pitch and volume as the rest of the sentence get lost. Pause just before the number and lift it.",
            [ev(x["t"], f"{quote(words, x['i'], 3, 2)} ({x['db']:+.1f} dB vs the sentence)") for x in flat_numbers[:3]],
            "Say the sentence three times, each time hitting the number a little harder.", impact=7)
    elif stressed:
        strengths.append("You stressed your key numbers.")
    if name_pace and name_pace["rushed"]:
        tip("Voice", "warning", "Your name went by quickly",
            "People can't remember a name they didn't catch. Slow down on it and leave a beat after.",
            [ev(name_pace["t"], f"{name_pace['name']} (gap after: {name_pace['gap']:.1f}s)")],
            "Say just \"Hi, I'm [name].\", pause for one second, then carry on.", impact=6)

    order = {"critical": 0, "serious": 1, "warning": 2}
    tips.sort(key=lambda x: (order[x["severity"]], -x["impact"]))

    # --- timeline for the UI -------------------------------------------------------------------------------
    pitch_line = []
    for k in range(0, len(P.t), 2):
        f = P.f0[k]
        pitch_line.append([round(float(P.t[k]), 2), round(float(P.semitones(f)), 2) if f > 0 else None])
    overused_set = {w for w, _ in overused}
    vague_idx = {v["i"] for v in ins["vague"]} if vague_n >= 3 else set()
    word_out = []
    for i, w in enumerate(words):
        flags = []
        if i in is_filler:
            flags.append("filler")
        if lemma_of.get(i) in overused_set:
            flags.append("overused")
        if any(s["i"] == i for s in stretched):
            flags.append("stretched")
        if i in vague_idx:
            flags.append("vague")
        word_out.append({"w": w["w"], "s": w["s"], "e": w["e"], "f": flags})

    r = lambda x, n=1: round(float(x), n)  # noqa: E731
    return {
        "scenario": scenario, "prompt": sc["prompt"], "target_seconds": target,
        "duration": r(duration), "speech_start": r(t0, 2), "speech_end": r(t1, 2),
        "transcript": text, "words": word_out, "pauses": pauses,
        "overall": overall, "scores": {k: round(v) for k, v in scores.items()},
        "metrics": {
            "speaking_time": r(span), "words": content_n, "wpm": r(wpm, 0), "articulation_wpm": r(articulation, 0),
            "silence_pct": r(silence_pct, 0), "pauses": len(pauses), "hesitations": len(hesitations),
            "long_pauses": len(long_pauses), "fillers": len(filler_idx), "hidden_fillers": len(hidden),
            "stretched_words": len(stretched), "fillers_per_min": r(fillers_total / minutes),
            "pitch_median_hz": r(P.f0_median, 0), "pitch_range_st": r(pitch_range), "pitch_sd_st": r(pitch_sd, 2),
            "monotone_secs": r(monotone_secs, 0), "uptalk": len(uptalk), "volume_sd_db": r(vol_sd),
            "energy_drift_db": r(energy_drift), "trailing_off": len(trailing),
            "hedges": dict(hedges), "overused": overused, "i_have_been": perfect, "long_sentences": len(long_sents),
            "avg_sentence_words": r(np.mean([x["n"] for x in sent_stats]) if sent_stats else 0),
            "longest_sentence_words": max((x["n"] for x in sent_stats), default=0),
        },
        "insights": {
            "concrete": [{"text": c["text"], "kind": c["kind"], "t": round(words[c["i"]]["s"], 2)} for c in concrete],
            "vague": vague.most_common(), "i_did": ins["i_did"], "we_did": ins["we_did"],
            "action_share": None if action_share is None else round(action_share, 2),
            "numbers": stressed, "flat_numbers": len(flat_numbers), "name": name_pace,
        },
        "structure": {"elements": [{"key": e["key"], "label": e["label"], "found": e["found"], "partial": bool(e.get("partial")),
                                "evidence": e["evidence"][0] if e.get("evidence") else None,
                                "t": round(e["evidence"][1], 2) if e.get("evidence") else None} for e in elements],
                      "weak_open": weak_open, "weak_end": weak_end, "length_ratio": r(length_ratio, 2)},
        "endings": endings, "monotone": monotone, "hidden_fillers": hidden, "pace_windows": pace_windows,
        "tips": tips, "strengths": strengths,
        "timeline": {"pitch": pitch_line, "median_hz": r(P.f0_median, 0)},
    }
