"""Intro structure detection with spaCy (en_core_web_sm, ~12 MB, CPU).

Each detector looks at one parsed sentence at a time and uses entities (names, durations, percentages, money),
lemmas (so "built", "building" and "builds" all count as "build"), tense and the dependency parse (who did what),
instead of matching exact phrases. Sentences come from the transcript's own punctuation so each one keeps its time.
"""
import re
import threading
from functools import lru_cache

QUANTITY_ENTS = {"PERCENT", "MONEY", "CARDINAL", "QUANTITY"}
ACHIEVE = {"build", "launch", "ship", "grow", "scale", "lead", "reduce", "increase", "save", "win", "found", "create",
           "deliver", "raise", "cut", "improve", "double", "triple", "halve", "migrate", "hire", "mentor", "publish",
           "sell", "drop", "boost", "speed", "automate", "serve", "reach", "generate", "handle", "process", "onboard",
           "close", "lower", "shrink", "bring", "help", "design", "redesign", "rebuild", "run", "manage", "train"}
ROLE_VERBS = {"work", "lead", "run", "manage", "own", "head", "teach", "study", "build", "design", "develop", "join",
              "found", "advise", "coach", "write", "research", "consult", "freelance", "handle", "oversee"}
FOCUS_LEMMAS = {"love", "enjoy", "specialize", "specialise", "focus", "care", "like", "passionate", "expertise",
                "expert", "strength", "obsess", "obsessed", "fascinate"}
FORWARD_LEMMAS = {"want", "hope", "plan", "aim", "seek", "excite", "excited", "eager", "keen", "look", "love", "next",
                  "goal", "join", "explore", "grow"}
VALUE_LEMMAS = {"help", "enable", "support", "serve", "teach", "coach", "advise", "let", "allow", "save", "make"}
REACH_LEMMAS = {"ping", "reach", "ask", "message", "find", "come", "grab", "talk", "email", "slack", "dm", "contact", "drop", "tell"}
PREVIEW_LEMMAS = {"show", "cover", "share", "walk", "talk", "explain", "tell", "look", "go", "dive", "discuss", "teach",
                  "demo", "unpack", "learn", "see", "explore", "break", "reveal", "compare"}
# what the listener will get: "you'll see why", "you'll leave with"
LISTENER_LEMMAS = {"see", "learn", "leave", "know", "get", "discover", "walk", "find", "understand", "hear"}
DESIRE = {"want", "hope", "plan", "aim", "wish", "try", "like", "love", "dream", "would"}
# nouns that describe a trait, not a job ("I'm a quick learner", "I'm a people person")
TRAIT_NOUNS = {"learner", "generalist", "person", "people", "fan", "guy", "girl", "kind", "type", "player", "believer", "lover",
               "thinker", "communicator", "solver", "problem", "worker", "listener", "self", "starter", "fit", "one"}
JOB_NOUNS = {"engineer", "developer", "designer", "manager", "founder", "cofounder", "analyst", "scientist", "architect",
             "consultant", "counsel", "lawyer", "attorney", "accountant", "nurse", "doctor", "teacher", "professor",
             "student", "intern", "researcher", "recruiter", "marketer", "writer", "editor", "photographer", "pilot",
             "director", "lead", "head", "officer", "specialist", "coordinator", "administrator", "assistant",
             "associate", "executive", "physiotherapist", "therapist", "cardiologist", "chef", "owner", "producer",
             "strategist", "advisor", "adviser", "freelancer", "contractor", "technician", "operator", "programmer",
             "ceo", "cto", "cfo", "coo", "vp", "president", "partner", "practitioner", "captain", "officer", "agent"}
# the small model's lemmatizer gets a few common forms wrong ("hoping" -> "hop")
LEMMA_FIX = {"hoping": "hope", "hoped": "hope", "hopes": "hope", "excited": "excite", "joining": "join",
             "leading": "lead", "led": "lead", "focusing": "focus", "focused": "focus"}


def lem(t) -> str:
    return LEMMA_FIX.get(t.lower_, t.lemma_.lower())
HUMAN_DETAIL = re.compile(r"\b(outside (of )?work|free time|spare time|fun fact|weekends?|hobby|hobbies)\b", re.I)


_nlp = None
_lock = threading.Lock()


def model():
    global _nlp
    with _lock:  # loaded once, from the startup warm-up thread or the first request, whichever comes first
        if _nlp is None:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        return _nlp


def warm_up() -> None:
    model()("Hi, I'm Alex.")


def _me(tok) -> bool:
    return tok.lower_ in {"i", "we", "my", "our"}


def _subject_is_me(verb) -> bool:
    subj = [c for c in verb.children if c.dep_ in {"nsubj", "nsubjpass"}]
    # "I'm a contractor helping with the audit": a verb describing my role noun
    if not subj and verb.dep_ == "acl" and verb.head.dep_ in {"attr", "appos"}:
        return _subject_is_me(verb.head.head) or verb.head.head.pos_ == "PROPN"
    if not subj and verb.dep_ in {"xcomp", "conj", "advcl", "relcl", "acl"}:
        return _subject_is_me(verb.head)
    # "I run a nonprofit that gives laptops to kids": what my thing does counts as mine
    if verb.dep_ == "relcl" and verb.head.dep_ in {"dobj", "attr", "pobj"} and verb.head.head.pos_ in {"VERB", "AUX", "ADP"}:
        anchor = verb.head.head.head if verb.head.head.pos_ == "ADP" else verb.head.head
        if any(_me(c) for c in anchor.children if c.dep_ == "nsubj"):
            return True
    return any(_me(s) or any(_me(t) for t in s.subtree) for s in subj)


def _aux(verb):
    return {c.lower_ for c in verb.children if c.dep_ in {"aux", "auxpass"}}


def _desire_head(tok) -> bool:
    """True when the word sits anywhere under want/hope/plan ("I want to build models that help millions"):
    an aim, not a done result."""
    return any(lem(a) in DESIRE for a in tok.ancestors) or any(c.lower_ in {"'d", "would"} for c in tok.head.children)


# ---------------------------------------------------------------- per-sentence detectors

def name(doc, k, n):
    if k > 1:
        return False
    if any(e.label_ == "PERSON" for e in doc.ents):
        return True
    toks = list(doc)
    for i, t in enumerate(toks):
        if t.pos_ != "PROPN" or not t.text[:1].isupper():
            continue
        nxt = toks[i + 1].lower_ if i + 1 < len(toks) else ""
        prev = toks[i - 1].lower_ if i else ""
        # "Ahmed here, ..." / "Hi, Ines, legal counsel."
        if nxt == "here" or (prev == "," and i <= 3 and nxt == "," and toks[0].lower_ in {"hi", "hello", "hey"}):
            return True
    # "I'm Alex" / "my name is Alex" when NER misses an unusual name
    for t in doc:
        if t.pos_ == "PROPN" and t.dep_ in {"attr", "appos", "npadvmod", "dobj"} and t.head.lemma_ in {"be", "call"}:
            return True
    return bool(re.search(r"\b(my name is|my name's|I'm|I am|this is)\s+[A-Z]", doc.text))


def _job_noun(t) -> bool:
    return t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() not in TRAIT_NOUNS


def role(doc, k, n):
    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    for t in doc:
        # "I am a designer", "Alex, a product designer": a common-noun description of the speaker
        if t.pos_ == "NOUN" and t.dep_ in {"attr", "appos"} and _job_noun(t):
            head = t.head
            if head.lemma_ == "be" and _subject_is_me(head):
                return True
            if head.pos_ == "PROPN" and (head.dep_ == "attr" and _subject_is_me(head.head) or k <= 1):
                return True
        # "I'm Felix from finance" / "I'm Rosa, on the payments squad" (the parser may hang the "from" on "'m")
        if t.pos_ == "PROPN" and t.dep_ == "attr" and _subject_is_me(t.head):
            for prep in (c for c in list(t.children) + list(t.head.children) if c.lower_ in {"from", "on", "in"}):
                if any(c.pos_ == "NOUN" for c in prep.children):
                    return True
        # "I have been working as a mobile developer", "I just started as a backend engineer"
        if t.lower_ == "as" and t.head.pos_ == "VERB" and _subject_is_me(t.head) \
                and (t.head.tag_ != "VBD" or lem(t.head) in {"start", "join", "begin"}) \
                and not _desire_head(t.head) and any(_job_noun(c) for c in t.children):
            return True
        # "I recently moved over from sales to customer success"
        if lem(t) in {"move", "switch", "transfer"} and _subject_is_me(t) and any(c.lower_ == "to" for c in t.subtree):
            return True
        # "I lead the mobile team", "I work at a bank", "I'm working on payments", "I just joined the data team"
        # (relative clauses excluded: in "a system that works", "works" isn't the speaker's job)
        if t.pos_ == "VERB" and t.lemma_ in ROLE_VERBS and t.dep_ != "relcl" and _subject_is_me(t) and not _desire_head(t):
            aux = _aux(t)
            if t.tag_ in {"VBP", "VBZ"} or (t.tag_ == "VBG" and aux & {"am", "'m", "are", "'re"} and not aux & {"been"}):
                return True
            if t.lemma_ == "join":
                return True
    # a title on its own near the start: "Senior product manager." / "Hi, Ines, legal counsel."
    if k <= 2 and not any(v.pos_ == "VERB" for v in verbs):
        if any(t.lemma_.lower() in JOB_NOUNS for t in doc if t.pos_ == "NOUN"):
            return True
    return False


def experience(doc, k, n):
    for e in doc.ents:
        if e.label_ != "DATE" or re.search(r"\b(last|this|next)\s+(year|month)\b", e.text, re.I):
            continue
        if re.search(r"\bdecades?\b", e.text, re.I):
            return True
        if re.search(r"\b(years|months)\b", e.text, re.I) and (
                any(t.like_num for t in e) or re.search(r"\b(a|several|few|many|couple|past|last)\b", e.text, re.I)):
            return True
        if e.label_ == "DATE" and re.search(r"\bsince\b", doc.text[max(0, e.start_char - 7): e.start_char], re.I):
            return True
    # "Before that I was at Stripe": a past role at a named organisation
    past = any(t.tag_ in {"VBD", "VBN"} and _subject_is_me(t) for t in doc if t.pos_ in {"VERB", "AUX"})
    return past and any(e.label_ in {"ORG", "GPE"} for e in doc.ents) and k > 0


def _quantities(doc):
    """Numbers that can measure a result. Short durations count ("from 24 hours to under one");
    spans of years or months don't: "5 years designing tools" is experience, not a result."""
    out = []
    for e in doc.ents:
        if e.label_ in QUANTITY_ENTS:
            out.append(e)
        elif e.label_ in {"TIME", "DATE"} and any(t.like_num for t in e) \
                and re.search(r"\b(hours?|minutes?|seconds?|days?|weeks?)\b", e.text, re.I):
            out.append(e)
    return out


def proof(doc, k, n):
    if not _quantities(doc):
        return False
    for t in doc:
        if t.lemma_ in ACHIEVE and t.pos_ in {"VERB", "NOUN"} and not _desire_head(t):
            if t.pos_ == "NOUN" or t.tag_ in {"VBD", "VBN", "VBP", "VBZ"}:
                return True
    # "a 40% drop in churn", "$2M in revenue": a quantified outcome noun
    return any(e.label_ in {"PERCENT", "MONEY"} for e in doc.ents)


def scale(doc, k, n):
    """A number attached to something the speaker or their team does: "we answer 10,000 tickets a month"."""
    if not any(e.label_ in QUANTITY_ENTS for e in doc.ents):
        return False
    return any(t.pos_ == "VERB" and _subject_is_me(t) and not _desire_head(t) and not _aux(t) & {"'ll", "will"}
               for t in doc)


def proof_partial(doc, k, n):
    return any(t.lemma_ in ACHIEVE and t.tag_ in {"VBD", "VBN"} and _subject_is_me(t) for t in doc)


def focus(doc, k, n):
    for t in doc:
        if t.lemma_ in FOCUS_LEMMAS and (t.pos_ in {"ADJ", "NOUN"} or _subject_is_me(t)):
            return True
    return bool(re.search(r"\b(best at|good at|known for)\b", doc.text, re.I))


def close(doc, k, n):
    if k < n * 0.5:
        return False
    for t in doc:
        if lem(t) in FORWARD_LEMMAS and (t.pos_ in {"ADJ", "NOUN"} or _subject_is_me(t) or t.lower_ == "next"):
            if lem(t) == "look" and not any(c.lower_ in {"for", "forward", "to"} for c in t.children):
                continue
            return True
        if t.lower_ in {"'ll", "will"} and _subject_is_me(t.head):
            return True
    return bool(re.search(r"\b(which is why|that's why|why I'm here|here to)\b", doc.text, re.I))


def value(doc, k, n):
    for t in doc:
        if t.lemma_ in VALUE_LEMMAS and t.pos_ == "VERB" and _subject_is_me(t):
            if any(c.dep_ in {"dobj", "ccomp", "xcomp", "dative"} for c in t.children):
                return True
        # "gives laptops to kids", "draw books for small publishers": who the work is for
        if t.pos_ == "VERB" and _subject_is_me(t) and not _desire_head(t):
            for prep in (c for c in t.subtree if c.dep_ in {"prep", "dative"} and c.lower_ in {"for", "to"} and c.head.i == t.i):
                if any(o.tag_ in {"NNS", "NNPS"} for o in prep.children):
                    return True
    return bool(re.search(r"\bso (that|they|you|people|customers|clients|teams|users)\b", doc.text, re.I))


def hook(doc, k, n):
    if k < n * 0.5:
        return False
    if doc.text.rstrip().endswith("?"):
        return True
    for t in doc:
        if t.tag_ == "VBG" and _aux(t) & {"am", "'m"}:
            return True
    return bool(re.search(r"\b(right now|currently|these days|lately|curious)\b", doc.text, re.I))


def current_work(doc, k, n):
    for t in doc:
        if t.tag_ == "VBG" and _aux(t) & {"am", "'m", "are", "'re", "be"} and _subject_is_me(t) and not _desire_head(t):
            return True
        if t.tag_ == "VBG" and t.dep_ == "acl" and _job_noun(t.head) and _subject_is_me(t):
            return True
    if any(e.label_ == "DATE" and re.search(r"\bthis (quarter|month|week|sprint|year|half)\b", e.text, re.I) for e in doc.ents):
        return True
    return bool(re.search(r"\b(right now|currently|these days)\b", doc.text, re.I))


def reach(doc, k, n):
    for t in doc:
        # imperatives: "ping me", "come find me", "ask me about..."
        if t.dep_ == "ROOT" and t.tag_ == "VB" and t.lemma_ in REACH_LEMMAS:
            return True
        if t.lemma_ in REACH_LEMMAS and any(c.lower_ == "me" for c in t.children):
            return True
    if re.search(r"\b(happy to help|feel free|if you('re| are)? ?(need|have|want|ever|signing|stuck|working)|my way|"
                 r"door is (always )?open|drop me a (line|note)|say hi|don't hesitate)\b", doc.text, re.I):
        return True
    return bool(HUMAN_DETAIL.search(doc.text))


def audience(doc, k, n):
    text = re.sub(r"\b(thank you|thanks|for having me|nice to meet you|good to see you)\b", "", doc.text, flags=re.I)
    if re.search(r"\b(you|your|yours|yourself|you've|you're|you'll|you'd)\b", text, re.I):
        return True
    return bool(re.search(r"\b(all of us|each of us|many of us|most of us|we've all|we all|everyone here|"
                          r"anyone who|those of us)\b", text, re.I))


def preview(doc, k, n):
    if k < n * 0.4:
        return False
    for t in doc:
        if t.pos_ != "VERB":
            continue
        aux = _aux(t)
        if lem(t) in PREVIEW_LEMMAS:
            if lem(t) == "go" and not any(c.lower_ in {"through", "over"} for c in t.children):
                continue
            if aux & {"'ll", "will", "'re", "are", "am", "'m"} and (_subject_is_me(t) or t.head.lower_ == "going"):
                return True
            if t.head.lower_ in {"let", "let's"} or any(c.lower_ in {"let", "'s"} for c in t.children):
                return True
            if lem(t.head) == "go" and _aux(t.head) & {"am", "'m", "are", "'re"}:
                return True
            # "Today I want to talk about..."
            if lem(t.head) in {"want", "like", "love", "hope"} and _subject_is_me(t.head):
                return True
        # "We're going to dig into...", "I'll take you through..."
        future = aux & {"'ll", "will"} or (t.head.lower_ == "going" and _aux(t.head) & {"am", "'m", "are", "'re"})
        if future and lem(t) not in {"be", "have", "stay", "try", "keep", "begin", "start", "get"} \
                and (_subject_is_me(t) or _subject_is_me(t.head)):
            return True
        # "By the end of this talk, you'll see why..."
        if lem(t) in LISTENER_LEMMAS and aux & {"'ll", "will"} and any(c.lower_ == "you" for c in t.children):
            return True
    return any(e.label_ == "DATE" and e.text.lower() == "today" for e in doc.ents) and any(t.lower_ in {"'ll", "will"} for t in doc)


DETECTORS = {"name": name, "role": role, "experience": experience, "proof": proof, "focus": focus, "close": close,
             "value": value, "hook": hook, "current_work": current_work, "reach": reach, "audience": audience,
             "preview": preview}


def detect(sentences: list[str], keys: list[str]) -> dict:
    """{key: {"found": sentence index or None, "partial": sentence index or None}} for each requested check."""
    docs = list(model().pipe(sentences))
    n = len(docs)

    def first(fn):
        return next((k for k, d in enumerate(docs) if fn(d, k, n)), None)

    out = {}
    for key in keys:
        if key == "credibility":  # a result, real experience, or the scale of what they do
            hit = next((h for h in (first(proof), first(experience), first(scale), first(role)) if h is not None), None)
            out[key] = {"found": hit, "partial": None if hit is not None else first(proof_partial)}
        elif key == "proof":
            hit = first(proof)
            out[key] = {"found": hit, "partial": None if hit is not None else first(proof_partial)}
        else:
            out[key] = {"found": first(DETECTORS[key]), "partial": None}
    return out


def word_lemmas(sentences: list[list[tuple[int, str]]]) -> dict[int, str]:
    """Base form of each content word, keyed by transcript word index, for spotting repetition across tenses and
    plurals ("build", "built", "building"). `sentences` holds (word index, word) pairs per sentence."""
    texts, spans = [], []
    for sent in sentences:
        pos, sp = 0, []
        for i, w in sent:
            sp.append((pos, pos + len(w), i))
            pos += len(w) + 1
        texts.append(" ".join(w for _, w in sent))
        spans.append(sp)
    out = {}
    for doc, sp in zip(model().pipe(texts), spans):
        for t in doc:
            if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and not t.is_stop and len(t) > 2:
                i = next((i for a, b, i in sp if a <= t.idx < b), None)
                if i is not None and i not in out:
                    out[i] = t.lemma_.lower()
    return out


# ---------------------------------------------------------------- language insights

CONCRETE_ENTS = {"ORG", "GPE", "LOC", "PRODUCT", "PERCENT", "MONEY", "CARDINAL", "QUANTITY", "NORP", "FAC", "EVENT",
                 "WORK_OF_ART", "LANGUAGE", "LAW", "DATE", "TIME"}
VAGUE = {"thing", "things", "stuff", "different", "various", "everything", "something", "somehow", "lots", "whatever",
         "several", "bunch", "certain", "etc", "nice", "great", "good", "interesting", "really", "very", "basically"}
STATE_VERBS = {"be", "have", "do", "get", "go", "like", "want", "think", "feel", "know", "mean", "guess", "seem", "say",
               "thank", "hope", "love"}


def _parse(sentences):
    """Parse sentences given as [(word index, word)] and map every token back to its transcript word."""
    texts, spans = [], []
    for sent in sentences:
        pos, sp = 0, []
        for i, w in sent:
            sp.append((pos, pos + len(w), i))
            pos += len(w) + 1
        texts.append(" ".join(w for _, w in sent))
        spans.append(sp)

    def word_of(sp, t):
        return next((i for a, b, i in sp if a <= t.idx < b), None)

    return [(doc, sp, word_of) for doc, sp in zip(model().pipe(texts), spans)]


@lru_cache(maxsize=4096)
def _common_word(text: str) -> bool:
    return model()(text.lower())[0].pos_ in {"VERB", "PRON", "ADV", "ADJ", "AUX", "DET", "ADP", "INTJ"} \
        or text.lower() in VAGUE


def insights(sentences: list[list[tuple[int, str]]]) -> dict:
    """What the words say about how specific, active and owned the intro is. Word indices point into the transcript."""
    concrete, vague, numbers, name_words = [], [], set(), []
    i_did, we_did, main_verbs, action_verbs, have_been = [], [], 0, 0, []
    for k, (doc, sp, word_of) in enumerate(_parse(sentences)):
        for e in doc.ents:
            wi = word_of(sp, e[0])
            if e.label_ == "PERSON" and k <= 1 and not name_words:
                name_words = sorted({word_of(sp, t) for t in e if word_of(sp, t) is not None})
                continue
            if e.label_ in CONCRETE_ENTS and wi is not None:
                if e.label_ in {"DATE", "TIME"} and not any(t.like_num for t in e) and "decade" not in e.text.lower():
                    continue  # "today", "last year": not a detail
                if len(e) == 1 and not e.text.isupper() and _common_word(e.text):
                    continue  # "Analyze", "Something": a capitalised ordinary word from the transcript, not a name
                concrete.append({"i": wi, "text": e.text, "kind": e.label_})
            if e.label_ in {"PERCENT", "MONEY", "CARDINAL", "QUANTITY"}:
                numbers |= {word_of(sp, t) for t in e if word_of(sp, t) is not None}
        for t in doc:
            wi = word_of(sp, t)
            if t.lower_ in VAGUE and wi is not None and t.pos_ in {"NOUN", "ADJ", "PRON", "ADV"}:
                vague.append({"i": wi, "text": t.lower_})
            if t.pos_ == "VERB" and t.dep_ in {"ROOT", "conj", "ccomp"}:
                main_verbs += 1
                if lem(t) not in STATE_VERBS:
                    action_verbs += 1
            if t.pos_ == "VERB" and lem(t) not in STATE_VERBS and not _desire_head(t):
                subj = [c.lower_ for c in t.children if c.dep_ == "nsubj"]
                if "i" in subj:
                    i_did.append(wi)
                elif "we" in subj or any(c.dep_ == "nsubj" and "team" in c.lower_ for c in t.children):
                    we_did.append(wi)
            if t.tag_ == "VBG" and _aux(t) >= {"been"} and _aux(t) & {"have", "'ve", "has"}:
                have_been.append(wi)
    return {"concrete": concrete, "vague": vague, "numbers": sorted(numbers), "name_words": name_words,
            "i_did": len(i_did), "we_did": len(we_did), "main_verbs": main_verbs, "action_verbs": action_verbs,
            "have_been": [i for i in have_been if i is not None]}
