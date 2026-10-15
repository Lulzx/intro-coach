"""Speech recognition with Fluid Speech Mini. Transcription comes back as a list of timed words.

    [{"w": "Hey,", "s": 0.56, "e": 0.88}, ...]

`w` keeps attached punctuation (the analysis uses it to tell sentence pauses from hesitations).
"""
import threading

ENGINE = "fluid-mini"
ENGINE_LABEL = "Fluid Speech Mini by FluidVoice"


_engine = None
_lock = threading.Lock()


def get_engine():
    """Load once. The server warms this up in a background thread, so a take that arrives while the model is still
    downloading or loading waits here instead of loading a half-downloaded copy."""
    global _engine
    with _lock:
        if _engine is None:
            from .fluid import FluidSpeechMini

            _engine = FluidSpeechMini()
        return _engine


def group_pieces(pieces, space="▁"):
    """Join subword pieces [(text, start, end)] into words; a piece starting with `space` opens a new word."""
    words = []
    new_word = True
    for text, s, e in pieces:
        if not text:
            continue
        if text.startswith(space):
            new_word = True
            text = text.lstrip(space)
            if not text:  # a bare space piece: numbers come as "▁", "5" and must not glue onto the last word
                continue
        if new_word or not words:
            words.append({"w": text, "s": s, "e": e})
            new_word = False
        else:
            words[-1]["w"] += text
            words[-1]["e"] = max(words[-1]["e"], e)
    for w in words:
        w["s"], w["e"] = round(w["s"], 3), round(max(w["e"], w["s"]), 3)
    return words
