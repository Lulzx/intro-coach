"""One take end to end: decode -> transcribe -> measure -> coach -> save."""
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import analysis, audio, prosody
from .asr import ENGINE, get_engine
from .paths import TAKES


def run(source: str, scenario: str = "interview", target: float | None = None,
        save: bool = True, keep_original: Path | None = None) -> dict:
    t = time.perf_counter()
    wav = audio.load(source)
    duration = len(wav) / audio.RATE
    words = get_engine().transcribe(wav)
    asr_s = time.perf_counter() - t
    P = prosody.analyse(wav, audio.RATE)
    report = analysis.analyse(words, P, duration, scenario, target)
    report.update({
        "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": ENGINE,
        "timing": {"asr_s": round(asr_s, 2), "total_s": round(time.perf_counter() - t, 2)},
    })
    if save:
        d = TAKES / report["id"]
        d.mkdir(parents=True, exist_ok=True)
        audio.save_wav(d / "audio.wav", wav)
        if keep_original:
            shutil.copy(keep_original, d / ("original" + keep_original.suffix))
        (d / "report.json").write_text(json.dumps(report))
    return report


def list_takes() -> list[dict]:
    out = []
    if not TAKES.exists():
        return out
    for d in sorted(TAKES.iterdir(), reverse=True):
        f = d / "report.json"
        if f.exists():
            r = json.loads(f.read_text())
            out.append({k: r.get(k) for k in ("id", "created", "scenario", "overall", "scores", "engine")}
                       | {"duration": r.get("metrics", {}).get("speaking_time"), "wpm": r.get("metrics", {}).get("wpm"),
                          "has_feedback": "ai" in r})
    return out


def load_take(take_id: str) -> dict:
    d = take_dir(take_id)
    return json.loads((d / "report.json").read_text())


def save_take(report: dict) -> None:
    (take_dir(report["id"]) / "report.json").write_text(json.dumps(report))


def take_dir(take_id: str) -> Path:
    d = (TAKES / take_id).resolve()
    if d.parent != TAKES.resolve() or not (d / "report.json").exists():
        raise FileNotFoundError(take_id)
    return d
