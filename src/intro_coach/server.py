"""Local web app: record in the browser, get coached. Everything runs on this machine except optional content feedback."""
import asyncio
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ai, asr, coach, nlp, settings
from .analysis import SCENARIOS

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Intro Coach")
# one take at a time: the models are shared and the machine is the bottleneck anyway
lock = asyncio.Lock()


@app.on_event("startup")
def warm_up():
    # load the speech and language models in the background so the first take isn't slowed by it
    threading.Thread(target=lambda: (asr.get_engine(), nlp.warm_up()), daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {
        "scenarios": SCENARIOS,
        "engine": asr.ENGINE_LABEL,
        "ai": ai.status(),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), scenario: str = Form("interview"),
                  target: float = Form(0)):
    suffix = Path(file.filename or "take.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        async with lock:
            return await run_in_threadpool(coach.run, str(path), scenario, target or None)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(422, str(e))
    finally:
        path.unlink(missing_ok=True)


@app.get("/api/takes")
def takes():
    return coach.list_takes()


def _take(take_id: str) -> dict:
    try:
        return coach.load_take(take_id)
    except FileNotFoundError:
        raise HTTPException(404, "no such take")


@app.get("/api/takes/{take_id}")
def take(take_id: str):
    return _take(take_id)


@app.get("/api/takes/{take_id}/audio")
def take_audio(take_id: str):
    _take(take_id)
    return FileResponse(coach.take_dir(take_id) / "audio.wav", media_type="audio/wav")


@app.delete("/api/takes/{take_id}")
def delete_take(take_id: str):
    import shutil

    _take(take_id)
    shutil.rmtree(coach.take_dir(take_id))
    return {"deleted": take_id}


class FeedbackIn(BaseModel):
    notes: str = ""


@app.post("/api/takes/{take_id}/feedback")
async def take_feedback(take_id: str, body: FeedbackIn):
    st = ai.status()
    if not st["ready"]:
        raise HTTPException(501, st["hint"])
    report = _take(take_id)
    try:
        report["ai"] = await run_in_threadpool(ai.feedback, report, body.notes)
    except Exception as e:  # surface auth/network/refusal problems to the page as a readable message
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    coach.save_take(report)
    return report["ai"]


class SettingsIn(BaseModel):
    openrouter_api_key: str


def _settings_view() -> dict:
    key, source = settings.openrouter_key()
    return {"openrouter": {"set": bool(key), "source": source, "masked": settings.mask(key) if key else None,
                           "model": ai.OPENROUTER_MODEL}}


@app.get("/api/settings")
def get_settings():
    return _settings_view()


@app.put("/api/settings")
async def put_settings(body: SettingsIn):
    key = body.openrouter_api_key.strip()
    if not key:
        raise HTTPException(400, "Enter a key")
    problem = await run_in_threadpool(ai.check_openrouter_key, key)
    if problem:
        raise HTTPException(400, problem)
    settings.set_openrouter_key(key)
    return _settings_view()


@app.delete("/api/settings/openrouter")
def delete_openrouter_key():
    settings.set_openrouter_key(None)
    return _settings_view()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
