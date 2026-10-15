"""Decode any audio/video input (file, URL, HLS playlist, browser webm) to 16 kHz mono float32."""
import shutil
import subprocess

import numpy as np

RATE = 16000


def load(source: str, rate: int = RATE) -> np.ndarray:
    """Decode with ffmpeg so every container the browser or a CDN hands us works the same way."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required: brew install ffmpeg")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto,data",
        "-i", source, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"could not decode audio: {proc.stderr.decode(errors='replace').strip()[:500]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size < rate // 2:
        raise ValueError("recording is shorter than half a second")
    return wav


def save_wav(path, wav: np.ndarray, rate: int = RATE) -> None:
    import soundfile as sf

    sf.write(str(path), wav, rate, subtype="PCM_16")
