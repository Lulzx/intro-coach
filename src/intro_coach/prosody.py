"""Acoustic measurements with Praat (via parselmouth): pitch contour, loudness and silent stretches."""
from dataclasses import dataclass

import numpy as np
import parselmouth

STEP = 0.01  # seconds per analysis frame


@dataclass
class Prosody:
    t: np.ndarray        # frame times
    f0: np.ndarray       # Hz, 0 where unvoiced
    db: np.ndarray       # intensity per frame
    silent: np.ndarray   # bool per frame
    f0_median: float
    floor: float         # pitch search range actually used
    ceiling: float

    def window(self, a: float, b: float) -> np.ndarray:
        return (self.t >= a) & (self.t < b)

    def voiced(self, a: float, b: float) -> np.ndarray:
        f = self.f0[self.window(a, b)]
        return f[f > 0]

    def semitones(self, hz):
        return 12 * np.log2(np.asarray(hz) / self.f0_median)


def analyse(wav: np.ndarray, rate: int = 16000) -> Prosody:
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=rate)

    # Two-pass pitch range (De Looze & Hirst): a wide first pass, then a range fitted to this speaker,
    # which avoids octave jumps for both low and high voices.
    first = snd.to_pitch_ac(time_step=STEP, pitch_floor=60, pitch_ceiling=600).selected_array["frequency"]
    v = first[first > 0]
    if len(v) > 20:
        floor = max(50.0, 0.75 * np.percentile(v, 15))
        ceiling = min(700.0, 1.5 * np.percentile(v, 85))
    else:
        floor, ceiling = 60.0, 600.0
    pitch = snd.to_pitch_ac(time_step=STEP, pitch_floor=floor, pitch_ceiling=ceiling)
    t = pitch.xs()
    f0 = pitch.selected_array["frequency"]

    inten = snd.to_intensity(minimum_pitch=max(floor, 75), time_step=STEP)
    db = np.interp(t, inten.xs(), inten.values[0])

    # Silence: well below this speaker's loud speech, but never below the room's noise floor
    loud, quiet = np.percentile(db, 95), np.percentile(db, 10)
    threshold = max(loud - 25.0, quiet + 4.0)
    silent = (db < threshold) & (f0 == 0)

    voiced = f0[f0 > 0]
    median = float(np.median(voiced)) if len(voiced) else 120.0
    return Prosody(t=t, f0=f0, db=db, silent=silent, f0_median=median, floor=float(floor), ceiling=float(ceiling))


def runs(mask: np.ndarray, t: np.ndarray, min_len: float):
    """[(start, end)] of True stretches in `mask` lasting at least `min_len` seconds."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) * STEP >= min_len:
                out.append((float(t[start]), float(t[i - 1] + STEP)))
            start = None
    if start is not None and (len(mask) - start) * STEP >= min_len:
        out.append((float(t[start]), float(t[-1] + STEP)))
    return out
