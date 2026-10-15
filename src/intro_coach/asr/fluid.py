"""Fluid Speech Mini by FluidVoice (CC BY 4.0), ONNX build: https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-onnx

Features, encoder and greedy TDT loop follow the repo's reference transcribe.py; this version also keeps the
encoder frame each token was emitted on, which gives word timestamps (one frame = 10 ms hop x 8 subsampling).
"""
import json
import os

import numpy as np
import onnxruntime as ort

from ..paths import MODELS
from . import group_pieces

REPO = "FluidVoice-Labs/fluid-speech-mini-onnx"
LOG_ZERO_GUARD = 2.0 ** -24
NORM_EPS = 1e-5


REQUIRED = ["config.json", "tokenizer.json", "mel_filters.npy", "encoder.onnx", "encoder.onnx.data", "decoder.onnx", "joint.onnx"]


def model_dir() -> str:
    path = os.environ.get("FLUID_MODEL_DIR") or str(MODELS / "fluid-speech-mini-onnx")
    if not all(os.path.exists(os.path.join(path, f)) for f in REQUIRED):
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=path, allow_patterns=["*.onnx", "*.onnx.data", "*.json", "*.npy", "LICENSE", "NOTICE"])
    return path


def log_mel(wav, cfg, filters):
    n_fft, hop, win = cfg["n_fft"], cfg["hop_length"], cfg["win_length"]
    x = wav.astype(np.float32)
    if cfg.get("preemphasis"):
        x = np.concatenate([x[:1], x[1:] - np.float32(cfg["preemphasis"]) * x[:-1]])
    window = np.zeros(n_fft, dtype=np.float64)
    left = (n_fft - win) // 2
    n = np.arange(win, dtype=np.float64)
    window[left:left + win] = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (win - 1))).astype(np.float32)
    padded = np.pad(x, n_fft // 2)
    frames = 1 + len(x) // hop
    mel = np.empty((frames, filters.shape[0]), dtype=np.float32)
    offsets = np.arange(n_fft)[None, :]
    for start in range(0, frames, 256):
        idx = offsets + hop * np.arange(start, min(start + 256, frames))[:, None]
        spec = np.fft.rfft(padded[idx].astype(np.float64) * window, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel[start:start + len(idx)] = np.log(power @ filters.T + np.float32(LOG_ZERO_GUARD))
    valid = min(len(x) // hop, frames)
    mean = mel[:valid].sum(axis=0, dtype=np.float32) / np.float32(valid)
    var = ((mel[:valid] - mean) ** 2).sum(axis=0, dtype=np.float32) / np.float32(max(valid - 1, 1))
    out = (mel - mean) / (np.sqrt(var) + np.float32(NORM_EPS))
    out[valid:] = 0.0
    mask = (np.arange(frames) < valid).astype(np.int64)
    return out[None].astype(np.float32), mask[None]


class FluidSpeechMini:
    name = "fluid-mini"

    def __init__(self, path: str | None = None, threads: int = 0):
        path = path or model_dir()
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        self.filters = np.load(os.path.join(path, "mel_filters.npy")).astype(np.float32)
        tok = json.load(open(os.path.join(path, "tokenizer.json"), encoding="utf-8"))
        self.pieces = {i: s for s, i in tok["model"]["vocab"].items()}
        self.skip = {self.cfg["blank_token_id"], self.cfg["pad_token_id"]}
        for a in tok.get("added_tokens", []):
            self.pieces[a["id"]] = a["content"]
            if a.get("special"):
                self.skip.add(a["id"])
        self.space = (tok.get("decoder") or {}).get("replacement", "▁")

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        if threads:
            so.intra_op_num_threads = threads
        small = ort.SessionOptions()
        small.intra_op_num_threads = 1
        cpu = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(os.path.join(path, "encoder.onnx"), so, providers=cpu)
        front = os.path.join(path, "frontend.onnx")
        self.frontend = ort.InferenceSession(front, so, providers=cpu) if os.path.exists(front) else None
        self.decoder = ort.InferenceSession(os.path.join(path, "decoder.onnx"), small, providers=cpu)
        self.joint = ort.InferenceSession(os.path.join(path, "joint.onnx"), small, providers=cpu)
        c = self.cfg
        self.frame_s = c["hop_length"] / c["sampling_rate"] * c["subsampling_stride"] ** c["subsampling_layers"]

    def _subsampled(self, frames):
        c = self.cfg
        pad = (c["subsampling_kernel"] - 1) // 2 * 2 - c["subsampling_kernel"]
        for _ in range(c["subsampling_layers"]):
            frames = int(np.floor((frames + pad) / c["subsampling_stride"] + 1.0))
        return frames

    def _encode(self, feats, mask):
        if self.frontend is None:
            enc, enc_len = self.encoder.run(None, {"input_features": feats, "attention_mask": mask})
            return enc[0], int(enc_len[0])
        c = self.cfg
        factor = c["subsampling_stride"] ** c["subsampling_layers"]
        frames, total, step = feats.shape[1], self._subsampled(feats.shape[1]), c["frontend_chunk_frames"]
        parts = []
        for a in range(0, total, step):
            b = min(a + step, total)
            s = factor * (a - 1) if a else 0
            e = min(frames, factor * (b + 1))
            y = self.frontend.run(None, {"input_features": feats[:, s:e], "attention_mask": mask[:, s:e]})[0]
            skip = 1 if a else 0
            parts.append(y[:, skip:skip + (b - a)])
        hidden = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        valid = self._subsampled(int(mask.sum()))
        enc = self.encoder.run(None, {"hidden_states": hidden, "length": np.array([valid], dtype=np.int64)})[0]
        return enc[0], valid

    def _greedy_tdt(self, enc, valid_len):
        """Greedy token-and-duration decoding; returns [(token_id, frame, duration_frames)]."""
        c = self.cfg
        vocab, blank, eos = c["vocab_size"], c["blank_token_id"], c.get("eos_token_id")
        durations = c["durations"]
        state = np.zeros((c["decoder_layers"], 1, c["decoder_hidden_size"]), dtype=np.float32)
        dec_out, h, cc = self.decoder.run(None, {"token": np.array([[blank]], dtype=np.int64), "h_in": state, "c_in": state})
        frames = enc.shape[0]
        budget = c["max_symbols_per_step"] * frames - 1
        out, t, steps = [], 0, 0
        while True:
            logits = self.joint.run(None, {"encoder_frame": enc[min(t, frames - 1)][None], "decoder_out": dec_out})[0][0]
            token = int(logits[:vocab].argmax())
            step = durations[int(logits[vocab:].argmax())]
            if token == blank and step == 0:
                step = 1
            if token != blank:
                out.append((token, t, step))
            t += step
            steps += 1
            if t >= valid_len or steps >= budget or token == eos:
                return out
            if token != blank:
                dec_out, h, cc = self.decoder.run(None, {"token": np.array([[token]], dtype=np.int64), "h_in": h, "c_in": cc})

    def _pass(self, wav, offset):
        feats, mask = log_mel(wav, self.cfg, self.filters)
        enc, enc_len = self._encode(feats, mask)
        pieces = []
        for token, frame, dur in self._greedy_tdt(enc, enc_len):
            if token in self.skip:
                continue
            s = offset + frame * self.frame_s
            pieces.append((self.pieces.get(token, ""), s, s + max(dur, 1) * self.frame_s))
        return pieces

    def _split(self, wav, target=30.0, search=10.0):
        """Cut long audio at the quietest point near every `target` seconds."""
        rate = self.cfg["sampling_rate"]
        hop = rate // 50
        energy = np.sqrt(np.add.reduceat(wav.astype(np.float64) ** 2, np.arange(0, len(wav), hop)) / hop)
        smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
        cuts, start = [], 0
        while len(wav) - start > self.cfg["max_single_pass_seconds"] * rate:
            lo = (start + int((target - search / 2) * rate)) // hop
            hi = (start + int((target + search / 2) * rate)) // hop
            cut = (lo + int(smooth[lo:hi].argmin())) * hop
            cuts.append((start, cut))
            start = cut
        cuts.append((start, len(wav)))
        return cuts

    def transcribe(self, wav: np.ndarray) -> list[dict]:
        rate = self.cfg["sampling_rate"]
        pieces = []
        for a, b in self._split(wav):
            pieces += self._pass(np.ascontiguousarray(wav[a:b]), a / rate)
        return group_pieces(pieces, self.space)
