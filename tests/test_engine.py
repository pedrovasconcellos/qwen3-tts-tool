"""Unit tests for the engine helpers and file synthesis path.

The Qwen3-TTS model is never loaded here: the model is injected as a fake and
`soundfile` is stubbed, so these tests run without downloading any weights.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from qwen3_tts_tool import engine
from qwen3_tts_tool.engine import (
    Qwen3Engine,
    apply_trailing_silence,
    build_atempo_filter,
    select_device_and_dtype,
)


def test_apply_trailing_silence_adds_samples():
    samples = np.ones(1000, dtype=np.float32)
    out = apply_trailing_silence(samples, sample_rate=1000, silence_ms=500)
    assert len(out) == 1500
    assert np.all(out[1000:] == 0)


def test_apply_trailing_silence_zero_is_noop():
    samples = np.ones(1000, dtype=np.float32)
    out = apply_trailing_silence(samples, sample_rate=1000, silence_ms=0)
    assert len(out) == 1000


def test_build_atempo_filter():
    assert build_atempo_filter(0.8) == "atempo=0.8"
    assert build_atempo_filter(1.5) == "atempo=1.5"


def test_select_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("QWEN3_TTS_DEVICE", raising=False)
    device, dtype = select_device_and_dtype()
    assert device == "cpu"
    assert dtype == "float32"


def test_select_device_unknown_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_DEVICE", "cuda")
    device, dtype = select_device_and_dtype()
    assert device == "cpu"
    assert dtype == "float32"


def test_apply_speed_in_place_noop_at_one(tmp_path):
    # A file that does not exist must not be touched when speed is 1.0.
    missing = tmp_path / "missing.wav"
    engine._apply_speed_in_place(missing, 1.0)
    assert not missing.exists()


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, text, language, speaker, instruct):
        self.calls.append((text, language, speaker, instruct))
        return [np.ones(2000, dtype=np.float32)], 24000


def test_save_audio_writes_with_silence(monkeypatch, tmp_path):
    written = {}

    fake_sf = types.ModuleType("soundfile")

    def _write(path, samples, sample_rate):
        written["path"] = path
        written["samples"] = np.asarray(samples)
        written["sample_rate"] = sample_rate

    fake_sf.write = _write
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    eng = Qwen3Engine()
    eng._model = _FakeModel()  # skip load()

    out = tmp_path / "out.wav"
    chars = eng.save_audio(
        "developer",
        out,
        speaker="Aiden",
        language="English",
        speed=1.0,
        silence_ms=1000,
        instruct=None,
    )

    assert chars == len("developer")
    assert written["sample_rate"] == 24000
    # 2000 generated samples + 24000 samples of 1000ms silence at 24kHz.
    assert len(written["samples"]) == 2000 + 24000
    assert eng._model.calls == [("developer", "English", "Aiden", None)]


def test_save_audio_missing_ffmpeg_errors(monkeypatch, tmp_path):
    fake_sf = types.ModuleType("soundfile")
    fake_sf.write = lambda *a, **k: Path(a[0]).write_bytes(b"RIFF")
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    monkeypatch.setattr(engine.shutil, "which", lambda _: None)

    eng = Qwen3Engine()
    eng._model = _FakeModel()

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        eng.save_audio(
            "hi",
            tmp_path / "out.wav",
            speaker="Aiden",
            language="English",
            speed=0.8,
            silence_ms=0,
        )
