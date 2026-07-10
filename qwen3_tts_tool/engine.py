"""TTS engine wrapper for Qwen3-TTS (CustomVoice).

Provides a high-level interface for local text-to-speech synthesis: model
loading with CPU/MPS device selection, audio generation via the qwen-tts
package, trailing-silence padding, optional pitch-preserving speed changes
(via ffmpeg atempo), file output, and speaker playback.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from qwen3_tts_tool.logging_config import get_logger
from qwen3_tts_tool.models import get_model_id

logger = get_logger(__name__)

CHANNELS = 1
DEFAULT_SILENCE_MS = 500  # Default trailing silence in milliseconds


def select_device_and_dtype() -> tuple[str, str]:
    """Select the inference device and dtype.

    Defaults to CPU/float32 (the only guaranteed-correct path on macOS).
    Set QWEN3_TTS_DEVICE=mps to opt into Apple GPU acceleration (experimental).

    Returns:
        Tuple of (device, dtype_name) where dtype_name is 'float32' or 'float16'
    """
    requested = os.environ.get("QWEN3_TTS_DEVICE", "cpu").strip().lower() or "cpu"

    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps", "float16"
            logger.warning("QWEN3_TTS_DEVICE=mps requested but MPS is unavailable; using CPU")
        except Exception:
            logger.warning("Could not verify MPS availability; using CPU")
        return "cpu", "float32"

    if requested not in ("cpu", "mps"):
        logger.warning(f"Unknown QWEN3_TTS_DEVICE={requested!r}; using CPU")

    return "cpu", "float32"


def apply_trailing_silence(
    samples: "np.ndarray", sample_rate: int, silence_ms: int
) -> "np.ndarray":
    """Append trailing silence to an audio buffer.

    Args:
        samples: Mono audio samples
        sample_rate: Sample rate in Hz
        silence_ms: Trailing silence in milliseconds

    Returns:
        Audio samples with silence appended (unchanged if silence_ms <= 0)
    """
    if silence_ms <= 0:
        return samples
    silence_samples = int(sample_rate * silence_ms / 1000)
    if silence_samples <= 0:
        return samples
    pad = np.zeros(silence_samples, dtype=samples.dtype)
    return np.concatenate([samples, pad])


def build_atempo_filter(speed: float) -> str:
    """Build an ffmpeg atempo filter string for the given speed.

    atempo preserves pitch and accepts factors in [0.5, 2.0], which matches
    the CLI's supported --speed range, so a single stage is sufficient.

    Args:
        speed: Speed multiplier (0.5-2.0)

    Returns:
        An ffmpeg -filter:a argument value, e.g. "atempo=0.8"
    """
    return f"atempo={speed:g}"


def _apply_speed_in_place(path: Path, speed: float) -> None:
    """Time-stretch a WAV file in place using ffmpeg atempo (pitch-preserving).

    Args:
        path: Path to the WAV file to modify
        speed: Speed multiplier (0.5-2.0); a value of 1.0 is a no-op

    Raises:
        RuntimeError: If ffmpeg is missing or the conversion fails
    """
    if abs(speed - 1.0) < 1e-6:
        return

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to change --speed but was not found.\n\n"
            "What to do:\n"
            "  Install it (macOS): brew install ffmpeg\n"
            "  Or run with --speed 1.0 to skip time-stretching."
        )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-filter:a",
            build_atempo_filter(speed),
            str(tmp_path),
        ]
        logger.debug(f"Running speed adjustment: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to adjust speed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class Qwen3Engine:
    """Wrapper for the Qwen3-TTS CustomVoice model via the qwen-tts package."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str | None = None
        self._device: str | None = None

    def load(self) -> None:
        """Load the model into memory, downloading weights if necessary."""
        if self._model is not None:
            logger.debug("Model already loaded")
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = get_model_id()
        device, dtype_name = select_device_and_dtype()
        dtype = getattr(torch, dtype_name)

        logger.info(f"Loading Qwen3-TTS model: {model_id}")
        logger.debug(f"Device: {device}, dtype: {dtype_name}, attn: sdpa")

        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self._model_id = model_id
        self._device = device
        logger.info("Qwen3-TTS model loaded successfully")

    def is_loaded(self) -> bool:
        """Check whether the model is loaded.

        Returns:
            True if the model is ready
        """
        return self._model is not None

    def generate(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str | None = None,
    ) -> tuple["np.ndarray", int]:
        """Generate speech audio from text.

        Args:
            text: Input text to synthesize
            speaker: Model speaker name (e.g., 'Aiden')
            language: Language name (e.g., 'English', 'Auto')
            instruct: Optional natural-language style instruction

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        self.load()
        if self._model is None:
            raise RuntimeError("Model failed to load")

        logger.debug(
            f"Generating speech: speaker={speaker}, language={language}, "
            f"chars={len(text)}"
        )
        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or None,
        )
        samples = np.asarray(wavs[0], dtype=np.float32)
        logger.debug(f"Generated {len(samples)} samples at {sample_rate}Hz")
        return samples, int(sample_rate)

    def save_audio(
        self,
        text: str,
        output_path: Path,
        speaker: str,
        language: str,
        speed: float = 1.0,
        silence_ms: int = DEFAULT_SILENCE_MS,
        instruct: str | None = None,
    ) -> int:
        """Generate and save audio to a WAV file.

        Args:
            text: Input text to synthesize
            output_path: Output WAV file path
            speaker: Model speaker name
            language: Language name
            speed: Speed multiplier (0.5-2.0), applied via ffmpeg atempo
            silence_ms: Trailing silence in milliseconds
            instruct: Optional style instruction

        Returns:
            Number of characters processed
        """
        import soundfile as sf

        samples, sample_rate = self.generate(text, speaker, language, instruct)
        samples = apply_trailing_silence(samples, sample_rate, silence_ms)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving audio to: {output_path}")
        sf.write(str(output_path), samples, sample_rate)

        _apply_speed_in_place(output_path, speed)
        return len(text)

    def play_audio(
        self,
        text: str,
        speaker: str,
        language: str,
        speed: float = 1.0,
        silence_ms: int = DEFAULT_SILENCE_MS,
        instruct: str | None = None,
    ) -> int:
        """Generate and play audio through the speakers.

        Args:
            text: Input text to synthesize
            speaker: Model speaker name
            language: Language name
            speed: Speed multiplier (0.5-2.0)
            silence_ms: Trailing silence in milliseconds
            instruct: Optional style instruction

        Returns:
            Number of characters processed
        """
        import sounddevice as sd

        samples, sample_rate = self.generate(text, speaker, language, instruct)
        samples = apply_trailing_silence(samples, sample_rate, silence_ms)

        # Playback speed is applied by resampling the playback rate, which is a
        # simple approximation (it shifts pitch). File output uses the higher
        # quality pitch-preserving ffmpeg path instead.
        play_rate = int(sample_rate * speed)

        logger.info("Playing audio through speakers...")
        logger.debug(f"Audio duration: {len(samples) / sample_rate:.2f}s")
        sd.play(samples, samplerate=play_rate)
        sd.wait()
        logger.info("Playback complete")
        return len(text)


# Global engine instance (singleton pattern for CLI efficiency).
_engine: Qwen3Engine | None = None


def get_engine() -> Qwen3Engine:
    """Get the global engine instance.

    Returns:
        Qwen3Engine instance (created on first use)
    """
    global _engine
    if _engine is None:
        _engine = Qwen3Engine()
    return _engine


def synthesize_to_file(
    text: str,
    output_path: Path,
    speaker: str,
    language: str,
    speed: float = 1.0,
    silence_ms: int = DEFAULT_SILENCE_MS,
    instruct: str | None = None,
) -> int:
    """Synthesize text to a WAV file.

    Returns:
        Number of characters processed
    """
    return get_engine().save_audio(
        text, output_path, speaker, language, speed, silence_ms, instruct
    )


def synthesize_and_play(
    text: str,
    speaker: str,
    language: str,
    speed: float = 1.0,
    silence_ms: int = DEFAULT_SILENCE_MS,
    instruct: str | None = None,
) -> int:
    """Synthesize text and play it through the speakers.

    Returns:
        Number of characters processed
    """
    return get_engine().play_audio(
        text, speaker, language, speed, silence_ms, instruct
    )


def read_from_stdin() -> str:
    """Read text from stdin.

    Returns:
        Text from stdin, stripped of leading/trailing whitespace
    """
    import sys

    logger.debug("Reading from stdin...")
    text = sys.stdin.read().strip()
    logger.debug(f"Read {len(text)} characters from stdin")
    return text
