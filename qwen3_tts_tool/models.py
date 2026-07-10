"""Model management for Qwen3-TTS.

Resolves the model/tokenizer ids, pre-downloads weights into the Hugging
Face cache, and reports installation status. The default model is the
lightweight 0.6B CustomVoice checkpoint (best trade-off for macOS CPU).
"""

import os
from pathlib import Path

from qwen3_tts_tool.logging_config import get_logger

logger = get_logger(__name__)

# The speech tokenizer required by all Qwen3-TTS checkpoints.
TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Default CustomVoice model. The 0.6B variant is the lightest/fastest on CPU.
# Override with the QWEN3_TTS_MODEL env var (e.g. the 1.7B variant for higher
# quality at the cost of speed and memory).
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Marker directory so we can record that init has completed at least once.
CONFIG_DIR = Path.home() / ".qwen3-tts"


def get_model_id() -> str:
    """Get the CustomVoice model id, honoring the QWEN3_TTS_MODEL override.

    Returns:
        The Hugging Face model id to load
    """
    return os.environ.get("QWEN3_TTS_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def get_config_dir() -> Path:
    """Get (and create) the tool's config directory.

    Returns:
        Path to ~/.qwen3-tts/
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def _snapshot_dir(repo_id: str) -> Path | None:
    """Return the local snapshot path for a repo if it is already cached.

    Args:
        repo_id: Hugging Face repo id

    Returns:
        Path to the cached snapshot, or None if not cached
    """
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, local_files_only=True)
        return Path(path)
    except Exception:
        return None


def models_exist() -> bool:
    """Check whether both the tokenizer and the model are cached locally.

    Returns:
        True if both snapshots are available offline
    """
    return _snapshot_dir(TOKENIZER_ID) is not None and _snapshot_dir(get_model_id()) is not None


def download_models(force: bool = False) -> tuple[Path, Path]:
    """Download the tokenizer and model snapshots into the Hugging Face cache.

    Args:
        force: If True, re-fetch even when files already exist

    Returns:
        Tuple of (tokenizer_path, model_path)

    Raises:
        RuntimeError: If a download fails
    """
    from huggingface_hub import snapshot_download

    model_id = get_model_id()
    try:
        logger.info(f"Ensuring tokenizer is available: {TOKENIZER_ID}")
        tokenizer_path = Path(
            snapshot_download(repo_id=TOKENIZER_ID, force_download=force)
        )
        logger.info(f"Ensuring model is available: {model_id}")
        model_path = Path(snapshot_download(repo_id=model_id, force_download=force))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Qwen3-TTS models: {e}\n\n"
            "What to do:\n"
            "  1. Check your internet connection\n"
            "  2. Try again with: qwen3-tts-tool init --force\n"
            "  3. Manually download from Hugging Face:\n"
            f"     https://huggingface.co/{TOKENIZER_ID}\n"
            f"     https://huggingface.co/{model_id}"
        ) from e

    # Record that initialization has completed.
    (get_config_dir() / "initialized").write_text(f"{model_id}\n", encoding="utf-8")

    return tokenizer_path, model_path


def get_model_info() -> dict[str, str | bool]:
    """Get information about the configured models and cache status.

    Returns:
        Dictionary with model ids, cache paths, and readiness status
    """
    model_id = get_model_id()
    tokenizer_path = _snapshot_dir(TOKENIZER_ID)
    model_path = _snapshot_dir(model_id)

    info: dict[str, str | bool] = {
        "tokenizer_id": TOKENIZER_ID,
        "model_id": model_id,
        "config_dir": str(CONFIG_DIR),
        "tokenizer_ready": tokenizer_path is not None,
        "model_ready": model_path is not None,
        "ready": tokenizer_path is not None and model_path is not None,
    }
    if tokenizer_path is not None:
        info["tokenizer_path"] = str(tokenizer_path)
    if model_path is not None:
        info["model_path"] = str(model_path)

    return info
