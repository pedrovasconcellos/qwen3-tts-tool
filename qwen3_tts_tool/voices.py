"""Voice (speaker) definitions and management for Qwen3-TTS CustomVoice.

Contains speaker metadata, validation, and listing for the 9 premium
speakers supported by the Qwen3-TTS CustomVoice models, plus the list of
supported synthesis languages.
"""

from typing import TypedDict

from qwen3_tts_tool.logging_config import get_logger

logger = get_logger(__name__)


class VoiceInfo(TypedDict):
    """Speaker information structure."""

    id: str
    name: str
    gender: str
    language: str
    description: str


# Default speaker (sunny American male, clear midrange).
DEFAULT_VOICE = "aiden"

# The 9 premium CustomVoice speakers shipped with Qwen3-TTS CustomVoice models.
# The `id` is the lowercase alias used on the CLI; `name` is the speaker name
# passed to the model's generate_custom_voice(speaker=...) call.
VOICES: dict[str, VoiceInfo] = {
    "vivian": {
        "id": "vivian",
        "name": "Vivian",
        "gender": "Female",
        "language": "Chinese",
        "description": "Bright, slightly edgy young female voice.",
    },
    "serena": {
        "id": "serena",
        "name": "Serena",
        "gender": "Female",
        "language": "Chinese",
        "description": "Warm, gentle young female voice.",
    },
    "uncle_fu": {
        "id": "uncle_fu",
        "name": "Uncle_Fu",
        "gender": "Male",
        "language": "Chinese",
        "description": "Seasoned male voice with a low, mellow timbre.",
    },
    "dylan": {
        "id": "dylan",
        "name": "Dylan",
        "gender": "Male",
        "language": "Chinese (Beijing Dialect)",
        "description": "Youthful Beijing male voice with a clear, natural timbre.",
    },
    "eric": {
        "id": "eric",
        "name": "Eric",
        "gender": "Male",
        "language": "Chinese (Sichuan Dialect)",
        "description": "Lively Chengdu male voice with a slightly husky brightness.",
    },
    "ryan": {
        "id": "ryan",
        "name": "Ryan",
        "gender": "Male",
        "language": "English",
        "description": "Dynamic male voice with strong rhythmic drive.",
    },
    "aiden": {
        "id": "aiden",
        "name": "Aiden",
        "gender": "Male",
        "language": "English",
        "description": "Sunny American male voice with a clear midrange.",
    },
    "ono_anna": {
        "id": "ono_anna",
        "name": "Ono_Anna",
        "gender": "Female",
        "language": "Japanese",
        "description": "Playful Japanese female voice with a light, nimble timbre.",
    },
    "sohee": {
        "id": "sohee",
        "name": "Sohee",
        "gender": "Female",
        "language": "Korean",
        "description": "Warm Korean female voice with rich emotion.",
    },
}

# Languages supported by the Qwen3-TTS models. "Auto" enables adaptive
# language detection based on the input text.
SUPPORTED_LANGUAGES: list[str] = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Default synthesis language (this tool primarily produces English audio).
DEFAULT_LANGUAGE = "English"


def get_voice(voice_id: str) -> VoiceInfo | None:
    """Get speaker information by id (case-insensitive).

    Args:
        voice_id: Speaker identifier (e.g., 'aiden')

    Returns:
        VoiceInfo if found, None otherwise
    """
    return VOICES.get(voice_id.lower())


def validate_voice(voice_id: str) -> str:
    """Validate a speaker id and return its canonical model speaker name.

    Args:
        voice_id: Speaker identifier to validate (case-insensitive)

    Returns:
        The model speaker name (e.g., 'Aiden') to pass to generate_custom_voice

    Raises:
        ValueError: If the speaker is unknown
    """
    voice = VOICES.get(voice_id.lower())
    if voice is None:
        available = ", ".join(sorted(VOICES.keys()))
        raise ValueError(
            f"Unknown voice: {voice_id}\n\n"
            f"Available voices: {available}\n\n"
            "Use 'qwen3-tts-tool list-voices' to see all options."
        )
    return voice["name"]


def validate_language(language: str) -> str:
    """Validate a language name (case-insensitive) and return canonical form.

    Args:
        language: Language name (e.g., 'English', 'Auto')

    Returns:
        The canonical language name accepted by the model

    Raises:
        ValueError: If the language is unsupported
    """
    lookup = {lang.lower(): lang for lang in SUPPORTED_LANGUAGES}
    canonical = lookup.get(language.lower())
    if canonical is None:
        available = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unsupported language: {language}\n\nSupported languages: {available}"
        )
    return canonical


def list_voices(
    language: str | None = None,
    gender: str | None = None,
) -> list[VoiceInfo]:
    """List speakers with optional filtering.

    Args:
        language: Filter by native language (substring match, case-insensitive)
        gender: Filter by gender ('Male' or 'Female')

    Returns:
        List of matching VoiceInfo entries, sorted by id
    """
    voices = list(VOICES.values())

    if language:
        lang_lower = language.lower()
        voices = [v for v in voices if lang_lower in v["language"].lower()]

    if gender:
        gender_lower = gender.lower()
        voices = [v for v in voices if v["gender"].lower() == gender_lower]

    return sorted(voices, key=lambda v: v["id"])


def list_languages() -> list[str]:
    """Get the list of supported synthesis languages.

    Returns:
        List of supported languages
    """
    return list(SUPPORTED_LANGUAGES)
