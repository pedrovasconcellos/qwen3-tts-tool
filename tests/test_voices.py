"""Unit tests for speaker and language validation."""

import pytest

from qwen3_tts_tool.voices import (
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    VOICES,
    list_voices,
    validate_language,
    validate_voice,
)


def test_default_voice_is_aiden():
    assert DEFAULT_VOICE == "aiden"
    assert DEFAULT_VOICE in VOICES


def test_there_are_nine_speakers():
    assert len(VOICES) == 9


def test_validate_voice_returns_model_speaker_name():
    assert validate_voice("aiden") == "Aiden"
    assert validate_voice("ryan") == "Ryan"


def test_validate_voice_is_case_insensitive():
    assert validate_voice("AIDEN") == "Aiden"
    assert validate_voice("Uncle_Fu") == "Uncle_Fu"


def test_validate_voice_rejects_unknown():
    with pytest.raises(ValueError):
        validate_voice("af_heart")


def test_validate_language_canonicalizes():
    assert validate_language("english") == "English"
    assert validate_language("AUTO") == "Auto"


def test_validate_language_rejects_unknown():
    with pytest.raises(ValueError):
        validate_language("Klingon")


def test_default_language_supported():
    assert validate_language(DEFAULT_LANGUAGE) == DEFAULT_LANGUAGE


def test_list_voices_filters_english_speakers():
    english = {v["id"] for v in list_voices(language="English")}
    assert english == {"aiden", "ryan"}


def test_list_voices_filters_by_gender():
    males = list_voices(gender="Male")
    assert all(v["gender"] == "Male" for v in males)
    assert {"aiden", "ryan"}.issubset({v["id"] for v in males})
