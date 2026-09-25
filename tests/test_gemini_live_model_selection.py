"""Regression tests for the current Gemini Live model selection."""

from gemini_live.config import (
    GeminiLiveConfig,
    model_supports_thinking,
    model_uses_realtime_text,
)


def test_default_live_model_is_gemini_3_8_live():
    config = GeminiLiveConfig()

    assert config.model == "gemini-3.8-live"


def test_gemini_3_8_live_uses_live_realtime_text_path():
    assert model_uses_realtime_text("gemini-3.8-live") is True


def test_standard_gemini_3_8_live_does_not_receive_thinking_config():
    assert model_supports_thinking("gemini-3.8-live") is False


def test_gemini_3_8_live_extended_thinking_supports_thinking_config():
    assert model_supports_thinking("gemini-3.8-live-extended-thinking") is True
