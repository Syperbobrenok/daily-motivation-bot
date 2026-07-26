"""Tests for the deterministic parts of main.py.

These tests do not call OpenAI or Telegram. The generated motivational
text itself is not deterministic, so it is not tested here.
"""

import pytest

from main import Config, ConfigError, build_message


def test_build_message_has_greeting_and_signoff():
    message = build_message("Keep going, you are doing great.")
    assert message.startswith("🚀 Good morning!")
    assert message.endswith("Have an amazing day! ❤️")


def test_build_message_keeps_generated_text_unchanged():
    text = "Bugs are just puzzles waiting to be solved."
    message = build_message(text)
    assert text in message


def test_build_message_handles_multiline_text():
    text = "Line one.\nLine two."
    message = build_message(text)
    assert text in message


def test_config_from_env_reads_all_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    config = Config.from_env()

    assert config.openai_api_key == "sk-test"
    assert config.telegram_token == "telegram-test"
    assert config.telegram_chat_id == "12345"


def test_config_from_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test  ")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    config = Config.from_env()

    assert config.openai_api_key == "sk-test"


def test_config_from_env_missing_variable_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_from_env_missing_variables_are_listed_by_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        Config.from_env()

    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "TELEGRAM_TOKEN" in str(exc_info.value)
    assert "TELEGRAM_CHAT_ID" in str(exc_info.value)
