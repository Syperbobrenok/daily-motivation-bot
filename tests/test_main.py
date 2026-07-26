"""Tests for the deterministic parts of main.py.

These tests do not call Gemini or Telegram. The generated motivational
text itself is not deterministic, so it is not tested here.
"""

import pytest

from main import (
    Config,
    ConfigError,
    build_message,
    count_sentences,
    count_words,
    is_valid_motivational_text,
    limit_message_length,
    shorten_to_limits,
)


def test_build_message_has_greeting_and_signoff():
    message = build_message("Keep going, you are doing great.")
    assert message.startswith("🚀 Good morning!")
    assert message.endswith("Have an amazing day! ❤️")


def test_limit_message_length_keeps_short_text_unchanged():
    text = "Keep going, you are doing great."
    assert limit_message_length(text, max_length=300) == text


def test_limit_message_length_shortens_long_text():
    text = "word " * 100
    result = limit_message_length(text, max_length=300)
    assert len(result) <= 300


def test_limit_message_length_collapses_newlines():
    text = "Line one.\nLine two.\nLine three."
    result = limit_message_length(text, max_length=300)
    assert "\n" not in result


def test_count_words_counts_by_whitespace():
    assert count_words("Keep going, you are doing great.") == 6


def test_count_sentences_counts_two_sentences():
    text = "You do not need to solve everything today. Keep moving forward."
    assert count_sentences(text) == 2


def test_count_sentences_counts_three_sentences():
    text = "One thing at a time. Small steps count. You are improving."
    assert count_sentences(text) == 3


def test_is_valid_motivational_text_accepts_two_sentences_within_word_limit():
    text = (
        "You do not need to solve everything today. Make one small "
        "improvement, learn from one mistake, and keep moving."
    )
    assert is_valid_motivational_text(text) is True


def test_is_valid_motivational_text_rejects_one_sentence():
    text = "Just keep going every single day no matter what happens."
    assert is_valid_motivational_text(text) is False


def test_is_valid_motivational_text_rejects_four_sentences():
    text = "One. Two. Three. Four."
    assert is_valid_motivational_text(text) is False


def test_is_valid_motivational_text_rejects_too_many_words():
    text = ("word " * 50).strip() + ". More words go here today."
    assert is_valid_motivational_text(text) is False


def test_shorten_to_limits_keeps_at_most_three_sentences():
    text = "One sentence here. Two sentence here. Three here. Four sentence here."
    result = shorten_to_limits(text)
    assert count_sentences(result) <= 3


def test_shorten_to_limits_keeps_at_most_45_words():
    text = ("word " * 60).strip() + "."
    result = shorten_to_limits(text)
    assert count_words(result) <= 45


def test_build_message_keeps_generated_text_unchanged():
    text = "Bugs are just puzzles waiting to be solved."
    message = build_message(text)
    assert text in message


def test_build_message_handles_multiline_text():
    text = "Line one.\nLine two."
    message = build_message(text)
    assert text in message


def test_config_from_env_reads_all_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    config = Config.from_env()

    assert config.gemini_api_key == "gemini-test-key"
    assert config.telegram_token == "telegram-test"
    assert config.telegram_chat_id == "12345"


def test_config_from_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  gemini-test-key  ")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    config = Config.from_env()

    assert config.gemini_api_key == "gemini-test-key"


def test_config_from_env_missing_variable_raises_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_from_env_missing_variables_are_listed_by_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        Config.from_env()

    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert "TELEGRAM_TOKEN" in str(exc_info.value)
    assert "TELEGRAM_CHAT_ID" in str(exc_info.value)
