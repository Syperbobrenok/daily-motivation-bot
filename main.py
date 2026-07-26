"""Sends a daily LLM-generated motivational message to Telegram."""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TELEGRAM_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"

GEMINI_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 30
MAX_MESSAGE_LENGTH = 300
MAX_WORDS = 45
MIN_SENTENCES = 2
MAX_SENTENCES = 3

SYSTEM_PROMPT = (
    "You are a supportive mentor writing a short daily message for someone "
    "learning software engineering. Be encouraging, positive, and realistic. "
    "Never be toxic, repetitive, or cheesy, and avoid generic clichés. "
    "Write exactly 2 or 3 short sentences, with a maximum of 45 words in "
    "total. Do not write long compound sentences, lists, or headings. "
    "Return only the motivational text itself: no introduction, no "
    "conclusion, no greeting, no sign-off, and no markdown."
)

RETRY_INSTRUCTION = (
    "Your previous response did not follow the required format. Rewrite it "
    "as exactly 2 or 3 short sentences, with a maximum of 45 words in "
    "total. Do not include a greeting, sign-off, lists, headings, or "
    "markdown. Return only the motivational text."
)

SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+")

TOPICS = [
    "learning programming",
    "staying consistent",
    "discipline",
    "overcoming a difficult bug",
    "technical interviews",
    "building confidence",
    "continuous improvement",
    "becoming a better engineer",
]


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Config:
    gemini_api_key: str
    telegram_token: str
    telegram_chat_id: str

    @classmethod
    def from_env(cls) -> "Config":
        required_vars = {
            "GEMINI_API_KEY": "gemini_api_key",
            "TELEGRAM_TOKEN": "telegram_token",
            "TELEGRAM_CHAT_ID": "telegram_chat_id",
        }
        values = {}
        missing = []
        for env_name, field_name in required_vars.items():
            value = (os.getenv(env_name) or "").strip()
            if not value:
                missing.append(env_name)
            values[field_name] = value

        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(**values)


def limit_message_length(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    return textwrap.shorten(text, width=max_length, placeholder="...")


def count_words(text: str) -> int:
    return len(text.split())


def split_into_sentences(text: str) -> list[str]:
    sentences = SENTENCE_END_PATTERN.split(text.strip())
    return [sentence for sentence in sentences if sentence]


def count_sentences(text: str) -> int:
    return len(split_into_sentences(text))


def is_valid_motivational_text(text: str) -> bool:
    sentence_count = count_sentences(text)
    word_count = count_words(text)
    return MIN_SENTENCES <= sentence_count <= MAX_SENTENCES and word_count <= MAX_WORDS


def shorten_to_limits(text: str) -> str:
    sentences = split_into_sentences(text)[:MAX_SENTENCES]
    shortened = " ".join(sentences)

    words = shortened.split()
    if len(words) > MAX_WORDS:
        shortened = " ".join(words[:MAX_WORDS]) + "."

    return shortened


def request_motivational_text(api_key: str, user_prompt: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 1.0},
    }
    headers = {"x-goog-api-key": api_key}
    url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent"

    response = requests.post(
        url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_motivational_message(api_key: str) -> str:
    topic = random.choice(TOPICS)
    user_prompt = f"Write today's motivational message, focusing especially on {topic}."

    text = request_motivational_text(api_key, user_prompt)

    if not is_valid_motivational_text(text):
        logger.info("Response did not match the required format, retrying")
        retry_prompt = f"{user_prompt} {RETRY_INSTRUCTION}"
        text = request_motivational_text(api_key, retry_prompt)

    if not is_valid_motivational_text(text):
        logger.info("Retry still did not match the required format, shortening it")
        text = shorten_to_limits(text)

    return limit_message_length(text)


def build_message(motivational_text: str) -> str:
    return f"🚀 Good morning!\n\n{motivational_text}\n\nHave an amazing day! ❤️"


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = TELEGRAM_URL_TEMPLATE.format(token=token)
    payload = {"chat_id": chat_id, "text": text}
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()


def main() -> None:
    try:
        config = Config.from_env()

        logger.info("Generating motivational message")
        motivational_text = generate_motivational_message(config.gemini_api_key)

        message = build_message(motivational_text)
        logger.info("Sending Telegram message")
        send_telegram_message(config.telegram_token, config.telegram_chat_id, message)

        logger.info("Motivational message sent successfully")
    except ConfigError as error:
        logger.error("Configuration error: %s", error)
        sys.exit(1)
    except requests.RequestException as error:
        logger.error("Request to an external API failed: %s", error)
        sys.exit(1)
    except (KeyError, IndexError, ValueError) as error:
        logger.error("Unexpected data received from an external API: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
