"""Sends a daily LLM-generated motivational message to Telegram."""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
TELEGRAM_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"

OPENAI_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a supportive mentor writing a short morning message for someone "
    "learning software engineering. Write one original motivational message. "
    "It must be encouraging, positive, realistic, and written in natural "
    "English. Never be toxic, repetitive, or cheesy, and avoid generic "
    "clichés. Keep it around 80-150 words. Write only the message itself, "
    "with no greeting and no sign-off."
)

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
    openai_api_key: str
    telegram_token: str
    telegram_chat_id: str

    @classmethod
    def from_env(cls) -> "Config":
        required_vars = {
            "OPENAI_API_KEY": "openai_api_key",
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


def generate_motivational_message(api_key: str) -> str:
    topic = random.choice(TOPICS)
    user_prompt = f"Write today's motivational message, focusing especially on {topic}."
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 1.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        OPENAI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


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
        motivational_text = generate_motivational_message(config.openai_api_key)

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
