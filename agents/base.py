"""
Base agent. All agents inherit from this.

Handles:
- Building the messages array from conversation history
- Calling the OpenAI API
- Extracting and JSON-parsing the response
- Logging errors without crashing the app
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from config import MAX_TOKENS, MODEL, client
from models import ConversationMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Subclasses implement:
    - system_prompt property
    - parse_response(raw_json: dict)
    """

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def parse_response(self, raw: dict) -> Any:
        pass

    def _call_llm(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        extra_system: str = "",
    ) -> dict:
        system = self.system_prompt

        if extra_system:
            system = f"{system}\n\n{extra_system}"

        messages: list[dict] = []

        for turn in history or []:
            messages.append(
                {
                    "role": turn.role,
                    "content": turn.content,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        openai_messages = [
            {
                "role": "system",
                "content": system,
            },
            *messages,
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=openai_messages,
                max_completion_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise

        choice = response.choices[0]
        raw_text = choice.message.content

        if not raw_text:
            finish = choice.finish_reason
            refusal = getattr(choice.message, "refusal", None)
            detail = refusal or f"finish_reason={finish}"
            logger.error("Empty model response: %s", detail)
            raise ValueError(
                f"Model returned no content ({detail}). Try rephrasing."
            )

        raw_text = raw_text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Non-JSON response from model:\n%s", raw_text)
            raise ValueError(f"Model returned non-JSON output: {exc}") from exc

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        **kwargs,
    ) -> Any:
        raw = self._call_llm(
            user_message=user_message,
            history=history,
            **kwargs,
        )

        return self.parse_response(raw)