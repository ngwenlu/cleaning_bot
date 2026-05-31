"""
Base agent. All agents inherit from this.

Handles:
- Building the messages array from conversation history
- Calling the Claude API
- Extracting and JSON-parsing the response
- Logging errors without crashing the app
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from config import MODEL, MAX_TOKENS, client
from models import ConversationMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Subclasses implement:
      - system_prompt (property) -> str
      - parse_response(raw_json: dict) -> appropriate Pydantic model
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
        """
        Call the Claude API and return the parsed JSON dict.

        Args:
            user_message: Latest user message.
            history: Previous conversation turns.
            extra_system: Additional instructions appended to system prompt.

        Returns:
            Parsed JSON response.

        Raises:
            ValueError if response is not valid JSON.
        """

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

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
            )

        except Exception as exc:
            logger.error("Claude API call failed: %s", exc)
            raise

        raw_text = response.content[0].text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0]

        try:
            return json.loads(raw_text)

        except json.JSONDecodeError as exc:
            logger.error(
                "Non-JSON response from model:\n%s",
                raw_text,
            )

            raise ValueError(
                f"Model returned non-JSON output: {exc}"
            ) from exc

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        **kwargs,
    ) -> Any:
        """
        Main entry point.

        Calls the LLM and parses the result into the
        agent's output model.
        """

        raw = self._call_llm(
            user_message=user_message,
            history=history,
            **kwargs,
        )

        return self.parse_response(raw)