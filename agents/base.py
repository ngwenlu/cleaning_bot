“””
Base agent. All agents inherit from this.

Handles:

- Building the messages array from conversation history
- Calling the Claude API
- Extracting and JSON-parsing the response
- Logging errors without crashing the app
  “””

from **future** import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from config import MODEL, MAX_TOKENS, client
from models import ConversationMessage

logger = logging.getLogger(**name**)

class BaseAgent(ABC):
“””
Abstract base for all agents.

```
Subclasses implement:
  - system_prompt (property) → str
  - parse_response(raw_json: dict) → the appropriate Pydantic model
"""

@property
@abstractmethod
def system_prompt(self) -> str: ...

@abstractmethod
def parse_response(self, raw: dict) -> Any: ...

# ── Core call ──────────────────────────────────────────────────────────

def _call_llm(
    self,
    user_message: str,
    history: list[ConversationMessage] | None = None,
    extra_system: str = "",
) -> dict:
    """
    Call the Claude API and return the parsed JSON dict.

    Args:
        user_message:  The latest user message to send.
        history:       Prior conversation turns (oldest first).
        extra_system:  Optional extra instructions appended to the system prompt.

    Returns:
        Parsed JSON dict from the model's response.

    Raises:
        ValueError: if the model returns non-JSON or the call fails.
    """
    system = self.system_prompt
    if extra_system:
        system = f"{system}\n\n{extra_system}"

    # Build messages from history + new user message
    messages: list[dict] = []
    for turn in (history or []):
        messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": user_message})

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

    # Strip markdown fences if the model wraps output
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        raw_text = raw_text.rsplit("```", 1)[0]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Non-JSON response from model:\n%s", raw_text)
        raise ValueError(f"Model returned non-JSON output: {exc}") from exc

# ── Public interface ───────────────────────────────────────────────────

def run(
    self,
    user_message: str,
    history: list[ConversationMessage] | None = None,
    **kwargs,
) -> Any:
    """
    Main entry point. Calls LLM and parses into the agent's output model.
    Subclasses can override kwargs to pass extra context (e.g. BookingDetails).
    """
    raw = self._call_llm(user_message, history, **kwargs)
    return self.parse_response(raw)
```