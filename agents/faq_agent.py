"""
FAQ Agent

Answers customer questions using the static knowledge base.
If the question is outside scope, sets answered=False so the
orchestrator can decide whether to escalate.
"""

from __future__ import annotations

import json

from agents.base import BaseAgent
from knowledge_base import get_full_kb_text, get_source_keys
from models import ConversationMessage, FAQAgentResponse


_SCHEMA = json.dumps(
    {
        "message": "Answer to the customer's question in plain conversational English",
        "sources": "list of knowledge base keys used, e.g. ['pricing', 'supplies']",
        "answered": "boolean, false if the question is outside the knowledge base",
    },
    indent=2,
)


class FAQAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        kb_text = get_full_kb_text()
        source_keys = get_source_keys()

        return f"""
You are the FAQ assistant for a part-time cleaning service in Singapore.

Answer questions using ONLY the knowledge base below.
Do not invent information.

If the question is not covered by the knowledge base:
- set answered=false
- politely tell the customer you'll connect them with the team for more information

Available source keys:
{source_keys}

=== KNOWLEDGE BASE ===
{kb_text}

Keep answers friendly and concise.
Use plain English.
For multi-point answers, use short bullet lists.

Respond ONLY with a valid JSON object.
No preamble.
No markdown.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> FAQAgentResponse:
        return FAQAgentResponse(
            message=raw["message"],
            sources=raw.get("sources", []),
            answered=raw.get("answered", True),
        )

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        **_,
    ) -> FAQAgentResponse:
        raw = self._call_llm(
            user_message=user_message,
            history=history,
        )

        return self.parse_response(raw)