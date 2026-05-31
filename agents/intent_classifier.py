"""
Intent Classifier Agent

LLM-based classifier that returns IntentClassification.
Detects intent, sentiment, urgency, and whether the booking is an emergency
same-day or next-day, even when the date is expressed naturally.
"""

from __future__ import annotations

import json
from datetime import date

from agents.base import BaseAgent
from models import ConversationMessage, IntentClassification


_SCHEMA = json.dumps(
    {
        "intent": (
            "one of: booking_enquiry | emergency_booking | faq | complaint | "
            "feedback | escalation | out_of_scope"
        ),
        "sentiment": "one of: positive | neutral | negative | urgent",
        "urgency": "one of: routine | high | critical",
        "confidence": "float 0.0-1.0",
        "reasoning": "one sentence explaining the classification",
        "is_emergency": "boolean, true if booking date is today or tomorrow",
        "detected_date": "ISO date string YYYY-MM-DD if user mentioned a date, else null",
    },
    indent=2,
)


_INTENT_DESCRIPTIONS = """
Intent definitions:

- booking_enquiry   -> user wants to schedule a cleaning session, not urgent
- emergency_booking -> user wants a session today or tomorrow
- faq               -> user has a question about the service, pricing, supplies, etc.
- complaint         -> user is unhappy about a past session or service quality
- feedback          -> user is giving post-service feedback
- escalation        -> user explicitly asks to speak to a human, or situation is unclear/dangerous
- out_of_scope      -> unrelated to the cleaning business
"""


class IntentClassifier(BaseAgent):
    @property
    def system_prompt(self) -> str:
        today = date.today().isoformat()

        return f"""
You are the intent and sentiment classifier for a part-time cleaning service chatbot.

Today's date is {today}.

{_INTENT_DESCRIPTIONS}

Sentiment:

- positive -> satisfied, grateful, happy
- neutral  -> factual, no strong emotion
- negative -> frustrated, upset, angry
- urgent   -> explicit time pressure, distress, emergencies

Urgency:

- routine  -> normal enquiry or FAQ
- high     -> complaint or strong negative sentiment
- critical -> emergency booking, safety concern, or explicit distress

IMPORTANT:
If the user mentions wanting a booking today, tonight, tomorrow, tomorrow morning,
tmr, tmrw, ASAP, or any phrase that resolves to today's date ({today}) or tomorrow,
set:

- is_emergency=true
- intent=emergency_booking
- urgency=critical

Respond ONLY with a valid JSON object matching this schema.
No preamble.
No markdown.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> IntentClassification:
        return IntentClassification(**raw)

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        **_,
    ) -> IntentClassification:
        raw = self._call_llm(
            user_message=user_message,
            history=history,
        )

        return self.parse_response(raw)