"""
Escalation Agent

Handles two cases:

1. Emergency bookings (same/next day) -> CRITICAL urgency
2. Complaints, explicit human requests, unanswerable FAQs -> HIGH urgency

Produces an EscalationRequest that:

- Tells the customer what's happening
- Gives the salesperson a conversation summary
- Specifies notification channels
"""

from __future__ import annotations

import json

from agents.base import BaseAgent
from config import SALESPERSON_EMAIL, SALESPERSON_WHATSAPP
from models import (
    BookingDetails,
    ConversationMessage,
    CustomerInfo,
    EscalationRequest,
    IntentClassification,
    UrgencyLevel,
)


_SCHEMA = json.dumps(
    {
        "urgency": "one of: routine | high | critical",
        "reason": "one sentence explaining why escalation was triggered",
        "customer": {
            "name": "string or null",
            "phone": "string or null",
            "email": "string or null",
        },
        "conversation_summary": (
            "3-5 sentence summary of the conversation for the salesperson. "
            "Include what the customer wants, any details already collected, "
            "and customer sentiment."
        ),
        "message_to_user": (
            "What to say to the customer right now. "
            "For emergencies: reassure and set expectations. "
            "For complaints: empathise and commit to follow-up."
        ),
        "notify_via": "list, e.g. ['whatsapp', 'email']",
    },
    indent=2,
)


class EscalationAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return f"""
You are the escalation handler for a part-time cleaning service chatbot.

You are invoked when:

- The booking is for TODAY or TOMORROW -> urgency=critical
- The customer has a complaint or is upset -> urgency=high
- The customer explicitly asks to speak to a human -> urgency=high
- The FAQ agent could not answer the question -> urgency=routine

Your responsibilities:

1. Write a calm, empathetic message to the customer explaining that a human team member will contact them shortly.
2. Do NOT make promises about timing that you cannot guarantee.
3. Write a clear internal summary for the salesperson.
4. Set the correct urgency level.

Salesperson contacts:

- WhatsApp: {SALESPERSON_WHATSAPP}
- Email: {SALESPERSON_EMAIL}

For emergency bookings, the message_to_user should convey urgency and reassurance.

Example:

"I see you need a cleaner very soon. I'm flagging this to our team right now. Someone will WhatsApp you shortly."

Respond ONLY with a valid JSON object.
No markdown.
No explanations.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> EscalationRequest:
        customer_raw = raw.pop("customer", {})

        customer = CustomerInfo(
            **{
                key: value
                for key, value in customer_raw.items()
                if value is not None
            }
        )

        return EscalationRequest(
            urgency=UrgencyLevel(raw["urgency"]),
            reason=raw["reason"],
            customer=customer,
            conversation_summary=raw["conversation_summary"],
            message_to_user=raw["message_to_user"],
            notify_via=raw.get("notify_via", ["whatsapp"]),
        )

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        classification: IntentClassification | None = None,
        booking_details: BookingDetails | None = None,
        **_,
    ) -> EscalationRequest:
        """
        Args:
            classification:
                IntentClassification from the orchestrator.

            booking_details:
                Booking details collected so far.
        """

        extra_parts: list[str] = []

        if classification:
            extra_parts.append(
                f"Intent classification:\n"
                f"intent={classification.intent}\n"
                f"urgency={classification.urgency}\n"
                f"sentiment={classification.sentiment}\n"
                f"is_emergency={classification.is_emergency}"
            )

        if booking_details:
            filled = {
                key: value
                for key, value in booking_details.model_dump().items()
                if value is not None and value != {}
            }

            extra_parts.append(
                "Booking details collected so far:\n"
                f"{json.dumps(filled, indent=2, default=str)}"
            )

        extra = "\n\n".join(extra_parts)

        raw = self._call_llm(
            user_message=user_message,
            history=history,
            extra_system=extra,
        )

        result = self.parse_response(raw)

        if booking_details:
            result.booking_details = booking_details

        return result