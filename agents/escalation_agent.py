"""
Escalation Agent

Handles two cases:

1. Emergency bookings, same-day or next-day -> CRITICAL urgency
2. Complaints, explicit human requests, unanswerable FAQs -> HIGH urgency

Produces an EscalationRequest that:
- Tells the customer what's happening
- Gives the salesperson a conversation summary
- Specifies notification channels
"""

from __future__ import annotations

import json
from datetime import date

from agents.base import BaseAgent
from config import SALESPERSON_EMAIL, SALESPERSON_WHATSAPP
from knowledge_base import CONFIG as _CONFIG
from models import (
    BookingDetails,
    ConversationMessage,
    CustomerInfo,
    EscalationRequest,
    Intent,
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
            "Include what the customer wants, any details already collected, and sentiment."
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

- The booking is for TODAY or TOMORROW, meaning urgency=critical
- The customer has a complaint or is upset, meaning urgency=high
- The customer asked to speak to a human, meaning urgency=high
- The FAQ agent could not answer the question, meaning urgency=routine

Your job:

1. Write a calm, empathetic message to the customer explaining that a human team member will contact them.
2. Do NOT make promises about timing that you cannot keep.
3. Write a clear internal summary for the salesperson so they can act immediately.
4. Set the correct urgency level.

Salesperson contacts:

- WhatsApp: {SALESPERSON_WHATSAPP}
- Email: {SALESPERSON_EMAIL}

For emergency bookings, the message_to_user should convey urgency and reassurance.

Example:
"I see you need a cleaner very soon. I'm flagging this to our team right now. Someone will WhatsApp you shortly."

SERVICE HOURS RULE:
Always check the BUSINESS RULES below before escalating.
If the requested time is outside the stated service hours:
- Do NOT escalate.
- Politely explain the service hours.
- Ask the customer to choose a time within the service window.
- Set urgency=routine.

DATE MISMATCH RULE:

If you are told that date_seems_wrong=True, the customer gave a date that is logically inconsistent with their request.

Case A: Booking with a PAST date
- The customer asked to book a cleaning but gave a date that has already passed.
- Your message_to_user MUST politely tell them a booking cannot be made retroactively.
- Ask for a valid future date.
- Example: "It looks like that date has already passed. Could you let me know which future date you'd like the cleaning?"

Case B: Complaint or feedback with a FUTURE date
- The customer is complaining about an incident that supposedly has not happened yet.
- Your message_to_user MUST gently flag that the date appears to be in the future.
- Ask them to confirm the correct past date.
- Example: "I noticed the date you mentioned appears to be in the future. Could you double-check and share the actual date of your cleaning session?"

In both cases:
- Do NOT record the wrong date as confirmed in conversation_summary.
- Flag it as unconfirmed.

{_CONFIG.rules_for_agents()}

Respond ONLY with a valid JSON object.
No preamble.
No markdown.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> EscalationRequest:
        customer_raw = raw.pop("customer", {}) or {}

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
        extra_parts: list[str] = []

        if classification:
            extra_parts.append(
                "Intent classification:\n"
                f"intent={classification.intent.value}\n"
                f"urgency={classification.urgency.value}\n"
                f"sentiment={classification.sentiment.value}\n"
                f"is_emergency={classification.is_emergency}\n"
                f"date_seems_wrong={classification.date_seems_wrong}\n"
                f"detected_date={classification.detected_date}"
            )

            if classification.date_seems_wrong:
                extra_parts.append(
                    self._build_date_mismatch_note(classification)
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

    def _build_date_mismatch_note(
        self,
        classification: IntentClassification,
    ) -> str:
        detected_date = classification.detected_date

        if detected_date is None:
            return (
                "NOTE: date_seems_wrong=True, but no valid detected_date was parsed. "
                "Ask the customer to clarify the date."
            )

        delta = (detected_date - date.today()).days

        if delta < 0 and classification.intent in (
            Intent.BOOKING_ENQUIRY,
            Intent.EMERGENCY_BOOKING,
        ):
            return (
                "NOTE: date_seems_wrong=True. "
                f"The customer gave a past date ({detected_date}) for a booking. "
                "A booking cannot be made in the past. Ask for a valid future date."
            )

        if delta > 0 and classification.intent in (
            Intent.COMPLAINT,
            Intent.FEEDBACK,
        ):
            return (
                "NOTE: date_seems_wrong=True. "
                f"The customer gave a future date ({detected_date}) for a complaint or feedback. "
                "The incident may not have happened yet. Ask for the correct past date."
            )

        return (
            "NOTE: date_seems_wrong=True. "
            f"The detected date ({detected_date}) seems inconsistent. Ask the customer to clarify."
        )