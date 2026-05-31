“””
Escalation Agent

Handles two cases:

1. Emergency bookings (same/next day) -> CRITICAL urgency
1. Complaints, explicit human requests, unanswerable FAQs -> HIGH urgency

Produces an EscalationRequest that:

- Tells the customer what’s happening
- Gives the salesperson a conversation summary
- Specifies notification channels
  “””

from **future** import annotations

import json

from agents.base import BaseAgent
from config import SALESPERSON_WHATSAPP, SALESPERSON_EMAIL
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
“urgency”: “one of: routine | high | critical”,
“reason”: “one sentence – why escalation was triggered”,
“customer”: {
“name”:  “string or null”,
“phone”: “string or null”,
“email”: “string or null”,
},
“conversation_summary”: (
“3-5 sentence summary of the conversation for the salesperson. “
“Include: what the customer wants, any details already collected, sentiment.”
),
“message_to_user”: (
“What to say to the customer right now. “
“For emergencies: reassure and set expectations. “
“For complaints: empathise and commit to follow-up.”
),
“notify_via”: “list – e.g. [‘whatsapp’, ‘email’]”,
},
indent=2,
)

class EscalationAgent(BaseAgent):

```
@property
def system_prompt(self) -> str:
    return f"""You are the escalation handler for a part-time cleaning service chatbot.
```

You are invoked when:

- The booking is for TODAY or TOMORROW (emergency) -> urgency=critical
- The customer has a complaint or is upset -> urgency=high
- The customer asked to speak to a human -> urgency=high
- The FAQ agent could not answer the question -> urgency=routine

Your job:

1. Write a calm, empathetic message to the customer explaining that a human team member
   will contact them shortly. Do NOT make promises about timing you can’t keep.
1. Write a clear internal summary for the salesperson so they can act immediately.
1. Set the correct urgency level.

Salesperson contacts:

- WhatsApp: {SALESPERSON_WHATSAPP}
- Email: {SALESPERSON_EMAIL}

For emergency bookings, the message_to_user should convey urgency and reassurance:
“I see you need a cleaner very soon! I’m flagging this to our team right now.
Someone will WhatsApp you within the next few minutes.”

DATE MISMATCH RULE (important):
If you are told that date_seems_wrong=True, the customer gave a date that is logically
inconsistent with their request. There are two cases:

Case A – Booking with a PAST date:
The customer asked to book a cleaning but gave a date that has already passed.
Your message_to_user MUST politely tell them a booking cannot be made retroactively
and ask for a valid future date.
Example: “It looks like that date has already passed – could you let me know
which future date you’d like the cleaning?”

Case B – Complaint/feedback with a FUTURE date:
The customer is complaining about an incident that supposedly hasn’t happened yet.
Your message_to_user MUST gently flag that the date appears to be in the future
and ask them to confirm the correct past date.
Example: “I noticed the date you mentioned (22/12/2028) appears to be in the future –
could you double-check and share the actual date of your cleaning session?”

In both cases: do NOT record the wrong date in conversation_summary – flag it as unconfirmed.

Respond ONLY with a valid JSON object. No preamble, no markdown.
{_SCHEMA}
“””

```
def parse_response(self, raw: dict) -> EscalationRequest:
    customer_raw = raw.pop("customer", {})
    customer = CustomerInfo(**{k: v for k, v in customer_raw.items() if v is not None})
    return EscalationRequest(
        urgency=UrgencyLevel(raw["urgency"]),
        reason=raw["reason"],
        customer=customer,
        conversation_summary=raw["conversation_summary"],
        message_to_user=raw["message_to_user"],
        notify_via=raw.get("notify_via", ["whatsapp"]),
    )

def run(  # type: ignore[override]
    self,
    user_message: str,
    history: list[ConversationMessage] | None = None,
    classification: IntentClassification | None = None,
    booking_details: BookingDetails | None = None,
    **_,
) -> EscalationRequest:
    """
    Args:
        classification:  IntentClassification from the orchestrator (provides context).
        booking_details: Any BookingDetails already collected (attached to the request).
    """
    extra_parts = []

    if classification:
        extra_parts.append(
            f"Intent classification:\n"
            f"  intent={classification.intent}, "
            f"  urgency={classification.urgency}, "
            f"  sentiment={classification.sentiment}, "
            f"  is_emergency={classification.is_emergency}, "
            f"  date_seems_wrong={classification.date_seems_wrong}"
            + (
                (
                    f"\n  NOTE: date_seems_wrong=True. "
                    + (
                        f"Customer gave a PAST date ({classification.detected_date}) for a booking -- "
                        f"cannot book in the past. Ask for a valid future date."
                        if classification.detected_date and
                           (classification.detected_date - __import__("datetime").date.today()).days < 0
                        else
                        f"Customer gave a FUTURE date ({classification.detected_date}) for a complaint -- "
                        f"incident hasn't happened yet. Ask for the correct past date."
                    )
                )
                if classification.date_seems_wrong else ""
            )
        )

    if booking_details:
        filled = {
            k: v for k, v in booking_details.model_dump().items()
            if v is not None and v != {}
        }
        extra_parts.append(
            f"Booking details collected so far:\n{json.dumps(filled, indent=2, default=str)}"
        )

    extra = "\n\n".join(extra_parts)

    raw = self._call_llm(user_message, history, extra_system=extra)
    result = self.parse_response(raw)

    # Always attach whatever booking details we have
    if booking_details:
        result.booking_details = booking_details

    return result
```