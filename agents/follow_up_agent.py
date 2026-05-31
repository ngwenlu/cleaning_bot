"""
Follow-up Agent

Handles outbound follow-up messages:

- reminder_24h
- post_service_feedback
- rebooking_prompt

In the Streamlit app this agent is triggered by a scheduler or cron job,
not by an inbound user message.
"""

from __future__ import annotations

import json

from agents.base import BaseAgent
from models import (
    BookingDetails,
    ConversationMessage,
    CustomerInfo,
    FollowUpAgentResponse,
)


_SCHEMA = json.dumps(
    {
        "message": "The follow-up message to send to the customer via WhatsApp or SMS",
        "follow_up_type": (
            "one of: reminder_24h | post_service_feedback | rebooking_prompt"
        ),
        "customer": {
            "name": "string or null",
            "phone": "string or null",
            "email": "string or null",
        },
    },
    indent=2,
)


class FollowUpAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return f"""
You are the follow-up messaging assistant for a part-time cleaning service.

You generate short, warm WhatsApp-style messages for three scenarios.

1. reminder_24h
- Sent the day before a confirmed booking.
- Remind the customer of the date, time, and address.
- Remind them to prepare all cleaning supplies.
- Keep it under 80 words.

2. post_service_feedback
- Sent a few hours after a completed session.
- Thank the customer and ask for a quick rating or feedback.
- Keep it under 60 words.

3. rebooking_prompt
- Sent 3 weeks after the last session.
- Warm and friendly, not pushy.
- Mention it might be time for another clean.
- Keep it under 60 words.

Write in a friendly, human tone.
This is WhatsApp, not a formal email.

Use the customer's first name if available.

Respond ONLY with a valid JSON object.
No markdown.
No explanations.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> FollowUpAgentResponse:
        customer_raw = raw.pop("customer", {})

        customer = CustomerInfo(
            **{
                key: value
                for key, value in customer_raw.items()
                if value is not None
            }
        )

        return FollowUpAgentResponse(
            message=raw["message"],
            follow_up_type=raw["follow_up_type"],
            customer=customer,
        )

    def run(
        self,
        follow_up_type: str,
        customer: CustomerInfo,
        booking_details: BookingDetails | None = None,
        history: list[ConversationMessage] | None = None,
        **_,
    ) -> FollowUpAgentResponse:
        """
        Args:
            follow_up_type:
                reminder_24h | post_service_feedback | rebooking_prompt

            customer:
                CustomerInfo used for personalisation.

            booking_details:
                Booking context for reminder messages.
        """

        context_parts = [
            f"Follow-up type: {follow_up_type}",
            f"Customer: {json.dumps(customer.model_dump(), default=str)}",
        ]

        if booking_details:
            context_parts.append(
                f"Booking details: "
                f"{json.dumps(booking_details.model_dump(), default=str)}"
            )

        prompt = "\n".join(context_parts)

        raw = self._call_llm(
            user_message=prompt,
            history=history,
        )

        return self.parse_response(raw)