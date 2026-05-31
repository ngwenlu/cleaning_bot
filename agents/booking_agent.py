"""
Booking Agent

Collects booking details progressively across turns.
Does NOT confirm, schedule, or commit to any booking.
Detects emergency bookings and flags them immediately.
"""

from __future__ import annotations

import json
from datetime import date as _date

from agents.base import BaseAgent
from knowledge_base import CONFIG as _CONFIG
from models import (
    BookingAgentResponse,
    BookingDetails,
    ConversationMessage,
)


_SCHEMA = json.dumps(
    {
        "message": "Natural language reply to the user",
        "collected": {
            "customer_name": "string or null",
            "address": "string or null",
            "requested_date": "YYYY-MM-DD or null",
            "requested_time": "HH:MM:SS or null",
            "hours_needed": "number or null",
            "has_pets": "boolean or null, where null means not yet asked",
            "contact": "phone or email string if volunteered, else null",
            "notes": "any extra context the customer mentions, else null",
        },
        "is_complete": "boolean, true only when ALL required fields are filled",
        "next_field_to_ask": "name of the next missing required field, or null if complete",
    },
    indent=2,
)


_REQUIRED_FIELDS_DESC = """
Required fields. Collect ALL before marking is_complete=true:

1. customer_name   - the customer's name
2. address         - full address of the property to be cleaned
3. requested_date  - date of the session, YYYY-MM-DD
4. requested_time  - start time of the session, HH:MM
5. hours_needed    - how many hours the cleaning will take
6. has_pets        - whether there are pets at the property, yes/no

Optional fields. Ask only after all required fields are collected, or if the customer volunteers:
- contact          - phone or email for the salesperson to follow up
- notes            - anything else relevant
"""


class BookingAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        today = _date.today()
        today_str = today.isoformat()
        today_day = today.strftime("%A")

        return f"""
You are the booking assistant for a part-time cleaning service.

Today is {today_str} ({today_day}).

YOUR ROLE:
- Collect the customer's booking details across the conversation, one or two fields at a time.
- Be friendly, concise, and conversational.
- Do not ask a long list of questions all at once.
- Ask for one missing field at a time unless the user volunteers multiple details at once.

DATE AND TIME RULES:
- When the customer mentions a relative date such as "next Saturday" or "tomorrow", calculate the actual YYYY-MM-DD date yourself.
- Do NOT ask them to type the date in YYYY-MM-DD format if you can infer it.
- Confirm inferred dates naturally, for example: "Got it, so that's Saturday 7 June."
- Reject impossible dates, for example month 14 or day 32, and ask for correction.
- Reject dates in the past. Bookings must be for future dates.
- Reject times outside service hours according to the BUSINESS RULES below.
- Also check that requested_time + hours_needed does not run past closing time.

CRITICAL RULES:
1. You CANNOT confirm, schedule, or commit to any booking.
2. NEVER say "your booking is confirmed" or anything that implies confirmation.
3. A human salesperson will follow up to confirm the booking.
4. After all details are collected, tell the customer:
   "Thank you! I've noted all your details. Our salesperson will contact you shortly to confirm your booking."
5. Always confirm that the customer will provide all cleaning supplies, including mop, vacuum, detergents, cloths, and gloves.
6. If the customer has not acknowledged supplies, ask them to confirm before marking complete.
7. If the requested date is today or tomorrow, DO NOT continue collecting details.
8. For urgent bookings, respond with exactly this message:
   "This looks like an urgent booking! Let me connect you with our team right away."
9. For urgent bookings, set is_complete=false and next_field_to_ask=null.

{_REQUIRED_FIELDS_DESC}

When returning collected data, include ALL previously known values.
This is the full current state of the form, not just what was collected in this turn.

{_CONFIG.rules_for_agents()}

Respond ONLY with a valid JSON object.
No preamble.
No markdown.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> BookingAgentResponse:
        collected_raw = raw.get("collected", {}) or {}

        collected = BookingDetails(
            **{
                key: value
                for key, value in collected_raw.items()
                if value is not None
            }
        )

        return BookingAgentResponse(
            message=raw["message"],
            collected=collected,
            is_complete=raw.get("is_complete", False),
            next_field_to_ask=raw.get("next_field_to_ask"),
        )

    def run(
        self,
        user_message: str,
        history: list[ConversationMessage] | None = None,
        existing_details: BookingDetails | None = None,
        **_,
    ) -> BookingAgentResponse:
        extra = ""

        if existing_details:
            filled = {
                key: value
                for key, value in existing_details.model_dump().items()
                if value is not None
                and value is not False
                and key
                not in (
                    "is_emergency",
                    "date_in_past",
                    "time_outside_hours",
                )
            }

            extra = (
                "Already collected. Do not re-ask these:\n"
                f"{json.dumps(filled, indent=2, default=str)}\n\n"
                f"Still missing: {existing_details.missing_fields()}"
            )

        raw = self._call_llm(
            user_message=user_message,
            history=history,
            extra_system=extra,
        )

        return self.parse_response(raw)