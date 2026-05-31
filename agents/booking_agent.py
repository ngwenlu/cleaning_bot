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
from models import (
    BookingAgentResponse,
    BookingDetails,
    ConversationMessage,
    CustomerInfo,
)


_SCHEMA = json.dumps(
    {
        "message": "Natural language reply to the user",
        "collected": {
            "customer": {
                "name": "string or null",
                "phone": "string or null",
                "email": "string or null",
            },
            "requested_date": "YYYY-MM-DD or null",
            "requested_time": "HH:MM:SS or null",
            "address": "string or null",
            "postal_code": "string or null",
            "apartment_type": (
                "one of: hdb_1_2_room | hdb_3_room | hdb_4_room | "
                "hdb_5_room | condo_studio | condo_1_bed | condo_2_bed | "
                "condo_3_bed | landed | office | other, or null"
            ),
            "hours_needed": "number or null",
            "num_rooms": "integer or null",
            "special_instructions": "string or null",
            "supplies_confirmed": "boolean",
        },
        "is_complete": "boolean, true only when ALL required fields are filled",
        "next_field_to_ask": "name of the next missing required field, or null if complete",
    },
    indent=2,
)


_REQUIRED_FIELDS_DESC = """
Required fields. Collect all of these before marking is_complete=true:

1. customer.name
2. customer.phone OR customer.email
3. requested_date  (YYYY-MM-DD)
4. requested_time  (HH:MM)
5. address
6. apartment_type
7. hours_needed
8. supplies_confirmed, meaning the customer explicitly acknowledges that they will provide all supplies
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

DATE RULES:
- When the customer mentions a relative date such as "next Saturday", "this Sunday", "tomorrow", or "tonight", calculate the actual YYYY-MM-DD date yourself using today's date ({today_str}).
- Do NOT ask the customer to type the date out again if you can calculate it.
- Confirm calculated dates naturally.
- Example: "Got it, so that's Saturday 6 June. I'll note that down."
- If the customer provides an impossible date, politely point out the error and ask them to correct it.
- Do not store an invalid date.
- If the customer provides a date in the past, tell them a booking cannot be made for a past date and ask for a valid future date.
- Do not store a past date.

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

Respond ONLY with a valid JSON object.
No preamble.
No markdown.

Schema:
{_SCHEMA}
"""

    def parse_response(self, raw: dict) -> BookingAgentResponse:
        collected_raw = raw.get("collected", {}).copy()
        customer_raw = collected_raw.pop("customer", {}) or {}

        customer = CustomerInfo(
            **{
                key: value
                for key, value in customer_raw.items()
                if value is not None
            }
        )

        collected = BookingDetails(
            customer=customer,
            **{
                key: value
                for key, value in collected_raw.items()
                if value is not None
            },
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
                if value is not None and value != {} and value is not False
            }

            extra = (
                "Current collected details already known. Do not re-ask these:\n"
                f"{json.dumps(filled, indent=2, default=str)}\n\n"
                f"Missing fields: {existing_details.missing_fields()}"
            )

        raw = self._call_llm(
            user_message=user_message,
            history=history,
            extra_system=extra,
        )

        return self.parse_response(raw)