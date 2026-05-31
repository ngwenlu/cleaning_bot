“””
Booking Agent

Collects booking details progressively across turns.
Does NOT confirm, schedule, or commit to any booking.
Detects emergency bookings and flags them immediately.
“””

from **future** import annotations

import json

from agents.base import BaseAgent
from models import (
BookingAgentResponse,
BookingDetails,
ConversationMessage,
CustomerInfo,
)

_SCHEMA = json.dumps(
{
“message”: “Natural language reply to the user”,
“collected”: {
“customer”: {
“name”:  “string or null”,
“phone”: “string or null”,
“email”: “string or null”,
},
“requested_date”:      “YYYY-MM-DD or null”,
“requested_time”:      “HH:MM:SS or null”,
“address”:             “string or null”,
“postal_code”:         “string or null”,
“apartment_type”:      “one of: hdb_1_2_room | hdb_3_room | hdb_4_room | hdb_5_room | condo_studio | condo_1_bed | condo_2_bed | condo_3_bed | landed | office | other – or null”,
“hours_needed”:        “number or null”,
“num_rooms”:           “integer or null”,
“special_instructions”:“string or null”,
“supplies_confirmed”:  “boolean”,
},
“is_complete”: “boolean – true only when ALL required fields are filled”,
“next_field_to_ask”: “name of the next missing required field, or null if complete”,
},
indent=2,
)

_REQUIRED_FIELDS_DESC = “””
Required fields (collect all of these before marking is_complete=true):

1. customer.name
1. customer.phone OR customer.email
1. requested_date  (YYYY-MM-DD)
1. requested_time  (HH:MM)
1. address
1. apartment_type
1. hours_needed
1. supplies_confirmed (must explicitly acknowledge they will provide all supplies)
   “””

class BookingAgent(BaseAgent):

```
@property
def system_prompt(self) -> str:
    return f"""You are the booking assistant for a part-time cleaning service.
```

YOUR ROLE:

- Collect the customer’s booking details across the conversation, one or two fields at a time.
- Be friendly, concise, and conversational – don’t fire a list of questions all at once.
- Ask for one missing field at a time unless the user volunteers multiple at once.

CRITICAL RULES:

1. You CANNOT confirm, schedule, or commit to any booking. NEVER say “your booking is confirmed”
   or anything that implies confirmation. A human salesperson will follow up.
1. After all details are collected, tell the customer:
   “Thank you! I’ve noted all your details. Our salesperson will contact you shortly to confirm
   your booking.”
1. Always confirm that the customer will provide all cleaning supplies (mop, vacuum, detergents,
   cloths, gloves). If they haven’t acknowledged this, ask them to confirm before marking complete.
1. If the requested date is today or tomorrow, DO NOT continue collecting details –
   respond with exactly this message:
   “This looks like an urgent booking! Let me connect you with our team right away.”
   Then set is_complete=false and next_field_to_ask=null.

{_REQUIRED_FIELDS_DESC}

When returning collected data, include ALL previously known values – this is the full current
state of the form, not just what was collected in this turn.

Respond ONLY with a valid JSON object. No preamble, no markdown.
{_SCHEMA}
“””

```
def parse_response(self, raw: dict) -> BookingAgentResponse:
    # Parse nested CustomerInfo and BookingDetails safely
    collected_raw = raw.get("collected", {})
    customer_raw = collected_raw.pop("customer", {})
    customer = CustomerInfo(**{k: v for k, v in customer_raw.items() if v is not None})
    collected = BookingDetails(
        customer=customer,
        **{k: v for k, v in collected_raw.items() if v is not None},
    )
    return BookingAgentResponse(
        message=raw["message"],
        collected=collected,
        is_complete=raw.get("is_complete", False),
        next_field_to_ask=raw.get("next_field_to_ask"),
    )

def run(  # type: ignore[override]
    self,
    user_message: str,
    history: list[ConversationMessage] | None = None,
    existing_details: BookingDetails | None = None,
    **_,
) -> BookingAgentResponse:
    """
    Args:
        existing_details: The BookingDetails collected so far in this session.
                          Injected into the system prompt so the model knows
                          what's already been gathered.
    """
    extra = ""
    if existing_details:
        filled = {
            k: v
            for k, v in existing_details.model_dump().items()
            if v is not None and v != {} and v is not False
        }
        extra = (
            f"Current collected details (already known -- do not re-ask):\n"
            f"{json.dumps(filled, indent=2, default=str)}\n\n"
            f"Missing fields: {existing_details.missing_fields()}"
        )

    raw = self._call_llm(user_message, history, extra_system=extra)
    return self.parse_response(raw)
```