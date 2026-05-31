"""
Static knowledge base for the FAQ Agent.
Each entry has a key, a list of trigger topics, and the answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KBEntry:
    key: str
    topics: list[str]
    answer: str


KNOWLEDGE_BASE: list[KBEntry] = [
    KBEntry(
        key="service_type",
        topics=[
            "service",
            "what do you offer",
            "cleaning type",
        ],
        answer=(
            "We offer part-time cleaning where our cleaners come to your home or office. "
            "You will need to provide all cleaning supplies (mop, vacuum, detergents, cloths, etc.). "
            "We do not bring any equipment or products."
        ),
    ),
    KBEntry(
        key="supplies",
        topics=[
            "supplies",
            "equipment",
            "what to prepare",
            "bring",
        ],
        answer=(
            "Please prepare the following before the cleaner arrives:\n"
            "* Mop and bucket\n"
            "* Vacuum cleaner\n"
            "* Broom and dustpan\n"
            "* Floor cleaner / detergent\n"
            "* Toilet cleaner and scrubber\n"
            "* Glass cleaner (optional)\n"
            "* Microfibre cloths or old rags\n"
            "* Rubber gloves for the cleaner\n\n"
            "If any supplies are missing, the cleaner may not be able to complete certain tasks."
        ),
    ),
    KBEntry(
        key="pricing",
        topics=[
            "price",
            "cost",
            "rate",
            "how much",
            "fee",
            "charge",
            "bedroom",
            "room",
        ],
        answer=(
            "Our rate is a flat $20 per hour regardless of home size, number of rooms, or location. "
            "The total cost depends only on how many hours the job takes. "
            "For example, a 3-hour session costs $60 and a 4-hour session costs $80. "
            "The number of hours needed is estimated when you make your booking, "
            "and final pricing is confirmed by our salesperson."
        ),
    ),
    KBEntry(
        key="booking_process",
        topics=[
            "how to book",
            "booking process",
            "how does it work",
            "steps",
        ],
        answer=(
            "Here's how it works:\n"
            "1. You share your details with our chatbot (date, address, home size, hours needed).\n"
            "2. Our salesperson reviews your request and contacts you to confirm the slot.\n"
            "3. Once confirmed, we send the cleaner to your place at the agreed time.\n\n"
            "Note: Bookings are NOT confirmed through this chat. "
            "A human salesperson will reach out to you."
        ),
    ),
    KBEntry(
        key="cancellation",
        topics=[
            "cancel",
            "reschedule",
            "change booking",
            "postpone",
        ],
        answer=(
            "Cancellations or reschedules must be done at least 24 hours before the scheduled session. "
            "Late cancellations under 24 hours may incur a 50% cancellation fee. "
            "To cancel or reschedule, please contact us via WhatsApp or reply to your booking confirmation."
        ),
    ),
    KBEntry(
        key="service_areas",
        topics=[
            "where",
            "area",
            "location",
            "service area",
            "coverage",
        ],
        answer=(
            "We currently serve all areas in Singapore including HDB estates, condominiums, "
            "landed properties, and small offices. There are no area restrictions."
        ),
    ),
    KBEntry(
        key="cleaner_profile",
        topics=[
            "who is the cleaner",
            "cleaner background",
            "trusted",
            "safe",
            "verified",
        ],
        answer=(
            "All our cleaners are vetted, experienced, and have been personally screened by us. "
            "We do not use agencies. You can expect a punctual, professional cleaner who respects your home."
        ),
    ),
    KBEntry(
        key="frequency",
        topics=[
            "recurring",
            "weekly",
            "monthly",
            "regular",
            "one time",
            "ad hoc",
        ],
        answer=(
            "We offer both one-time and recurring cleaning sessions "
            "(weekly, fortnightly, or monthly). "
            "Recurring sessions are arranged directly with the salesperson "
            "and may qualify for a discount."
        ),
    ),
    KBEntry(
        key="payment",
        topics=[
            "payment",
            "pay",
            "cash",
            "paynow",
            "transfer",
        ],
        answer=(
            "Payment is made directly to the cleaner after the session. "
            "Accepted methods include Cash and PayNow."
        ),
    ),
]


def get_full_kb_text() -> str:
    """
    Return the full knowledge base as a formatted string
    for injection into prompts.
    """
    lines: list[str] = []

    for entry in KNOWLEDGE_BASE:
        lines.append(f"### {entry.key}")
        lines.append(entry.answer)
        lines.append("")

    return "\n".join(lines)


def get_source_keys() -> list[str]:
    return [entry.key for entry in KNOWLEDGE_BASE]