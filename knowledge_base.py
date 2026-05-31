"""
knowledge_base.py – Single source of truth for ALL business rules and FAQ content.

To change any business rule such as hours, pricing, cancellation policy:
1. Edit the relevant field inside BusinessConfig below.
2. Done. All agents, validators, and FAQ answers update automatically.

To add a new FAQ topic:
1. Add a KBEntry to KNOWLEDGE_BASE below.
2. Done. The FAQ agent picks it up on next deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServiceHours:
    start_hour: int = 9
    end_hour: int = 21
    days: str = "7 days a week"

    @property
    def start_label(self) -> str:
        hour = self.start_hour
        return f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}"

    @property
    def end_label(self) -> str:
        hour = self.end_hour
        return f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}"

    @property
    def latest_start_for(self) -> str:
        return (
            f"Sessions must finish by {self.end_label}, so the latest start time "
            f"depends on how many hours are needed. For example, a 3-hour session "
            f"must start by {(self.end_hour - 3) % 12 or 12}pm."
        )


@dataclass
class Pricing:
    hourly_rate: int = 20
    currency: str = "SGD"

    @property
    def description(self) -> str:
        return (
            f"${self.hourly_rate} {self.currency} per hour "
            f"(flat rate, no variation by home size or location)"
        )


@dataclass
class CancellationPolicy:
    notice_hours: int = 24
    late_fee_pct: int = 50


@dataclass
class BookingRules:
    emergency_window_days: int = 1
    customer_provides_supplies: bool = True
    required_supplies: list[str] = field(
        default_factory=lambda: [
            "Mop and bucket",
            "Vacuum cleaner",
            "Broom and dustpan",
            "Floor cleaner / detergent",
            "Toilet cleaner and scrubber",
            "Microfibre cloths or old rags",
            "Rubber gloves for the cleaner",
        ]
    )


@dataclass
class BusinessConfig:
    company_name: str = "Dad's Cleaning Services"
    service_areas: str = "all areas in Singapore"
    payment_methods: list[str] = field(default_factory=lambda: ["Cash", "PayNow"])
    hours: ServiceHours = field(default_factory=ServiceHours)
    pricing: Pricing = field(default_factory=Pricing)
    cancellation: CancellationPolicy = field(default_factory=CancellationPolicy)
    booking: BookingRules = field(default_factory=BookingRules)

    def rules_for_agents(self) -> str:
        h = self.hours
        p = self.pricing
        c = self.cancellation
        b = self.booking

        supplies_list = "\n".join(f"    - {item}" for item in b.required_supplies)
        payment_methods = ", ".join(self.payment_methods)

        return f"""
=== BUSINESS RULES: AUTHORITATIVE ===

Company: {self.company_name}
Service area: {self.service_areas}
Payment: {payment_methods}

SERVICE HOURS:
Open: {h.start_label} to {h.end_label}, {h.days}
Start time: no earlier than {h.start_label}
End time: no later than {h.end_label}. Sessions must finish by {h.end_label}.
{h.latest_start_for}
Any time outside {h.start_label} to {h.end_label} must be politely rejected.

PRICING:
{p.description}

CANCELLATION POLICY:
Notice required: {c.notice_hours} hours before the session
Late fee: {c.late_fee_pct}% of session cost if cancelled with less than {c.notice_hours} hours notice

BOOKING RULES:
Emergency booking: any session for today or within the next {b.emergency_window_days} day(s)
Supplies: customer provides ALL cleaning supplies. Cleaners bring NOTHING.
Required supplies the customer must prepare:
{supplies_list}

=== END BUSINESS RULES ===
""".strip()


CONFIG = BusinessConfig()


@dataclass
class KBEntry:
    key: str
    topics: list[str]
    answer: str


KNOWLEDGE_BASE: list[KBEntry] = [
    KBEntry(
        key="service_hours",
        topics=[
            "hours",
            "time",
            "when",
            "available",
            "availability",
            "early",
            "late",
            "am",
            "pm",
            "open",
            "3am",
            "midnight",
        ],
        answer=(
            f"Our cleaners are available from {CONFIG.hours.start_label} to "
            f"{CONFIG.hours.end_label}, {CONFIG.hours.days}. "
            f"{CONFIG.hours.latest_start_for} "
            "We cannot accommodate requests outside these hours."
        ),
    ),
    KBEntry(
        key="service_type",
        topics=[
            "service",
            "what do you offer",
            "cleaning type",
        ],
        answer=(
            "We offer part-time cleaning where our cleaners come to your home or office. "
            "You will need to provide all cleaning supplies. "
            "Our cleaners do not bring any equipment or products."
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
            + "\n".join(f"- {item}" for item in CONFIG.booking.required_supplies)
            + "\n\nIf any supplies are missing, the cleaner may not be able to complete certain tasks."
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
            f"Our rate is {CONFIG.pricing.description}. "
            "The total cost depends only on how many hours the job takes. "
            f"For example, a 3-hour session costs ${CONFIG.pricing.hourly_rate * 3} "
            f"and a 4-hour session costs ${CONFIG.pricing.hourly_rate * 4}. "
            "Final pricing is confirmed by our salesperson after reviewing your details."
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
            "1. You share your details with our chatbot, including date, address, home size, and hours needed.\n"
            "2. Our salesperson reviews your request and contacts you to confirm the slot.\n"
            "3. Once confirmed, we send the cleaner to your place at the agreed time.\n\n"
            "Note: Bookings are NOT confirmed through this chat. A human salesperson will reach out to you."
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
            f"Cancellations or reschedules must be made at least "
            f"{CONFIG.cancellation.notice_hours} hours before the scheduled session. "
            f"Late cancellations under {CONFIG.cancellation.notice_hours} hours may incur a "
            f"{CONFIG.cancellation.late_fee_pct}% cancellation fee. "
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
            f"We currently serve {CONFIG.service_areas}, including HDB estates, "
            "condominiums, landed properties, and small offices."
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
            "We offer both one-time and recurring cleaning sessions, including weekly, fortnightly, or monthly. "
            "Recurring sessions are arranged directly with the salesperson and may qualify for a discount."
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
            f"Accepted methods: {', '.join(CONFIG.payment_methods)}."
        ),
    ),
]


def get_full_kb_text() -> str:
    """Return the full knowledge base as formatted text for injection into prompts."""
    lines: list[str] = []

    for entry in KNOWLEDGE_BASE:
        lines.append(f"### {entry.key}")
        lines.append(entry.answer)
        lines.append("")

    return "\n".join(lines)


def get_source_keys() -> list[str]:
    return [entry.key for entry in KNOWLEDGE_BASE]