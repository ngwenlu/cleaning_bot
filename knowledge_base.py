"""
knowledge_base.py -- Single source of truth for ALL business rules and FAQ content.

To change any business rule (hours, pricing, cancellation policy, etc.):
  1. Edit the relevant field inside BusinessConfig below.
  2. Done -- all agents, validators, and FAQ answers update automatically.

To add a new FAQ topic:
  1. Add a KBEntry to KNOWLEDGE_BASE below.
  2. Done -- the FAQ agent picks it up on next deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ===========================================================================
# BUSINESS CONFIGURATION -- edit here to update any rule across the whole app
# ===========================================================================

@dataclass
class ServiceHours:
    start_hour: int = 9    # 24hr -- earliest session START (9am)
    end_hour:   int = 21   # 24hr -- latest session END    (9pm)
    days:       str = "7 days a week"

    @property
    def start_label(self) -> str:
        h = self.start_hour
        return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"

    @property
    def end_label(self) -> str:
        h = self.end_hour
        return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"

    def latest_start(self, min_hours: float) -> str:
        """Latest allowed start time given a minimum session length."""
        h = self.end_hour - int(min_hours)
        return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


@dataclass
class Pricing:
    hourly_rate: int    = 20
    currency:    str    = "SGD"

    @property
    def description(self) -> str:
        return f"${self.hourly_rate}/{self.currency} per hour (flat rate, no variation by home size or location)"


@dataclass
class CancellationPolicy:
    notice_hours:    int = 24   # hours of notice required
    late_fee_pct:    int = 50   # % of session cost charged for late cancellation


@dataclass
class BookingRules:
    # Number of days ahead that counts as an "emergency" (0 = today, 1 = tomorrow)
    emergency_window_days: int = 1

    # Session duration limits (hours)
    min_hours: float = 3.0   # minimum session length
    max_hours: float = 8.0   # maximum session length

    # How far ahead a booking can be made
    max_advance_days: int = 183  # approx 6 months

    # Customer must provide all supplies (our cleaners bring nothing)
    customer_provides_supplies: bool = True
    required_supplies: list[str] = field(default_factory=lambda: [
        "Mop and bucket",
        "Vacuum cleaner",
        "Broom and dustpan",
        "Floor cleaner / detergent",
        "Toilet cleaner and scrubber",
        "Microfibre cloths or old rags",
        "Rubber gloves for the cleaner",
    ])


@dataclass
class BusinessConfig:
    company_name:    str   = "Dad's Cleaning Services"
    service_areas:   str   = "all areas in Singapore"
    payment_methods: list[str] = field(default_factory=lambda: ["Cash", "PayNow"])
    hours:           ServiceHours       = field(default_factory=ServiceHours)
    pricing:         Pricing            = field(default_factory=Pricing)
    cancellation:    CancellationPolicy = field(default_factory=CancellationPolicy)
    booking:         BookingRules       = field(default_factory=BookingRules)

    def rules_for_agents(self) -> str:
        """
        Returns a plain-English rules block injected into every agent system prompt.
        When you update any field above, this text updates automatically.
        """
        h = self.hours
        p = self.pricing
        c = self.cancellation
        b = self.booking
        supplies_list = "\n".join(f"    - {s}" for s in b.required_supplies)
        return f"""
=== BUSINESS RULES (authoritative -- always follow these exactly) ===
Company      : {self.company_name}
Service area : {self.service_areas}
Payment      : {', '.join(self.payment_methods)}

SERVICE HOURS:
  Open            : {h.start_label} to {h.end_label}, {h.days}
  Earliest start  : {h.start_label} (INCLUSIVE -- {h.start_label} is a valid start)
  Latest end      : {h.end_label} (INCLUSIVE -- session can end exactly at {h.end_label})
  Latest start    : {h.latest_start(b.min_hours)} (INCLUSIVE -- {h.latest_start(b.min_hours)} IS a valid start time)
  Example         : {h.latest_start(b.min_hours)} start + {b.min_hours:.0f}h minimum = {h.end_label} end. VALID.
  Only reject start times AFTER {h.latest_start(b.min_hours)} -- i.e. 7pm or later is invalid.
  Any time before {h.start_label} or after {h.end_label} must be politely rejected.

PRICING:
  {p.description}

CANCELLATION POLICY:
  Notice required : {c.notice_hours} hours before the session
  Late fee        : {c.late_fee_pct}% of session cost if cancelled with less than {c.notice_hours}h notice

BOOKING RULES:
  Chatbot role       : This chatbot can ONLY collect booking details. It CANNOT check
                       availability, confirm bookings, or assign cleaners. A human
                       salesperson will review the details and follow up with the customer
                       to confirm. Always make this clear when collecting details. Chatbot
                       must ALWAYS collect name, hours, start time, pet information and contact number.
  Emergency booking  : any session for today or within the next {b.emergency_window_days} day(s)
  Minimum hours      : {b.min_hours:.0f} hours per session
  Maximum hours      : {b.max_hours:.0f} hours per session
  Furthest advance   : {b.max_advance_days} days ({b.max_advance_days // 30} months) from today
  Latest start time  : {h.latest_start(b.min_hours)} INCLUSIVE -- a {b.min_hours:.0f}h session starting at {h.latest_start(b.min_hours)} ends exactly at {h.end_label}. Valid.
  Invalid start      : {int(h.end_hour - b.min_hours) + 1}pm or later (would end after {h.end_label})
  Supplies           : customer provides ALL cleaning supplies. Cleaners bring NOTHING.
  Required supplies the customer must prepare:
{supplies_list}
=== END BUSINESS RULES ===
""".strip()


# ---------------------------------------------------------------------------
# THE SINGLE INSTANCE -- import CONFIG anywhere you need a business rule
# ---------------------------------------------------------------------------
CONFIG = BusinessConfig()


# ===========================================================================
# FAQ KNOWLEDGE BASE -- add KBEntry items here to extend what the bot knows
# ===========================================================================

from dataclasses import dataclass as _dc


@_dc
class KBEntry:
    key:     str
    topics:  list[str]
    answer:  str


KNOWLEDGE_BASE: list[KBEntry] = [
    KBEntry(
        key="service_hours",
        topics=["hours", "time", "when", "available", "availability", "early", "late",
                "am", "pm", "open", "3am", "midnight", "latest", "earliest"],
        answer=(
            f"Our cleaners are available from {CONFIG.hours.start_label} to "
            f"{CONFIG.hours.end_label}, {CONFIG.hours.days}. "
            f"Sessions must finish by {CONFIG.hours.end_label}, and the minimum session "
            f"length is {CONFIG.booking.min_hours:.0f} hours, so the latest you can start "
            f"is {CONFIG.hours.latest_start(CONFIG.booking.min_hours)}. "
            f"We cannot accommodate requests outside these hours."
        ),
    ),
    KBEntry(
        key="service_type",
        topics=["service", "what do you offer", "cleaning type"],
        answer=(
            "We offer part-time cleaning where our cleaners come to your home or office. "
            f"You will need to provide all cleaning supplies. "
            f"Our cleaners do not bring any equipment or products."
        ),
    ),
    KBEntry(
        key="supplies",
        topics=["supplies", "equipment", "what to prepare", "bring"],
        answer=(
            "Please prepare the following before the cleaner arrives:\n"
            + "\n".join(f"- {s}" for s in CONFIG.booking.required_supplies)
            + "\n\nIf any supplies are missing, the cleaner may not be able to complete certain tasks."
        ),
    ),
    KBEntry(
        key="pricing",
        topics=["price", "cost", "rate", "how much", "fee", "charge", "bedroom", "room"],
        answer=(
            f"Our rate is {CONFIG.pricing.description}. "
            f"The total cost depends only on how many hours the job takes. "
            f"For example, a 3-hour session costs "
            f"${CONFIG.pricing.hourly_rate * 3} and a 4-hour session costs "
            f"${CONFIG.pricing.hourly_rate * 4}. "
            f"Final pricing is confirmed by our salesperson after reviewing your details."
        ),
    ),
    KBEntry(
        key="advance_booking",
        topics=["how far", "advance", "future", "months ahead", "plan ahead", "earliest", "far in advance"],
        answer=(
            f"You can book up to {CONFIG.booking.max_advance_days // 30} months "
            f"({CONFIG.booking.max_advance_days} days) in advance. "
            f"Bookings beyond this window cannot be accepted. "
            f"The minimum session length is {CONFIG.booking.min_hours:.0f} hours, "
            f"so the latest start time is {CONFIG.hours.latest_start(CONFIG.booking.min_hours)}."
        ),
    ),
    KBEntry(
        key="booking_process",
        topics=["how to book", "booking process", "how does it work", "steps"],
        answer=(
            "Here's how it works:\n"
            "1. You share your details with our chatbot (date, address, home size, hours needed).\n"
            "2. Our salesperson reviews your request and contacts you to confirm the slot.\n"
            "3. Once confirmed, we send the cleaner to your place at the agreed time.\n\n"
            "Note: Bookings are NOT confirmed through this chat -- a human salesperson will reach out to you."
        ),
    ),
    KBEntry(
        key="cancellation",
        topics=["cancel", "reschedule", "change booking", "postpone"],
        answer=(
            f"Cancellations or reschedules must be made at least "
            f"{CONFIG.cancellation.notice_hours} hours before the scheduled session. "
            f"Late cancellations (under {CONFIG.cancellation.notice_hours} hours) may incur a "
            f"{CONFIG.cancellation.late_fee_pct}% cancellation fee. "
            f"To cancel or reschedule, please contact us via WhatsApp or reply to your booking confirmation."
        ),
    ),
    KBEntry(
        key="service_areas",
        topics=["where", "area", "location", "service area", "coverage"],
        answer=(
            f"We currently serve {CONFIG.service_areas}, including HDB estates, condominiums, "
            f"landed properties, and small offices."
        ),
    ),
    KBEntry(
        key="cleaner_profile",
        topics=["who is the cleaner", "cleaner background", "trusted", "safe", "verified"],
        answer=(
            "All our cleaners are vetted, experienced, and have been personally screened by us. "
            "We do not use agencies. You can expect a punctual, professional cleaner who respects your home."
        ),
    ),
    KBEntry(
        key="frequency",
        topics=["recurring", "weekly", "monthly", "regular", "one time", "ad hoc"],
        answer=(
            "We offer both one-time and recurring cleaning sessions (weekly, fortnightly, or monthly). "
            "Recurring sessions are arranged directly with the salesperson and may qualify for a discount."
        ),
    ),
    KBEntry(
        key="payment",
        topics=["payment", "pay", "cash", "paynow", "transfer"],
        answer=(
            f"Payment is made directly to the cleaner after the session. "
            f"Accepted methods: {', '.join(CONFIG.payment_methods)}."
        ),
    ),
]


def get_full_kb_text() -> str:
    """Return the full knowledge base as formatted text for injection into prompts."""
    lines = []
    for entry in KNOWLEDGE_BASE:
        lines.append(f"### {entry.key}")
        lines.append(entry.answer)
        lines.append("")
    return "\n".join(lines)


def get_source_keys() -> list[str]:
    return [e.key for e in KNOWLEDGE_BASE]
