"""
agents.py – All agent logic in one place.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from dateutil import parser as date_parser

from config import (
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_MODEL,
    MAX_TOKENS,
    MODEL,
    SALESPERSON_EMAIL,
    SALESPERSON_WHATSAPP,
    client,
)
from knowledge_base import CONFIG, get_full_kb_text

log = logging.getLogger(__name__)


@dataclass
class Response:
    message: str
    agent: str = "assistant"
    form: dict = field(default_factory=dict)
    complete: bool = False
    escalate: bool = False
    intent: str = "unknown"
    debug: dict = field(default_factory=dict)


def _llm(
    system: str,
    messages: list[dict],
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
            response_format={"type": "json_object"},
        )

        content = (resp.choices[0].message.content or "").strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return json.loads(content) if content else {}

    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return {}


def _get(raw: dict, key: str, fallback):
    value = raw.get(key)
    return value if value is not None else fallback


def _extract_date_text(message: str) -> str | None:
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
        r"\btomorrow\b",
        r"\btoday\b",
        r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    ]

    lowered = message.lower()

    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def _extract_time_text(message: str) -> str | None:
    patterns = [
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        r"\b\d{1,2}\s*(?:am|pm)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def _parse_date(date_text: str | None, today: date) -> date | None:
    if not date_text:
        return None

    text = date_text.strip().lower()

    if text == "today":
        return today

    if text == "tomorrow":
        return today + timedelta(days=1)

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    if text.startswith("next "):
        day_name = text.replace("next ", "").strip()
        if day_name in weekdays:
            target = weekdays[day_name]
            days_ahead = (target - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    try:
        return date_parser.parse(
            date_text,
            dayfirst=True,
            fuzzy=True,
        ).date()
    except Exception:
        return None


def _parse_time(time_text: str | None) -> time | None:
    if not time_text:
        return None

    try:
        return date_parser.parse(time_text).time().replace(second=0, microsecond=0)
    except Exception:
        return None


_CLASSIFY_SCHEMA = json.dumps(
    {
        "intent": "booking | faq | escalation | feedback | out_of_scope",
        "sentiment": "positive | neutral | negative | urgent",
        "urgency": "routine | high | critical",
        "confidence": "float 0.0-1.0",
        "reasoning": "one sentence",
    },
    indent=2,
)


def classify(message: str, history: list[dict]) -> dict:
    today = date.today()

    system = f"""You are the intent classifier for a cleaning service chatbot.

Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

Classify intent and sentiment only.
Do NOT parse, validate, or reason about dates and times.

Return ONLY valid JSON:
{_CLASSIFY_SCHEMA}
"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
    )

    date_text = _extract_date_text(message)
    time_text = _extract_time_text(message)

    detected_date = _parse_date(date_text, today)
    detected_time = _parse_time(time_text)

    emergency_window = CONFIG.booking.emergency_window_days
    start_hour = CONFIG.hours.start_hour
    end_hour = CONFIG.hours.end_hour
    min_hours = int(CONFIG.booking.min_hours)
    max_advance_days = CONFIG.booking.max_advance_days

    delta = (detected_date - today).days if detected_date else None

    clf = {
        "intent": raw.get("intent", "faq"),
        "sentiment": raw.get("sentiment", "neutral"),
        "urgency": raw.get("urgency", "routine"),
        "confidence": raw.get("confidence", 0.5),
        "reasoning": raw.get("reasoning", ""),
        "date_text": date_text,
        "time_text": time_text,
        "detected_date": detected_date.isoformat() if detected_date else None,
        "detected_time": detected_time.strftime("%H:%M") if detected_time else None,
    }

    clf["is_emergency"] = (
        detected_date is not None
        and delta is not None
        and 0 <= delta <= emergency_window
    )

    clf["date_in_past"] = (
        detected_date is not None
        and delta is not None
        and delta < 0
    )

    clf["date_too_far"] = (
        detected_date is not None
        and delta is not None
        and delta > max_advance_days
    )

    clf["time_outside_hours"] = (
        detected_time is not None
        and (
            detected_time.hour < start_hour
            or detected_time.hour >= end_hour
        )
    )

    clf["time_too_late"] = (
        detected_time is not None
        and not clf["time_outside_hours"]
        and (
            detected_time.hour > (end_hour - min_hours)
            or (
                detected_time.hour == (end_hour - min_hours)
                and detected_time.minute > 0
            )
        )
    )

    clf["date_seems_wrong"] = (
        clf["date_in_past"] and clf["intent"] == "booking"
    ) or (
        detected_date is not None
        and delta is not None
        and delta > emergency_window
        and clf["intent"] in ("feedback", "escalation")
    )

    return clf


_BOOKING_REQUIRED = [
    "customer_name",
    "address",
    "requested_date",
    "requested_time",
    "hours_needed",
    "has_pets",
]


_BOOKING_SCHEMA = json.dumps(
    {
        "message": "Conversational reply to the user",
        "collected": {
            "customer_name": "string or null",
            "address": "string or null",
            "requested_date": "YYYY-MM-DD or null",
            "requested_time": "HH:MM or null",
            "hours_needed": "number or null",
            "has_pets": "boolean or null",
            "contact": "phone or email if volunteered, else null",
            "notes": "any extra context, else null",
        },
        "is_complete": "true when ALL 6 required fields are filled",
        "next_field": "next missing required field, or null",
        "escalate": "true ONLY if booking is today or tomorrow",
    },
    indent=2,
)


def run_booking(
    message: str,
    history: list[dict],
    form: dict,
    clf: dict | None = None,
) -> Response:
    today = date.today()

    filled = {k: v for k, v in form.items() if v is not None}
    missing = [k for k in _BOOKING_REQUIRED if form.get(k) is None]

    alerts = []

    if clf:
        if clf.get("date_in_past") and clf.get("detected_date"):
            alerts.append(
                f"The requested date {clf['detected_date']} is in the past. "
                "Reject this date and ask for a future date."
            )

        if clf.get("date_too_far") and clf.get("detected_date"):
            alerts.append(
                f"The requested date {clf['detected_date']} is too far in advance. "
                f"Customers can only book up to {CONFIG.booking.max_advance_days} days "
                f"({CONFIG.booking.max_advance_days // 30} months) from today. "
                "Reject this date and ask for a date within the next 6 months."
            )

        if clf.get("is_emergency") and clf.get("detected_date"):
            alerts.append(
                f"The requested date {clf['detected_date']} is today or tomorrow. "
                "This is an emergency booking. Set escalate=true."
            )

        if clf.get("time_outside_hours") and clf.get("detected_time"):
            alerts.append(
                f"The requested time {clf['detected_time']} is outside service hours. "
                f"Service hours are {CONFIG.hours.start_label} to {CONFIG.hours.end_label}. "
                "Ask for a time within service hours."
            )

        if clf.get("time_too_late") and clf.get("detected_time"):
            alerts.append(
                f"The requested start time {clf['detected_time']} is too late. "
                f"The latest start time is {CONFIG.hours.latest_start(CONFIG.booking.min_hours)} "
                f"because the minimum booking is {CONFIG.booking.min_hours:.0f} hours "
                f"and sessions must finish by {CONFIG.hours.end_label}. "
                "Reject this time and ask for an earlier start time."
            )

    alerts_text = "\n".join(f"- {alert}" for alert in alerts)

    system = f"""You are the booking assistant for a cleaning service.

Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

PYTHON-VERIFIED DATETIME ALERTS:
{alerts_text if alerts_text else "None"}

FORM STATE:
Required fields: {_BOOKING_REQUIRED}
Already collected: {json.dumps(filled, default=str)}
Still missing: {missing}

RULES:
- Collect required booking details one or two fields at a time.
- Never confirm or commit to a booking.
- A salesperson must confirm all bookings.
- If datetime alerts are present, handle them first.
- If multiple datetime alerts are present, handle them in this priority:
  1. date in past
  2. date too far ahead
  3. emergency date
  4. time outside hours
  5. time too late
- If date is too far ahead, explain that customers can only book up to 6 months ahead.
- If time is too late, explain that the latest start is 6pm because the minimum booking is 3 hours and cleaning must end by 9pm.
- If all required fields are collected, say:
  "Thank you! I've noted everything down. Our salesperson will contact you shortly to confirm."
- Set escalate=true only if the booking date is today or tomorrow.

Return ONLY valid JSON:
{_BOOKING_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    new_collected = raw.get("collected") or {}
    updated_form = {
        **form,
        **{k: v for k, v in new_collected.items() if v is not None},
    }

    return Response(
        message=_get(
            raw,
            "message",
            "Sorry, I didn't catch that — could you repeat it?",
        ),
        agent="booking",
        form=updated_form,
        complete=bool(raw.get("is_complete", False)),
        escalate=bool(raw.get("escalate", False)),
        debug={"next_field": raw.get("next_field")},
    )


_FAQ_SCHEMA = json.dumps(
    {
        "message": "Answer in plain conversational English",
        "sources": ["list", "of", "kb", "keys", "used"],
        "answered": "true if the KB covers the question, false otherwise",
    },
    indent=2,
)


def run_faq(message: str, history: list[dict]) -> Response:
    system = f"""You are the FAQ assistant for a cleaning service in Singapore.

Answer ONLY using the knowledge base below.
If the question is not covered, set answered=false.

{CONFIG.rules_for_agents()}

KNOWLEDGE BASE:
{get_full_kb_text()}

Return ONLY valid JSON:
{_FAQ_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    sources = raw.get("sources", [])

    if isinstance(sources, str):
        try:
            sources = ast.literal_eval(sources)
        except Exception:
            sources = [sources] if sources else []

    if not isinstance(sources, list):
        sources = []

    answered = bool(raw.get("answered", True))

    return Response(
        message=_get(
            raw,
            "message",
            "Let me connect you with our team for more details.",
        ),
        agent="faq",
        escalate=not answered,
        debug={"sources": sources},
    )


_ESCALATION_SCHEMA = json.dumps(
    {
        "message_to_user": "Empathetic reply to user",
        "summary": "2-3 sentence summary for salesperson",
        "urgency": "routine | high | critical",
    },
    indent=2,
)


def run_escalation(
    message: str,
    history: list[dict],
    context: dict | None = None,
) -> Response:
    system = f"""You are the escalation handler for a cleaning service chatbot.

{CONFIG.rules_for_agents()}

Salesperson contacts:
WhatsApp: {SALESPERSON_WHATSAPP}
Email: {SALESPERSON_EMAIL}

Context:
{json.dumps(context or {}, indent=2, default=str)}

RULES:
- Emergency booking today/tomorrow: reassure the customer and say the team will WhatsApp them shortly. Set urgency=critical.
- Complaint or negative feedback: empathise and promise human follow-up. Set urgency=high.
- Unanswerable FAQ: politely say you will connect them with the team. Set urgency=routine.
- Invalid date/time: explain clearly and ask the customer to clarify. Do not treat as urgent unless today/tomorrow.

Return ONLY valid JSON:
{_ESCALATION_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    return Response(
        message=_get(
            raw,
            "message_to_user",
            "I'm connecting you with our team. Someone will be in touch shortly.",
        ),
        agent="escalation",
        escalate=True,
        debug={
            "summary": raw.get("summary"),
            "urgency": raw.get("urgency"),
        },
    )


def process_message(
    message: str,
    history: list[dict],
    form: dict,
) -> Response:
    try:
        clf = classify(message, history)
        intent = clf["intent"]

        booking_started = any(
            form.get(k)
            for k in [
                "customer_name",
                "address",
                "requested_date",
                "requested_time",
            ]
        )

        is_emergency = clf.get("is_emergency", False)
        date_wrong = clf.get("date_seems_wrong", False)

        if is_emergency:
            resp = run_escalation(message, history, clf)

        elif intent == "escalation" or (date_wrong and intent == "feedback"):
            resp = run_escalation(message, history, clf)

        elif intent == "booking" or (booking_started and intent != "feedback"):
            resp = run_booking(message, history, form, clf=clf)

            if resp.escalate:
                saved_form = resp.form
                resp = run_escalation(message, history, {**clf, "form": saved_form})
                resp.form = saved_form

        elif intent in ("faq", "out_of_scope"):
            resp = run_faq(message, history)

            if resp.escalate:
                resp = run_escalation(message, history, clf)

        elif intent == "feedback":
            resp = run_escalation(message, history, clf)

        else:
            resp = run_faq(message, history)

        resp.intent = intent

        resp.debug.update(
            {
                "intent": intent,
                "sentiment": clf.get("sentiment"),
                "urgency": clf.get("urgency"),
                "confidence": clf.get("confidence"),
                "reasoning": clf.get("reasoning"),
                "date_text": clf.get("date_text"),
                "time_text": clf.get("time_text"),
                "detected_date": clf.get("detected_date"),
                "detected_time": clf.get("detected_time"),
                "is_emergency": clf.get("is_emergency"),
                "date_in_past": clf.get("date_in_past"),
                "date_too_far": clf.get("date_too_far"),
                "time_outside_hours": clf.get("time_outside_hours"),
                "time_too_late": clf.get("time_too_late"),
            }
        )

        return resp

    except Exception as exc:
        log.error("process_message crashed: %s", exc, exc_info=True)

        return Response(
            message="I'm sorry, I ran into a technical issue. Could you try again?",
            agent="system",
            form=form,
        )