"""
agents.py – All agent logic in one place.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, time, timedelta

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

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
        r"\btoday\b",
        r"\btomorrow\b",
        r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
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
        return date_parser.parse(date_text, dayfirst=True, fuzzy=True).date()
    except Exception:
        return None


def _parse_time(time_text: str | None) -> time | None:
    if not time_text:
        return None

    try:
        return date_parser.parse(time_text).time().replace(second=0, microsecond=0)
    except Exception:
        return None


def _time_label(value) -> str:
    try:
        parsed = date_parser.parse(str(value)).time()
        hour = parsed.hour
        minute = parsed.minute
        suffix = "am" if hour < 12 else "pm"
        hour12 = hour % 12 or 12
        return f"{hour12}{suffix}" if minute == 0 else f"{hour12}:{minute:02d}{suffix}"
    except Exception:
        return str(value)


def _format_hours(hours) -> str:
    try:
        h = float(hours)
        return f"{h:g}"
    except Exception:
        return str(hours)


def _sanitize_no_confirmation_message(message: str) -> tuple[str, bool]:
    banned_phrases = [
        "confirmed",
        "booking is confirmed",
        "your booking is confirmed",
        "your cleaning session is confirmed",
        "confirmed for",
        "we have confirmed",
        "booking has been confirmed",
    ]

    lower = message.lower()

    if any(phrase in lower for phrase in banned_phrases):
        return (
            "Thank you. I've noted your request. "
            "A salesperson will review your details and contact you to confirm availability. "
            "Bookings cannot be confirmed through this chatbot.",
            True,
        )

    return message, False


def _sanitize_no_confirmation(resp: Response) -> Response:
    cleaned, changed = _sanitize_no_confirmation_message(resp.message)

    if changed:
        resp.message = cleaned
        resp.complete = False
        resp.escalate = False
        resp.agent = "booking" if resp.agent == "escalation" else resp.agent
        resp.debug["confirmation_removed"] = True

    return resp


def _validate_duration_against_time(form: dict) -> str | None:
    requested_time = form.get("requested_time")
    hours_needed = form.get("hours_needed")

    if requested_time is None or hours_needed is None:
        return None

    try:
        start = date_parser.parse(str(requested_time)).time()
        hours = float(hours_needed)
    except Exception:
        return None

    start_decimal = start.hour + (start.minute / 60)
    end_decimal = start_decimal + hours

    if end_decimal > CONFIG.hours.end_hour:
        latest_start_decimal = CONFIG.hours.end_hour - hours
        latest_hour = int(latest_start_decimal)
        latest_minute = int(round((latest_start_decimal - latest_hour) * 60))

        latest_label = _time_label(f"{latest_hour:02d}:{latest_minute:02d}")

        return (
            "The requested duration does not fit the start time. "
            f"A {_format_hours(hours)}-hour session starting at {_time_label(str(requested_time))} "
            f"would end after {CONFIG.hours.end_label}. "
            f"For a {_format_hours(hours)}-hour session, the latest start time is {latest_label}. "
            "Please choose an earlier start time or fewer hours."
        )

    return None


def _invalid_datetime_response(clf: dict, form: dict) -> Response | None:
    messages = []

    if clf.get("date_in_past") and clf.get("detected_date"):
        messages.append(
            f"The requested date {clf['detected_date']} has already passed. "
            "Please choose a future date."
        )

    if clf.get("date_too_far") and clf.get("detected_date"):
        messages.append(
            f"The requested date {clf['detected_date']} is too far in advance. "
            f"We only accept bookings up to 6 months ahead, until {clf.get('max_booking_date')}. "
            "Please choose a date within the next 6 months."
        )

    if clf.get("time_outside_hours") and clf.get("detected_time"):
        messages.append(
            f"The requested time {_time_label(clf['detected_time'])} is outside our service hours. "
            f"Our service hours are {CONFIG.hours.start_label} to {CONFIG.hours.end_label}. "
            "Please choose a time within this window."
        )

    if clf.get("time_too_late") and clf.get("detected_time"):
        messages.append(
            f"The requested start time {_time_label(clf['detected_time'])} is too late for the minimum booking. "
            f"The minimum booking is {CONFIG.booking.min_hours:.0f} hours, "
            f"and all sessions must finish by {CONFIG.hours.end_label}. "
            f"So the latest start time for a {CONFIG.booking.min_hours:.0f}-hour session is "
            f"{CONFIG.hours.latest_start(CONFIG.booking.min_hours)}. "
            "Please choose an earlier start time."
        )

    if not messages:
        return None

    return Response(
        message=" ".join(messages),
        agent="booking",
        form=form,
        complete=False,
        escalate=False,
        debug={"blocked_by_python_validation": True},
    )


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
The Python app handles all date/time rules.

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

    max_booking_date = today + relativedelta(months=6)

    start_hour = CONFIG.hours.start_hour
    end_hour = CONFIG.hours.end_hour
    min_hours = float(CONFIG.booking.min_hours)
    latest_start_decimal = end_hour - min_hours

    delta = (detected_date - today).days if detected_date else None
    date_text_lower = date_text.lower().strip() if date_text else ""

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
        "max_booking_date": max_booking_date.isoformat(),
    }

    clf["date_in_past"] = detected_date is not None and delta is not None and delta < 0

    clf["date_too_far"] = (
        detected_date is not None
        and detected_date > max_booking_date
    )

    clf["is_emergency"] = (
        detected_date is not None
        and date_text_lower in {"today", "tomorrow"}
        and not clf["date_in_past"]
        and not clf["date_too_far"]
    )

    clf["time_outside_hours"] = (
        detected_time is not None
        and (
            detected_time.hour < start_hour
            or detected_time.hour >= end_hour
        )
    )

    if detected_time is not None:
        start_decimal = detected_time.hour + (detected_time.minute / 60)
        clf["time_too_late"] = (
            not clf["time_outside_hours"]
            and start_decimal > latest_start_decimal
        )
    else:
        clf["time_too_late"] = False

    print("DEBUG CLASSIFIER:", json.dumps(clf, indent=2, default=str), flush=True)
    log.warning("DEBUG CLASSIFIER: %s", json.dumps(clf, default=str))

    clf["classifier_debug_generated"] = True

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
        "escalate": "false always",
    },
    indent=2,
)


def run_booking(
    message: str,
    history: list[dict],
    form: dict,
    clf: dict | None = None,
) -> Response:
    filled = {k: v for k, v in form.items() if v is not None}
    missing = [k for k in _BOOKING_REQUIRED if form.get(k) is None]

    system = f"""You are the booking assistant for a cleaning service.

{CONFIG.rules_for_agents()}

FORM STATE:
Required fields: {_BOOKING_REQUIRED}
Already collected: {json.dumps(filled, default=str)}
Still missing: {missing}

CURRENT CLASSIFIER CONTEXT:
{json.dumps(clf or {}, indent=2, default=str)}

RULES:
- Collect required booking details one or two fields at a time.
- Never confirm or commit to a booking.
- Never say "confirmed".
- A salesperson must confirm all bookings.
- If the user gives a valid date and time, continue collecting missing fields.
- 6pm is a VALID start time for a 3-hour session because sessions end by 9pm.
- If requested_time plus hours_needed exceeds {CONFIG.hours.end_label}, do not complete the booking.
- Never mark is_complete=true until requested_time plus hours_needed ends by {CONFIG.hours.end_label}.
- Always set escalate=false.
- If all required fields are collected and time/duration is valid, say:
  "Thank you! I've noted everything down. Our salesperson will contact you shortly to confirm."

Return ONLY valid JSON:
{_BOOKING_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    # ------------------------------------------------------------------
    # SAFETY OVERRIDE
    # 6pm is valid if duration allows finishing by 9pm
    # ------------------------------------------------------------------

    if (
        clf
        and clf.get("detected_time") == "18:00"
        and not clf.get("time_outside_hours")
        and not clf.get("time_too_late")
    ):
        raw["escalate"] = False

    new_collected = raw.get("collected") or {}

    updated_form = {
        **form,
        **{k: v for k, v in new_collected.items() if v is not None},
    }

    duration_error = _validate_duration_against_time(updated_form)

    if duration_error:
        return Response(
            message=duration_error,
            agent="booking",
            form=updated_form,
            complete=False,
            escalate=False,
            debug={
                "duration_validation_failed": True,
                "requested_time": updated_form.get("requested_time"),
                "hours_needed": updated_form.get("hours_needed"),
            },
        )

    message_text = _get(
        raw,
        "message",
        "Sorry, I didn't catch that — could you repeat it?",
    )

    cleaned_message, confirmation_removed = _sanitize_no_confirmation_message(message_text)

    resp = Response(
        message=cleaned_message,
        agent="booking",
        form=updated_form,
        complete=bool(raw.get("is_complete", False)) and not confirmation_removed,
        escalate=False,
        debug={
            "next_field": raw.get("next_field"),
            "confirmation_removed": confirmation_removed,
        },
    )

    return resp

def run_booking(
    message: str,
    history: list[dict],
    form: dict,
    clf: dict | None = None,
) -> Response:
    filled = {k: v for k, v in form.items() if v is not None}
    missing = [k for k in _BOOKING_REQUIRED if form.get(k) is None]

    system = f"""You are the booking assistant for a cleaning service.

{CONFIG.rules_for_agents()}

FORM STATE:
Required fields: {_BOOKING_REQUIRED}
Already collected: {json.dumps(filled, default=str)}
Still missing: {missing}

CURRENT CLASSIFIER CONTEXT:
{json.dumps(clf or {}, indent=2, default=str)}

RULES:
- Extract any booking details the user provides.
- Never confirm or commit to a booking.
- Never say "confirmed".
- A salesperson must confirm all bookings.
- Always set escalate=false.
- Return collected fields in JSON.

Return ONLY valid JSON:
{_BOOKING_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    new_collected = raw.get("collected") or {}

    updated_form = {
        **form,
        **{k: v for k, v in new_collected.items() if v is not None},
    }

    duration_error = _validate_duration_against_time(updated_form)

    if duration_error:
        return Response(
            message=duration_error,
            agent="booking",
            form=updated_form,
            complete=False,
            escalate=False,
            debug={
                "duration_validation_failed": True,
                "requested_time": updated_form.get("requested_time"),
                "hours_needed": updated_form.get("hours_needed"),
            },
        )

    missing_after = [k for k in _BOOKING_REQUIRED if updated_form.get(k) is None]

    question_map = {
        "customer_name": "Please provide your name.",
        "address": "Please provide the cleaning address.",
        "requested_date": "What date would you like the cleaning?",
        "requested_time": "What start time would you prefer?",
        "hours_needed": "How many hours do you need?",
        "has_pets": "Do you have any pets at the property?",
    }

    if missing_after:
        next_field = missing_after[0]

        return Response(
            message=question_map[next_field],
            agent="booking",
            form=updated_form,
            complete=False,
            escalate=False,
            debug={
                "next_field": next_field,
                "missing_fields": missing_after,
                "forced_booking_flow": True,
            },
        )

    return Response(
        message=(
            "Thank you! I’ve noted everything down. "
            "Our salesperson will contact you shortly to confirm availability."
        ),
        agent="booking",
        form=updated_form,
        complete=True,
        escalate=False,
        debug={
            "next_field": None,
            "missing_fields": [],
            "forced_booking_flow": True,
        },
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
- Never confirm bookings.
- Never say "confirmed".
- Emergency booking today/tomorrow: say a salesperson will WhatsApp them shortly to check availability.
- Complaint or negative feedback: empathise and promise human follow-up. Set urgency=high.
- Unanswerable FAQ: politely say you will connect them with the team. Set urgency=routine.

Return ONLY valid JSON:
{_ESCALATION_SCHEMA}
"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    resp = Response(
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

    return _sanitize_no_confirmation(resp)


def process_message(
    message: str,
    history: list[dict],
    form: dict,
) -> Response:
    try:
        clf = classify(message, history)
        booking_keywords = [
        "book",
        "booking",
        "cleaning",
        "cleaner",
        "hour",
        "hours",
        "am",
        "pm",
    ]

    if any(word in message.lower() for word in booking_keywords):
    intent = "booking"
        intent = clf["intent"]

        invalid_resp = _invalid_datetime_response(clf, form)

        if invalid_resp is not None:
            invalid_resp.intent = intent
            invalid_resp.debug.update(clf)
            return invalid_resp

        booking_started = any(
            form.get(k)
            for k in [
                "customer_name",
                "address",
                "requested_date",
                "requested_time",
            ]
        )

        if intent == "booking" or (booking_started and intent != "feedback"):
            resp = run_booking(message, history, form, clf=clf)

        elif intent == "escalation":
            resp = run_escalation(message, history, clf)

        elif intent in ("faq", "out_of_scope"):
            resp = run_faq(message, history)

            if resp.escalate:
                resp = run_escalation(message, history, clf)

        elif intent == "feedback":
            resp = run_escalation(message, history, clf)

        else:
            resp = run_faq(message, history)

        resp.intent = intent

        # ------------------------------------------------------------------
        # Never allow booking flow to become escalation
        # ------------------------------------------------------------------

        if resp.agent == "booking":
            resp.escalate = False

        if intent == "booking":
            resp.agent = "booking"
            resp.escalate = False

        resp.debug.update(
            {
                "intent": intent,
                "agent": resp.agent,
                "sentiment": clf.get("sentiment"),
                "urgency": clf.get("urgency"),
                "confidence": clf.get("confidence"),
                "reasoning": clf.get("reasoning"),
                "date_text": clf.get("date_text"),
                "time_text": clf.get("time_text"),
                "detected_date": clf.get("detected_date"),
                "detected_time": clf.get("detected_time"),
                "max_booking_date": clf.get("max_booking_date"),
                "is_emergency": clf.get("is_emergency"),
                "date_in_past": clf.get("date_in_past"),
                "date_too_far": clf.get("date_too_far"),
                "time_outside_hours": clf.get("time_outside_hours"),
                "time_too_late": clf.get("time_too_late"),
                "form": resp.form,
            }
        )

        if resp.agent == "booking":
            resp.escalate = False

        return _sanitize_no_confirmation(resp)

    except Exception as exc:
        log.error("process_message crashed: %s", exc, exc_info=True)

        return Response(
            message="I'm sorry, I ran into a technical issue. Could you try again?",
            agent="system",
            form=form,
        )
