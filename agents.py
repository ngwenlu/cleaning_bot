"""
agents.py – All agent logic in one place.

Business rules come entirely from knowledge_base.CONFIG.
To change any rule, edit knowledge_base.py only.

Architecture:
classify()         -> detects intent, urgency, date/time flags
run_booking()      -> collects booking form fields
run_faq()          -> answers from knowledge base
run_escalation()   -> human handoff
process_message()  -> orchestrator entry point; returns Response, never raises
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

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
    """Universal return type from any agent. Never raises; always has a message."""

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
    """Call the LLM, parse JSON, return {} on any failure."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "").strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0].strip()

        return json.loads(content) if content else {}

    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return {}


def _get(raw: dict, key: str, fallback):
    """Safe dict access with a fallback for missing or None values."""
    value = raw.get(key)
    return value if value is not None else fallback


_CLASSIFY_SCHEMA = json.dumps(
    {
        "intent": "booking | faq | escalation | feedback | out_of_scope",
        "sentiment": "positive | neutral | negative | urgent",
        "urgency": "routine | high | critical",
        "confidence": "float 0.0-1.0",
        "reasoning": "one sentence",
        "detected_date": "YYYY-MM-DD if user mentioned a date, else null",
        "detected_time": "HH:MM if user mentioned a time, else null",
        "is_emergency": "true if booking date is today or tomorrow",
        "date_seems_wrong": "true if future date for complaint OR past date for booking",
        "time_outside_hours": "true if time is outside service hours",
    },
    indent=2,
)


def classify(message: str, history: list[dict]) -> dict:
    today = date.today()

    system = f"""
You are the intent and sentiment classifier for a cleaning service chatbot.

Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

Classify the latest user message.

Return ONLY valid JSON:
{_CLASSIFY_SCHEMA}
"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
    )

    return {
        "intent": raw.get("intent", "faq"),
        "sentiment": raw.get("sentiment", "neutral"),
        "urgency": raw.get("urgency", "routine"),
        "confidence": raw.get("confidence", 0.5),
        "reasoning": raw.get("reasoning", ""),
        "detected_date": raw.get("detected_date"),
        "detected_time": raw.get("detected_time"),
        "is_emergency": bool(raw.get("is_emergency", False)),
        "date_seems_wrong": bool(raw.get("date_seems_wrong", False)),
        "time_outside_hours": bool(raw.get("time_outside_hours", False)),
    }


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
            "has_pets": "boolean or null, where null means not yet asked",
            "contact": "phone or email if the customer volunteers it, else null",
            "notes": "any extra context, else null",
        },
        "is_complete": "true when ALL 6 required fields are filled",
        "next_field": "name of next missing required field, or null",
        "escalate": "true ONLY if the booking is for today or tomorrow",
    },
    indent=2,
)


def run_booking(
    message: str,
    history: list[dict],
    form: dict,
) -> Response:
    today = date.today()

    filled = {
        key: value
        for key, value in form.items()
        if value is not None
    }

    missing = [
        key
        for key in _BOOKING_REQUIRED
        if form.get(key) is None
    ]

    system = f"""
You are the booking assistant for a cleaning service.

Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

TASK:
Collect the 6 required booking fields, one or two at a time, conversationally.

Required fields:
{_BOOKING_REQUIRED}

Already collected. Do not re-ask:
{json.dumps(filled, default=str)}

Still missing:
{missing}

RULES:
- Compute relative dates yourself. For example, "next Saturday" should be calculated and confirmed.
- Reject and re-ask for past dates, impossible dates, and out-of-hours times.
- For out-of-hours times, explain the allowed window and ask again. Do not crash.
- Warn if start_time + hours_needed would exceed closing time, and suggest an earlier start.
- NEVER confirm or commit to a booking. A salesperson will follow up.
- When all 6 fields are collected, say:
  "Thank you! I've noted everything down. Our salesperson will contact you shortly to confirm."
- Set escalate=true only if the date is today or tomorrow.

Return ONLY valid JSON:
{_BOOKING_SCHEMA}
"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
    )

    new_collected = raw.get("collected") or {}

    updated_form = {
        **form,
        **{
            key: value
            for key, value in new_collected.items()
            if value is not None
        },
    }

    return Response(
        message=_get(
            raw,
            "message",
            "Sorry, I didn't catch that. Could you repeat it?",
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


def run_faq(
    message: str,
    history: list[dict],
) -> Response:
    system = f"""
You are the FAQ assistant for a cleaning service in Singapore.

Answer ONLY using the knowledge base below.
If the question is not covered, set answered=false.

{CONFIG.rules_for_agents()}

=== KNOWLEDGE BASE ===
{get_full_kb_text()}

Return ONLY valid JSON:
{_FAQ_SCHEMA}
"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
    )

    sources = raw.get("sources", [])

    if isinstance(sources, str):
        try:
            import ast

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
        "message_to_user": "Empathetic reply. Emergency: reassure. Complaint: acknowledge.",
        "summary": "2-3 sentences for the salesperson covering what happened and what the customer needs.",
        "urgency": "routine | high | critical",
    },
    indent=2,
)


def run_escalation(
    message: str,
    history: list[dict],
    context: dict | None = None,
) -> Response:
    system = f"""
You are the escalation handler for a cleaning service chatbot.

You are called for:
- emergency bookings
- complaints
- unanswerable FAQs
- explicit human requests

{CONFIG.rules_for_agents()}

Salesperson contacts:
WhatsApp: {SALESPERSON_WHATSAPP}
Email: {SALESPERSON_EMAIL}

Context from classifier:
{json.dumps(context or {}, default=str)}

RULES:
- Emergency booking today or tomorrow: reassure the customer and say the team will WhatsApp them shortly.
- Complaint: empathise, do not argue, do not offer refunds, promise human follow-up.
- Unanswerable FAQ: say you'll connect them with the team.
- Out-of-hours request: explain the service window and ask for a valid time. Do NOT escalate as urgent.
- Date/time mismatch where date_seems_wrong=true: point out the inconsistency and ask the customer to clarify.

Return ONLY valid JSON:
{_ESCALATION_SCHEMA}
"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
    )

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
    """
    Process one user message end-to-end.
    Never raises. Always returns a Response with a safe message.
    """
    try:
        classifier_result = classify(message, history)

        intent = classifier_result["intent"]
        is_emergency = classifier_result["is_emergency"]
        date_wrong = classifier_result["date_seems_wrong"]

        booking_started = any(
            form.get(key)
            for key in [
                "customer_name",
                "address",
                "requested_date",
                "requested_time",
            ]
        )

        if is_emergency or intent == "escalation":
            response = run_escalation(
                message,
                history,
                classifier_result,
            )

        elif intent == "booking" or (
            booking_started and intent not in ("feedback",)
        ):
            response = run_booking(
                message,
                history,
                form,
            )

            if response.escalate:
                response = run_escalation(
                    message,
                    history,
                    {
                        **classifier_result,
                        "form": response.form,
                    },
                )
                response.form = form

        elif intent in ("faq", "out_of_scope"):
            response = run_faq(
                message,
                history,
            )

            if response.escalate:
                response = run_escalation(
                    message,
                    history,
                    classifier_result,
                )

        elif date_wrong:
            response = run_escalation(
                message,
                history,
                classifier_result,
            )

        else:
            response = run_faq(
                message,
                history,
            )

        response.intent = intent

        response.debug.update(
            {
                "intent": intent,
                "sentiment": classifier_result.get("sentiment"),
                "urgency": classifier_result.get("urgency"),
                "confidence": classifier_result.get("confidence"),
                "reasoning": classifier_result.get("reasoning"),
                "is_emergency": is_emergency,
            }
        )

        return response

    except Exception as exc:
        log.error("process_message crashed: %s", exc, exc_info=True)

        return Response(
            message="I'm sorry, I ran into a technical issue. Could you try again?",
            agent="system",
            form=form,
        )