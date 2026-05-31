"""
agents.py – All agent logic in one place.
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
    """Universal return type from any agent. Never raises – always has a message."""

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
    """Safe dict access with a fallback for missing/None values."""
    value = raw.get(key)
    return value if value is not None else fallback


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

    system = f"""You are the intent and sentiment classifier for a cleaning service chatbot.
Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

Classify the latest user message. Return ONLY valid JSON:
{_CLASSIFY_SCHEMA}"""

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
            "has_pets": "boolean or null",
            "contact": "phone or email if the customer volunteers it, else null",
            "notes": "any extra context, else null",
        },
        "is_complete": "true when ALL 6 required fields are filled",
        "next_field": "name of next missing required field, or null",
        "escalate": "true ONLY if the booking is for today or tomorrow",
    },
    indent=2,
)


def run_booking(message: str, history: list[dict], form: dict) -> Response:
    today = date.today()
    filled = {k: v for k, v in form.items() if v is not None}
    missing = [k for k in _BOOKING_REQUIRED if form.get(k) is None]

    system = f"""You are the booking assistant for a cleaning service.

Today: {today.isoformat()} ({today.strftime("%A")})

{CONFIG.rules_for_agents()}

TASK: Collect the 6 required booking fields, one or two at a time, conversationally.

Required fields: {_BOOKING_REQUIRED}
Already collected, do not re-ask: {json.dumps(filled, default=str)}
Still missing: {missing}

RULES:
- Compute relative dates yourself.
- Reject and re-ask for past dates, impossible dates, and out-of-hours times.
- For out-of-hours times, explain the allowed window and ask again.
- Warn if start_time + hours_needed would exceed closing time.
- NEVER confirm or commit to a booking.
- When all 6 fields are collected, say:
  "Thank you! I’ve noted everything down. Our salesperson will contact you shortly to confirm."
- Set escalate=true only if the date is today or tomorrow.

Return ONLY valid JSON:
{_BOOKING_SCHEMA}"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    new_collected = raw.get("collected") or {}
    updated_form = {
        **form,
        **{k: v for k, v in new_collected.items() if v is not None},
    }

    return Response(
        message=_get(raw, "message", "Sorry, I didn't catch that — could you repeat it?"),
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
Answer ONLY using the knowledge base below. If the question is not covered, set answered=false.

{CONFIG.rules_for_agents()}

# === KNOWLEDGE BASE ===
{get_full_kb_text()}

Return ONLY valid JSON:
{_FAQ_SCHEMA}"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

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
        message=_get(raw, "message", "Let me connect you with our team for more details."),
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
    system = f"""You are the escalation handler for a cleaning service chatbot.
You are called for emergency bookings, complaints, unanswerable FAQs, and explicit human requests.

{CONFIG.rules_for_agents()}

Salesperson contacts:
WhatsApp: {SALESPERSON_WHATSAPP}
Email: {SALESPERSON_EMAIL}

Context from classifier: {json.dumps(context or {}, default=str)}

RULES:
- Emergency booking today/tomorrow: reassure the customer and say the team will WhatsApp them shortly.
- Complaint: empathise, do not argue or offer refunds, promise human follow-up.
- Unanswerable FAQ: say you’ll connect them with the team.
- Out-of-hours request: explain the service window and ask for a valid time. Do NOT escalate as urgent.
- Date/time mismatch: point out the inconsistency and ask the customer to clarify.

Return ONLY valid JSON:
{_ESCALATION_SCHEMA}"""

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
    """
    Process one user message end-to-end.
    NEVER raises – always returns a Response with a safe message.
    """

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

        if intent == "escalation":
            resp = run_escalation(message, history, clf)

        elif intent == "booking" or (booking_started and intent != "feedback"):
            resp = run_booking(message, history, form)

            if resp.escalate:
                saved_form = resp.form
                resp = run_escalation(message, history, {**clf, "form": saved_form})
                resp.form = saved_form

        elif intent in ("faq", "out_of_scope"):
            resp = run_faq(message, history)

            if resp.escalate:
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