"""
agents.py -- All agent logic in one place.

Business rules come entirely from knowledge_base.CONFIG.
To change any rule, edit knowledge_base.py only.

Architecture:
  classify()        -> extracts intent + date/time strings (LLM),
                       computes all flags via Python datetime arithmetic
  run_booking()     -> collects booking form fields
  run_faq()         -> answers from knowledge base
  run_escalation()  -> human handoff
  process_message() -> orchestrator; returns Response, never raises
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL,
    MAX_TOKENS, MODEL, SALESPERSON_EMAIL, SALESPERSON_WHATSAPP, client,
)
from knowledge_base import CONFIG, get_full_kb_text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response -- the only data type app.py needs
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """Universal return type. Never raises -- always has a message."""
    message:  str
    agent:    str  = "assistant"
    form:     dict = field(default_factory=dict)
    complete: bool = False
    escalate: bool = False
    intent:   str  = "unknown"
    debug:    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core LLM helper
# ---------------------------------------------------------------------------

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
    v = raw.get(key)
    return v if v is not None else fallback


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

_CLASSIFY_SCHEMA = json.dumps({
    "intent":        "booking | faq | escalation | feedback | out_of_scope",
    "sentiment":     "positive | neutral | negative | urgent",
    "urgency":       "routine | high | critical",
    "confidence":    "float 0.0-1.0",
    "reasoning":     "one sentence",
    "detected_date": "ISO date YYYY-MM-DD if the user mentioned any date, else null",
    "detected_time": "24hr time HH:MM if the user mentioned any time, else null",
}, indent=2)


def classify(message: str, history: list[dict]) -> dict:
    today = date.today()

    system = f"""<system>
  <role>
    Intent and sentiment classifier for a cleaning service chatbot.
    You extract intent and any mentioned date/time strings. You do NOT compute
    whether dates are valid -- that is handled by the application in Python.
  </role>

  <context>
    <today>{today.isoformat()} ({today.strftime('%A')})</today>
  </context>

  <business_rules>
{CONFIG.rules_for_agents()}
  </business_rules>

  <intent_definitions>
    <intent name="booking">User wants to schedule a cleaning session.</intent>
    <intent name="faq">User has a question about the service, pricing, hours, etc.</intent>
    <intent name="escalation">User wants to speak to a human or situation needs urgent attention.</intent>
    <intent name="feedback">User is giving feedback or complaining about a past session.</intent>
    <intent name="out_of_scope">Unrelated to the cleaning business.</intent>
  </intent_definitions>

  <date_time_extraction>
    Extract detected_date and detected_time as plain strings only.
    Do NOT judge whether they are valid, past, future, or in-hours -- just extract.
    For relative dates, resolve to an ISO date using today as the anchor.

    <examples>
      <!-- today is {today.isoformat()}, a {today.strftime('%A')} -->
      <example>
        <input>can i book today?</input>
        <detected_date>{today.isoformat()}</detected_date>
      </example>
      <example>
        <input>tomorrow morning</input>
        <detected_date>{(today + __import__('datetime').timedelta(days=1)).isoformat()}</detected_date>
      </example>
      <example>
        <input>next saturday</input>
        <detected_date>{(today + __import__('datetime').timedelta(days=(5 - today.weekday()) % 7 or 7)).isoformat()}</detected_date>
      </example>
      <example>
        <input>this sunday</input>
        <detected_date>{(today + __import__('datetime').timedelta(days=(6 - today.weekday()) % 7 or 7)).isoformat()}</detected_date>
      </example>
      <example>
        <input>next week friday</input>
        <detected_date>{(today + __import__('datetime').timedelta(days=(4 - today.weekday()) % 7 + 7)).isoformat()}</detected_date>
      </example>
    </examples>

    If no date is mentioned, return null.
</date_time_extraction>

  <output_format>
    Return ONLY valid JSON with no preamble:
    {_CLASSIFY_SCHEMA}
  </output_format>
</system>"""

    raw = _llm(
        system,
        history + [{"role": "user", "content": message}],
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
    )

    from datetime import date as _date, time as _time

    clf = {
        "intent":        raw.get("intent", "faq"),
        "sentiment":     raw.get("sentiment", "neutral"),
        "urgency":       raw.get("urgency", "routine"),
        "confidence":    raw.get("confidence", 0.5),
        "reasoning":     raw.get("reasoning", ""),
        "detected_date": raw.get("detected_date"),
        "detected_time": raw.get("detected_time"),
    }

    # All datetime arithmetic in Python -- never trust LLM for this
    today = _date.today()
    ew     = CONFIG.booking.emergency_window_days
    sh     = CONFIG.hours.start_hour
    eh     = CONFIG.hours.end_hour
    min_h  = int(CONFIG.booking.min_hours)
    max_adv = CONFIG.booking.max_advance_days

    detected_date = None
    if clf["detected_date"]:
        try:
            detected_date = _date.fromisoformat(clf["detected_date"])
        except (ValueError, TypeError):
            pass

    detected_time_obj = None
    if clf["detected_time"]:
        try:
            detected_time_obj = _time.fromisoformat(clf["detected_time"])
        except (ValueError, TypeError):
            pass

    delta = (detected_date - today).days if detected_date else None

    clf["is_emergency"]       = detected_date is not None and 0 <= delta <= ew
    clf["date_in_past"]       = detected_date is not None and delta < 0
    clf["date_too_far"]       = detected_date is not None and delta is not None and delta > max_adv
    clf["time_outside_hours"] = (detected_time_obj is not None
                                 and (detected_time_obj.hour < sh
                                      or detected_time_obj.hour >= eh))
    clf["time_too_late"]      = (detected_time_obj is not None
                                 and not clf["time_outside_hours"]
                                 and detected_time_obj.hour > (eh - min_h))
    # e.g. eh=21, min_h=3 -> threshold=18 -> hour>18 means 7pm+ is too late
    # 6pm (hour=18) is valid: 18+3=21=9pm exactly at closing
    clf["date_seems_wrong"]   = (
        (clf["date_in_past"] and clf["intent"] == "booking")
        or (detected_date is not None and delta is not None
            and delta > ew and clf["intent"] in ("feedback", "escalation"))
    )
    return clf


# ---------------------------------------------------------------------------
# Booking agent
# ---------------------------------------------------------------------------

_BOOKING_REQUIRED = ["customer_name", "address", "requested_date",
                     "requested_time", "hours_needed", "has_pets"]

_BOOKING_SCHEMA = json.dumps({
    "message":  "Conversational reply to the user",
    "collected": {
        "customer_name":  "string or null",
        "address":        "string or null",
        "requested_date": "YYYY-MM-DD or null",
        "requested_time": "HH:MM or null",
        "hours_needed":   "number or null",
        "has_pets":       "boolean or null (null = not yet asked)",
        "contact":        "phone or email if volunteered, else null",
        "notes":          "any extra context, else null",
    },
    "is_complete": "true when ALL 6 required fields are filled",
    "next_field":  "name of next missing required field, or null",
    "escalate":    "true ONLY if the booking is for today or tomorrow",
}, indent=2)


def run_booking(message: str, history: list[dict], form: dict, clf: dict | None = None) -> Response:
    today = date.today()
    filled  = {k: v for k, v in form.items() if v is not None}
    missing = [k for k in _BOOKING_REQUIRED if form.get(k) is None]

    # Build Python-verified alerts from datetime flags
    alerts = []
    if clf:
        if clf.get("date_in_past") and clf.get("detected_date"):
            alerts.append(
                f"The date {clf['detected_date']} has already passed "
                f"(today is {today.isoformat()}). "
                f"Tell the customer this date is in the past and ask for a future date."
            )
        if clf.get("date_too_far") and clf.get("detected_date"):
            alerts.append(
                f"The booking date {clf['detected_date']} is too far in advance. "
                f"The maximum is {CONFIG.booking.max_advance_days // 30} months ahead. "
                f"Tell the customer their booking date is too far in advance and that "
                f"the maximum is 6 months ahead."
            )
        if clf.get("time_too_late") and clf.get("detected_time"):
            alerts.append(
                f"The start time {clf['detected_time']} is too late. "
                f"Even the minimum {CONFIG.booking.min_hours:.0f}-hour session would run "
                f"past {CONFIG.hours.end_label} when cleaners stop service. "
                f"The latest start time is {CONFIG.hours.latest_start(CONFIG.booking.min_hours)}. "
                f"Tell the customer the latest start time is "
                f"{CONFIG.hours.latest_start(CONFIG.booking.min_hours)}."
            )
        if clf.get("time_outside_hours") and clf.get("detected_time"):
            alerts.append(
                f"The time {clf['detected_time']} is outside service hours "
                f"({CONFIG.hours.start_label}-{CONFIG.hours.end_label}). "
                f"Explain the service hours and ask for a valid time."
            )
        if clf.get("is_emergency") and clf.get("detected_date"):
            alerts.append(
                f"The date {clf['detected_date']} is today or tomorrow -- "
                f"this is an emergency booking. Set escalate=true."
            )

    # Check overrun using form data: start_time + hours_needed > closing time
    # This runs whenever both values are available (either from form or just collected)
    check_time  = (clf.get("detected_time") or form.get("requested_time")) if clf else form.get("requested_time")
    check_hours = form.get("hours_needed")
    if check_time and check_hours:
        try:
            from datetime import datetime as _dt
            t = _dt.strptime(str(check_time)[:5], "%H:%M")
            end_hour_float = t.hour + t.minute / 60 + float(check_hours)
            if end_hour_float > CONFIG.hours.end_hour:
                end_h = int(end_hour_float)
                end_m = int((end_hour_float - end_h) * 60)
                end_label = f"{end_h % 12 or 12}:{end_m:02d}{'am' if end_h < 12 else 'pm'}"
                # Only add if not already flagged by time_too_late
                if not (clf and clf.get("time_too_late")):
                    alerts.append(
                        f"The time slot {check_time} for {check_hours} hours would end at "
                        f"{end_label}, which is after {CONFIG.hours.end_label} when cleaners "
                        f"stop service. Tell the customer their time slot would overrun 9pm "
                        f"and ask them to choose an earlier start time or fewer hours."
                    )
        except (ValueError, TypeError):
            pass

    alerts_xml = ""
    if alerts:
        items = "\n".join(f"    <alert>{a}</alert>" for a in alerts)
        alerts_xml = f"\n  <datetime_alerts>\n{items}\n  </datetime_alerts>"

    system = f"""<system>
  <role>
    Booking assistant for a cleaning service.
    Collect the required booking details conversationally, one or two fields at a time.
    You do NOT confirm or commit to any booking -- a salesperson will follow up.
  </role>

  <context>
    <today>{today.isoformat()} ({today.strftime('%A')})</today>
  </context>

  <business_rules>
{CONFIG.rules_for_agents()}
  </business_rules>
{alerts_xml}
  <form_state>
    <required_fields>{_BOOKING_REQUIRED}</required_fields>
    <already_collected>{json.dumps(filled, default=str)}</already_collected>
    <still_missing>{missing}</still_missing>
  </form_state>

  <rules>
    <rule>Compute relative dates yourself ("next Saturday" -> calculate from today and confirm naturally).</rule>
    <rule>Valid start times are 9am up to and INCLUDING 6pm. Only reject 7pm (19pm) or later.</rule>
    <rule>6pm IS valid: 6pm + 3h minimum = 9pm exactly. Do NOT reject 6pm.</rule>
    <rule>After collecting both requested_time and hours_needed, calculate: end_time = start_time + hours_needed. If end_time is strictly AFTER 9pm, tell the customer their time slot would overrun 9pm when cleaners stop service, and ask them to choose an earlier start time or fewer hours.</rule>
    <rule>When all 6 required fields are filled, say: "Thank you! I've noted everything down. Our salesperson will contact you shortly to confirm." Then set is_complete=true.</rule>
    <rule>Set escalate=true only if the booking date is today or tomorrow.</rule>
    <rule>NEVER say the booking is confirmed. NEVER commit to a specific cleaner or time slot.</rule>
    <rule>If datetime_alerts are present above, address them immediately before asking for the next field.</rule>
  </rules>

  <output_format>
    Return ONLY valid JSON with no preamble:
    {_BOOKING_SCHEMA}
  </output_format>
</system>"""

    raw = _llm(system, history + [{"role": "user", "content": message}])

    new_collected = raw.get("collected") or {}
    updated_form = {
        **form,
        **{k: v for k, v in new_collected.items() if v is not None},
    }

    return Response(
        message=_get(raw, "message", "Sorry, I didn't catch that -- could you repeat it?"),
        agent="booking",
        form=updated_form,
        complete=bool(raw.get("is_complete", False)),
        escalate=bool(raw.get("escalate", False)),
        debug={"next_field": raw.get("next_field")},
    )


# ---------------------------------------------------------------------------
# FAQ agent
# ---------------------------------------------------------------------------

_FAQ_SCHEMA = json.dumps({
    "message":  "Answer in plain conversational English",
    "sources":  ["list", "of", "kb", "keys", "used"],
    "answered": "true if the KB covers the question, false otherwise",
}, indent=2)


def run_faq(message: str, history: list[dict]) -> Response:
    system = f"""<system>
  <role>
    FAQ assistant for a cleaning service in Singapore.
    Answer questions using ONLY the knowledge base provided.
    If the question is not covered by the knowledge base, set answered=false.
  </role>

  <business_rules>
{CONFIG.rules_for_agents()}
  </business_rules>

  <knowledge_base>
{get_full_kb_text()}
  </knowledge_base>

  <rules>
    <rule>Answer only from the knowledge base. Do not invent information.</rule>
    <rule>If the question is not covered, set answered=false and politely say you will connect them with the team.</rule>
    <rule>Keep answers friendly and concise.</rule>
  </rules>

  <output_format>
    Return ONLY valid JSON with no preamble:
    {_FAQ_SCHEMA}
  </output_format>
</system>"""

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


# ---------------------------------------------------------------------------
# Escalation agent
# ---------------------------------------------------------------------------

_ESCALATION_SCHEMA = json.dumps({
    "message_to_user": "Empathetic reply. Emergency: reassure. Complaint: acknowledge.",
    "summary":         "2-3 sentences for the salesperson: what happened, what the customer needs.",
    "urgency":         "routine | high | critical",
}, indent=2)


def run_escalation(
    message: str,
    history: list[dict],
    context: dict | None = None,
) -> Response:
    system = f"""<system>
  <role>
    Escalation handler for a cleaning service chatbot.
    You handle: emergency bookings, complaints, unanswerable FAQs, explicit human requests.
  </role>

  <business_rules>
{CONFIG.rules_for_agents()}
  </business_rules>

  <salesperson_contacts>
    <whatsapp>{SALESPERSON_WHATSAPP}</whatsapp>
    <email>{SALESPERSON_EMAIL}</email>
  </salesperson_contacts>

  <classifier_context>
    {json.dumps(context or {}, indent=4, default=str)}
  </classifier_context>

  <rules>
    <rule>Emergency booking (today/tomorrow): reassure the customer and say the team will WhatsApp them shortly. Set urgency=critical.</rule>
    <rule>Complaint or negative feedback: empathise, do not argue or offer refunds, promise a human will follow up. Set urgency=high.</rule>
    <rule>Unanswerable FAQ: politely say you will connect them with the team. Set urgency=routine.</rule>
    <rule>Out-of-hours or invalid time: explain the service window and ask for a valid time. Do NOT treat as urgent.</rule>
    <rule>Date mismatch (date_seems_wrong=true): point out the inconsistency clearly and ask the customer to confirm the correct date.</rule>
    <rule>Write summary for the salesperson in plain English covering: what the customer wants, any details collected, and the sentiment.</rule>
  </rules>

  <output_format>
    Return ONLY valid JSON with no preamble:
    {_ESCALATION_SCHEMA}
  </output_format>
</system>"""

    raw = _llm(system, history + [{"role": "user", "content": message}])
    return Response(
        message=_get(raw, "message_to_user", "I'm connecting you with our team. Someone will be in touch shortly."),
        agent="escalation",
        escalate=True,
        debug={"summary": raw.get("summary"), "urgency": raw.get("urgency")},
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def process_message(
    message: str,
    history: list[dict],
    form: dict,
) -> Response:
    """
    Single public entry point. Runs on every message.
    NEVER raises -- always returns a Response.
    """
    try:
        clf    = classify(message, history)
        intent = clf["intent"]

        booking_started = any(form.get(k) for k in
            ["customer_name", "address", "requested_date", "requested_time"])

        is_emerg   = clf.get("is_emergency", False)
        date_wrong = clf.get("date_seems_wrong", False)

        if is_emerg:
            resp = run_escalation(message, history, clf)

        elif intent == "escalation" or (date_wrong and intent == "feedback"):
            resp = run_escalation(message, history, clf)

        elif intent == "booking" or (booking_started and intent not in ("feedback",)):
            resp = run_booking(message, history, form, clf=clf)
            if resp.escalate:
                resp = run_escalation(message, history, {**clf, "form": resp.form})
                resp.form = form

        elif intent in ("faq", "out_of_scope"):
            resp = run_faq(message, history)
            if resp.escalate:
                resp = run_escalation(message, history, clf)

        elif intent == "feedback":
            resp = run_escalation(message, history, clf)

        else:
            resp = run_faq(message, history)

        resp.intent = intent
        resp.debug.update({
            "intent":             intent,
            "sentiment":          clf.get("sentiment"),
            "urgency":            clf.get("urgency"),
            "confidence":         clf.get("confidence"),
            "reasoning":          clf.get("reasoning"),
            "is_emergency":       clf.get("is_emergency"),
            "date_in_past":       clf.get("date_in_past"),
            "date_too_far":       clf.get("date_too_far"),
            "time_outside_hours": clf.get("time_outside_hours"),
            "time_too_late":      clf.get("time_too_late"),
            "detected_date":      clf.get("detected_date"),
            "detected_time":      clf.get("detected_time"),
        })
        return resp

    except Exception as exc:
        log.error("process_message crashed: %s", exc, exc_info=True)
        return Response(
            message="I'm sorry, I ran into a technical issue. Could you try again?",
            agent="system",
            form=form,
        )
