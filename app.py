"""
app.py – Streamlit UI for the Cleaning Company Chatbot.
Calls agents.process_message(); has no business logic of its own.
"""

from __future__ import annotations

import html as _html
import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from agents import Response, process_message


st.set_page_config(
    page_title="CleanBot | Dad's Cleaning",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #FAF8F4;
}

section[data-testid="stSidebar"] {
    background-color: #1A2E2A;
    border-right: none;
}

section[data-testid="stSidebar"] * {
    color: #E8F0ED !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif !important;
}

.agent-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 4px;
}

.badge-booking { background: #DBEAFE; color: #1D4ED8; }
.badge-faq { background: #D1FAE5; color: #065F46; }
.badge-escalation { background: #FEE2E2; color: #991B1B; }
.badge-system { background: #F3F4F6; color: #374151; }

.alert-box {
    border-left: 4px solid #EF4444;
    background: #FFF5F5;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-top: 6px;
    color: #1C1C1C;
    line-height: 1.6;
}

.progress-card {
    background: rgba(255,255,255,.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
}

.progress-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    padding: 3px 0;
}

.check-done {
    color: #4ADE80;
}

.check-empty {
    color: #6B7280;
}

.company-header {
    text-align: center;
    padding: 8px 0 16px 0;
}

.company-icon {
    font-size: 40px;
    display: block;
    margin-bottom: 6px;
}

.company-name {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #FFFFFF !important;
    display: block;
    margin-bottom: 2px;
}

.company-tagline {
    font-size: 12px;
    color: #9BBFB3 !important;
    letter-spacing: .05em;
}

.debug-row {
    font-size: 11px;
    color: #9CA3AF;
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
}

.debug-val {
    color: #D1FAE5;
    font-weight: 500;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


def _init() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "history": [],
        "form": {},
        "last_debug": {},
        "turn": 1,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init()


_BADGE_CSS = {
    "booking": "badge-booking",
    "faq": "badge-faq",
    "escalation": "badge-escalation",
    "system": "badge-system",
}

_BADGE_LABEL = {
    "booking": "BOOKING",
    "faq": "FAQ",
    "escalation": "URGENT",
    "system": "SYSTEM",
}


def _badge(agent: str) -> str:
    agent = agent or "system"
    css = _BADGE_CSS.get(agent, "badge-system")
    label = _BADGE_LABEL.get(agent, agent.upper())
    return f'<span class="agent-badge {css}">{label}</span>'


def _render_assistant(
    badge_html: str,
    body: str,
    agent: str,
) -> None:
    if body:
        safe = _html.escape(body).replace("\n", "<br>")
    else:
        safe = "<em style='color:#9CA3AF'>No response</em>"

    if agent == "escalation":
        inner = f'<div class="alert-box">{safe}</div>'
    else:
        inner = (
            '<p style="margin:6px 0 0 0;font-size:15px;'
            f'color:#1C1C1C;line-height:1.6">{safe}</p>'
        )

    st.markdown(
        f"{badge_html}{inner}",
        unsafe_allow_html=True,
    )


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg.get("content", "")
    agent = msg.get("agent", "faq")

    with st.chat_message(
        role,
        avatar="🧹" if role == "assistant" else "👤",
    ):
        if role == "user":
            st.markdown(content)
        else:
            _render_assistant(
                _badge(agent),
                content,
                agent,
            )


def _render_form_progress(form: dict) -> None:
    fields = [
        ("Name", form.get("customer_name")),
        ("Address", form.get("address")),
        ("Date", form.get("requested_date")),
        ("Start time", form.get("requested_time")),
        (
            "Hours",
            str(form["hours_needed"]) if form.get("hours_needed") else None,
        ),
        (
            "Pets",
            "Yes"
            if form.get("has_pets") is True
            else ("No" if form.get("has_pets") is False else None),
        ),
        ("Contact", form.get("contact")),
    ]

    filled = sum(1 for _, value in fields if value)
    pct = int(filled / len(fields) * 100)
    rows = ""

    for label, value in fields:
        icon = (
            '<span class="check-done">●</span>'
            if value
            else '<span class="check-empty">○</span>'
        )

        display = (
            f"<b style='color:#E8F0ED'>{_html.escape(str(value))}</b>"
            if value
            else "<span style='color:#6B7280'>-</span>"
        )

        rows += (
            f'<div class="progress-row">{icon} '
            f'<span style="flex:1">{_html.escape(label)}</span>{display}</div>'
        )

    st.markdown(
        f"""
<div class="progress-card">
  <div style="font-size:11px;letter-spacing:.07em;color:#9BBFB3;margin-bottom:6px">
    BOOKING FORM · {filled}/{len(fields)} fields
  </div>
  <div style="background:rgba(255,255,255,.1);border-radius:999px;height:4px;margin-bottom:10px">
    <div style="background:#4ADE80;width:{pct}%;height:4px;border-radius:999px"></div>
  </div>
  {rows}
</div>
""",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        """
<div class="company-header">
    <span class="company-icon">🧹</span>
    <span class="company-name">Dad's Cleaning</span>
    <span class="company-tagline">PART-TIME HOME CLEANING · SINGAPORE</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    form = st.session_state.form

    if any(
        form.get(key)
        for key in [
            "customer_name",
            "address",
            "requested_date",
            "requested_time",
        ]
    ):
        st.markdown(
            "<div style='font-size:13px;font-weight:600;letter-spacing:.05em'>📋 BOOKING IN PROGRESS</div>",
            unsafe_allow_html=True,
        )
        _render_form_progress(form)
        st.markdown("---")

    debug = st.session_state.last_debug

    if debug:
        with st.expander("🔍 Last classification", expanded=True):
            rows = [
                ("Intent", debug.get("intent")),
                ("Agent", debug.get("agent")),
                ("Emergency", "YES" if debug.get("is_emergency") else "No"),
                ("Date text", debug.get("date_text")),
                ("Time text", debug.get("time_text")),
                ("Detected date", debug.get("detected_date")),
                ("Detected time", debug.get("detected_time")),
                ("Date too far", debug.get("date_too_far")),
                ("Time too late", debug.get("time_too_late")),
                ("Outside hours", debug.get("time_outside_hours")),
            ]

            for label, value in rows:
                st.markdown(
                    f'<div class="debug-row"><span>{_html.escape(str(label))}</span>'
                    f'<span class="debug-val">{_html.escape(str(value))}</span></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("🧪 Full debug JSON", expanded=False):
            st.json(debug)

    st.markdown("---")

    if st.button("🔄 Start new conversation", use_container_width=True):
        for key in [
            "messages",
            "history",
            "form",
            "last_debug",
            "turn",
            "session_id",
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    st.markdown(
        "<div style='font-size:11px;color:#4A7A6E;text-align:center;margin-top:12px'>"
        "Powered by OpenAI</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    "<h1 style='font-family:\"DM Serif Display\",serif;font-size:28px;"
    "color:#1C1C1C;margin-bottom:4px'>Hi there 👋</h1>"
    "<p style='color:#6B7280;margin-top:0;margin-bottom:24px'>"
    "I can answer questions about our cleaning service or help get your booking details ready.</p>",
    unsafe_allow_html=True,
)


for message in st.session_state.messages:
    _render_message(message)


if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧹"):
        with st.spinner(""):
            response: Response = process_message(
                message=prompt,
                history=st.session_state.history,
                form=st.session_state.form,
            )

        # IMPORTANT:
        # Badge is based ONLY on response.agent.
        # Do not use debug["urgency"], debug["sentiment"], or response.escalate for the badge.
        agent = response.agent or "system"

        response.debug["agent"] = agent

        _render_assistant(
            _badge(agent),
            response.message,
            agent,
        )

    st.session_state.form = response.form
    st.session_state.last_debug = response.debug
    st.session_state.turn += 1

    st.session_state.history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": response.message,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.message,
            "agent": agent,
        }
    )

    st.rerun()