"""
app.py – Streamlit UI for the Cleaning Company Multiagent Chatbot
Run: streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from models import (
    AgentType,
    BookingAgentResponse,
    BookingDetails,
    ConversationMessage,
    EscalationRequest,
    FAQAgentResponse,
    FollowUpAgentResponse,
)
from agents.orchestrator import process_message


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
}

section[data-testid="stSidebar"] * {
    color: #E8F0ED !important;
}

.agent-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}

.badge-booking { background: #DBEAFE; color: #1D4ED8; }
.badge-faq { background: #D1FAE5; color: #065F46; }
.badge-escalation { background: #FEE2E2; color: #991B1B; }
.badge-followup { background: #FEF3C7; color: #92400E; }
.badge-system { background: #F3F4F6; color: #374151; }

.bot-text {
    color: #1C1C1C !important;
    font-size: 16px;
    line-height: 1.65;
    margin-top: 8px;
    margin-bottom: 16px;
}

.escalation-box {
    color: #1C1C1C !important;
    background: #FFF5F5;
    border-left: 4px solid #EF4444;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-top: 6px;
}

.emergency-box {
    color: #1C1C1C !important;
    background: #FFF7ED;
    border-left: 4px solid #F97316;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-top: 6px;
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
}

.company-tagline {
    font-size: 12px;
    color: #9BBFB3 !important;
    letter-spacing: 0.05em;
}

.progress-card {
    background: rgba(255,255,255,0.07);
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

.check-done { color: #4ADE80; font-size: 14px; }
.check-empty { color: #6B7280; font-size: 14px; }

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


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    if "booking_details" not in st.session_state:
        st.session_state.booking_details = None

    if "turn_number" not in st.session_state:
        st.session_state.turn_number = 1

    if "last_classification" not in st.session_state:
        st.session_state.last_classification = None


_init_state()


def _agent_badge(agent_type: AgentType | str | None) -> str:
    labels = {
        AgentType.BOOKING: ("BOOKING", "badge-booking"),
        AgentType.FAQ: ("FAQ", "badge-faq"),
        AgentType.ESCALATION: ("URGENT", "badge-escalation"),
        AgentType.FOLLOW_UP: ("FOLLOW UP", "badge-followup"),
        AgentType.ORCHESTRATOR: ("SYSTEM", "badge-system"),
    }

    if agent_type is None:
        return ""

    try:
        key = agent_type if isinstance(agent_type, AgentType) else AgentType(agent_type)
    except Exception:
        key = AgentType.ORCHESTRATOR

    label, cls = labels.get(key, ("AGENT", "badge-system"))
    return f'<span class="agent-badge {cls}">{label}</span>'


def _render_bot_text(text: str) -> None:
    safe_text = html.escape(text or "_No response_").replace("\n", "<br>")
    st.markdown(
        f'<div class="bot-text">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def _render_booking_progress(bd: BookingDetails) -> None:
    fields = [
        ("Name", bd.customer.name),
        ("Contact", bd.customer.phone or bd.customer.email),
        ("Date", str(bd.requested_date) if bd.requested_date else None),
        ("Time", str(bd.requested_time) if bd.requested_time else None),
        ("Address", bd.address),
        ("Apt type", bd.apartment_type.value if bd.apartment_type else None),
        ("Hours needed", str(bd.hours_needed) if bd.hours_needed else None),
        ("Supplies done", "Yes" if bd.supplies_confirmed else None),
    ]

    rows = ""
    filled = sum(1 for _, value in fields if value)

    for label, value in fields:
        icon = (
            '<span class="check-done">O</span>'
            if value
            else '<span class="check-empty">o</span>'
        )
        display = (
            f"<b style='color:#E8F0ED'>{html.escape(str(value))}</b>"
            if value
            else "<span style='color:#6B7280'>-</span>"
        )
        rows += (
            f'<div class="progress-row">{icon} '
            f'<span style="flex:1">{label}</span>{display}</div>'
        )

    pct = int(filled / len(fields) * 100)

    st.markdown(
        f"""
<div class="progress-card">
    <div style="font-size:11px;letter-spacing:.07em;color:#9BBFB3;margin-bottom:6px">
        BOOKING FORM &nbsp;.&nbsp; {filled}/{len(fields)} fields
    </div>
    <div style="background:rgba(255,255,255,.1);border-radius:999px;height:4px;margin-bottom:10px">
        <div style="background:#4ADE80;width:{pct}%;height:4px;border-radius:999px"></div>
    </div>
    {rows}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg.get("content") or ""
    agent_type = msg.get("agent_type")
    metadata = msg.get("metadata", {})
    is_emergency = metadata.get("is_emergency", False)

    with st.chat_message(role, avatar="🧹" if role == "assistant" else "👤"):
        if role != "assistant":
            st.markdown(content)
            return

        badge = _agent_badge(agent_type)

        if agent_type == AgentType.ESCALATION:
            box_class = "emergency-box" if is_emergency else "escalation-box"
            safe_content = html.escape(content).replace("\n", "<br>")
            st.markdown(
                f'{badge}<div class="{box_class}">{safe_content}</div>',
                unsafe_allow_html=True,
            )
        else:
            if badge:
                st.markdown(badge, unsafe_allow_html=True)
            _render_bot_text(content)


with st.sidebar:
    st.markdown(
        """
<div class="company-header">
    <span class="company-icon">🧹</span>
    <span class="company-name">Dad's Cleaning</span>
    <span class="company-tagline">PART-TIME HOME CLEANING . SINGAPORE</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    bd: BookingDetails | None = st.session_state.booking_details

    if bd and any(
        [
            bd.customer.name,
            bd.requested_date,
            bd.address,
            bd.apartment_type,
        ]
    ):
        st.markdown(
            "<div style='font-size:13px;font-weight:600;letter-spacing:.05em'>📋 BOOKING IN PROGRESS</div>",
            unsafe_allow_html=True,
        )
        _render_booking_progress(bd)
        st.markdown("---")

    clf = st.session_state.last_classification

    if clf:
        with st.expander("🔍 Last classification", expanded=False):

            def _row(label: str, value: str) -> None:
                st.markdown(
                    f'<div class="debug-row"><span>{label}</span>'
                    f'<span class="debug-val">{value}</span></div>',
                    unsafe_allow_html=True,
                )

            _row("Intent", clf.intent.value)
            _row("Sentiment", clf.sentiment.value)
            _row("Urgency", clf.urgency.value)
            _row("Confidence", f"{clf.confidence:.0%}")
            _row("Emergency", "YES" if clf.is_emergency else "No")

            if clf.reasoning:
                st.markdown(
                    f"<div style='font-size:11px;color:#9CA3AF;margin-top:6px'>{html.escape(clf.reasoning)}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    if st.button("🔄 Start new conversation", use_container_width=True):
        for key in [
            "messages",
            "history",
            "booking_details",
            "turn_number",
            "last_classification",
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


for msg in st.session_state.messages:
    _render_message(msg)


if prompt := st.chat_input("Type your message..."):
    user_msg_dict = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg_dict)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧹"):
        with st.spinner(""):
            try:
                chat_turn, updated_booking = process_message(
                    user_message=prompt,
                    history=st.session_state.history,
                    session_id=st.session_state.session_id,
                    booking_details=st.session_state.booking_details,
                    turn_number=st.session_state.turn_number,
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

        response = chat_turn.agent_response
        route_to = chat_turn.orchestrator.route_to
        clf = chat_turn.orchestrator.classification

        if isinstance(response, BookingAgentResponse):
            reply_text = response.message
        elif isinstance(response, FAQAgentResponse):
            reply_text = response.message
        elif isinstance(response, EscalationRequest):
            reply_text = response.message_to_user
        elif isinstance(response, FollowUpAgentResponse):
            reply_text = response.message
        else:
            reply_text = str(response)

        is_emergency = clf.is_emergency
        badge = _agent_badge(route_to)

        if route_to == AgentType.ESCALATION:
            box_class = "emergency-box" if is_emergency else "escalation-box"
            safe_reply = html.escape(reply_text).replace("\n", "<br>")
            st.markdown(
                f'{badge}<div class="{box_class}">{safe_reply}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(badge, unsafe_allow_html=True)
            _render_bot_text(reply_text)

    st.session_state.last_classification = clf

    if isinstance(response, BookingAgentResponse):
        st.session_state.booking_details = response.collected

    if updated_booking:
        st.session_state.booking_details = updated_booking

    st.session_state.history.append(
        ConversationMessage(role="user", content=prompt)
    )

    st.session_state.history.append(
        ConversationMessage(
            role="assistant",
            content=reply_text,
            agent_type=route_to,
        )
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply_text,
            "agent_type": route_to,
            "metadata": {"is_emergency": is_emergency},
        }
    )

    st.session_state.turn_number += 1
    st.rerun()