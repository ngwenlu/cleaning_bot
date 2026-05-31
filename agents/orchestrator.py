"""
Orchestrator

Entry point for every user message.

Flow:
1. Run IntentClassifier
2. Apply routing rules
3. Call the appropriate agent
4. Return a ChatTurn

Session state (BookingDetails accumulated across turns) is passed in
and returned so the Streamlit app can persist it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from agents.booking import BookingAgent
from agents.escalation import EscalationAgent
from agents.faq import FAQAgent
from agents.follow_up import FollowUpAgent
from agents.intent_classifier import IntentClassifier

from models import (
    AgentType,
    BookingDetails,
    ChatTurn,
    ConversationMessage,
    Intent,
    IntentClassification,
    OrchestratorDecision,
)


# Singleton agents

_intent_classifier = IntentClassifier()
_booking_agent = BookingAgent()
_faq_agent = FAQAgent()
_escalation_agent = EscalationAgent()
_follow_up_agent = FollowUpAgent()


def _routing_note(
    classification: IntentClassification,
    route_to: AgentType,
) -> str:
    notes: list[str] = []

    if classification.is_emergency:
        notes.append(
            "Emergency booking detected, routed to escalation."
        )

    if classification.confidence < 0.60:
        notes.append(
            f"Low confidence ({classification.confidence:.2f})."
        )

    notes.append(
        f"Sentiment={classification.sentiment.value}, "
        f"Urgency={classification.urgency.value}"
    )

    return " | ".join(notes)


def process_message(
    user_message: str,
    history: list[ConversationMessage],
    session_id: str | None = None,
    booking_details: BookingDetails | None = None,
    turn_number: int = 1,
) -> tuple[ChatTurn, BookingDetails | None]:
    """
    Process one user message end-to-end.

    Returns:
        (ChatTurn, updated_booking_details)
    """

    session_id = session_id or str(uuid.uuid4())

    # 1. Classify intent

    classification: IntentClassification = (
        _intent_classifier.run(
            user_message=user_message,
            history=history,
        )
    )

    # 2. Determine route

    if classification.is_emergency:
        route_to = AgentType.ESCALATION

    elif classification.intent == Intent.BOOKING_ENQUIRY:
        route_to = AgentType.BOOKING

    elif classification.intent == Intent.FAQ:
        route_to = AgentType.FAQ

    elif classification.intent in (
        Intent.COMPLAINT,
        Intent.ESCALATION,
    ):
        route_to = AgentType.ESCALATION

    elif classification.intent == Intent.FEEDBACK:
        route_to = AgentType.FOLLOW_UP

    else:
        route_to = AgentType.FAQ

    decision = OrchestratorDecision(
        classification=classification,
        route_to=route_to,
        system_note=_routing_note(
            classification,
            route_to,
        ),
    )

    # 3. Call routed agent

    updated_booking: BookingDetails | None = booking_details

    if route_to == AgentType.BOOKING:

        agent_response = _booking_agent.run(
            user_message=user_message,
            history=history,
            existing_details=booking_details
            or BookingDetails(),
        )

        updated_booking = agent_response.collected

        # Safety check in case emergency is detected later

        if updated_booking.is_emergency:

            escalation_response = _escalation_agent.run(
                user_message=user_message,
                history=history,
                classification=classification,
                booking_details=updated_booking,
            )

            agent_response = escalation_response
            decision.route_to = AgentType.ESCALATION

    elif route_to == AgentType.FAQ:

        agent_response = _faq_agent.run(
            user_message=user_message,
            history=history,
        )

        if not agent_response.answered:

            agent_response = _escalation_agent.run(
                user_message=user_message,
                history=history,
                classification=classification,
                booking_details=booking_details,
            )

            decision.route_to = AgentType.ESCALATION

    elif route_to == AgentType.ESCALATION:

        agent_response = _escalation_agent.run(
            user_message=user_message,
            history=history,
            classification=classification,
            booking_details=booking_details,
        )

    elif route_to == AgentType.FOLLOW_UP:

        # Feedback messages inside chat
        # For now route to FAQ response handling

        agent_response = _faq_agent.run(
            user_message=user_message,
            history=history,
        )

    else:

        agent_response = _faq_agent.run(
            user_message=user_message,
            history=history,
        )

    # 4. Create chat turn

    user_turn = ConversationMessage(
        role="user",
        content=user_message,
        timestamp=datetime.utcnow(),
    )

    chat_turn = ChatTurn(
        session_id=session_id,
        turn_number=turn_number,
        user_message=user_turn,
        orchestrator=decision,
        agent_response=agent_response,
    )

    return chat_turn, updated_booking