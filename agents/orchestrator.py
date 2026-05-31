“””
Orchestrator

Entry point for every user message. Steps:

1. Run IntentClassifier → get IntentClassification
1. Apply routing rules (emergency always → escalation)
1. Call the appropriate agent
1. Return a ChatTurn

Session state (BookingDetails accumulated across turns) is passed in
and returned so the Streamlit app can persist it.
“””

from **future** import annotations

import uuid
from datetime import datetime

from agents.booking_agent import BookingAgent
from agents.escalation_agent import EscalationAgent
from agents.faq_agent import FAQAgent
from agents.follow_up_agent import FollowUpAgent
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

# ── Singleton agents (stateless — safe to reuse) ───────────────────────────

_intent_classifier = IntentClassifier()
_booking_agent     = BookingAgent()
_faq_agent         = FAQAgent()
_escalation_agent  = EscalationAgent()
_follow_up_agent   = FollowUpAgent()

def _routing_note(classification: IntentClassification, route_to: AgentType) -> str:
notes = []
if classification.is_emergency:
notes.append(“Emergency booking detected — routed to escalation.”)
if classification.confidence < 0.6:
notes.append(f”Low confidence ({classification.confidence:.2f}) — monitor.”)
notes.append(f”Sentiment: {classification.sentiment}, Urgency: {classification.urgency}”)
return “ | “.join(notes)

def process_message(
user_message: str,
history: list[ConversationMessage],
session_id: str | None = None,
booking_details: BookingDetails | None = None,
turn_number: int = 1,
) -> tuple[ChatTurn, BookingDetails | None]:
“””
Process one user message end-to-end.

```
Args:
    user_message:    The raw user input string.
    history:         Full prior conversation (ConversationMessage list).
    session_id:      Unique session identifier (auto-generated if None).
    booking_details: Accumulated booking form state from prior turns.
    turn_number:     Current turn index (for logging/display).

Returns:
    (ChatTurn, updated BookingDetails | None)
    The Streamlit app stores both in session_state.
"""
session_id = session_id or str(uuid.uuid4())

# 1. Classify intent
classification: IntentClassification = _intent_classifier.run(
    user_message, history
)

# 2. Determine route
# Emergency bookings short-circuit to escalation regardless of intent label
if classification.is_emergency:
    route_to = AgentType.ESCALATION
elif classification.intent == Intent.BOOKING_ENQUIRY:
    route_to = AgentType.BOOKING
elif classification.intent == Intent.FAQ:
    route_to = AgentType.FAQ
elif classification.intent in (Intent.COMPLAINT, Intent.ESCALATION):
    route_to = AgentType.ESCALATION
elif classification.intent == Intent.FEEDBACK:
    route_to = AgentType.FOLLOW_UP
else:
    # out_of_scope or anything else → FAQ agent handles gracefully
    route_to = AgentType.FAQ

decision = OrchestratorDecision(
    classification=classification,
    route_to=route_to,
    system_note=_routing_note(classification, route_to),
)

# 3. Call the routed agent
updated_booking: BookingDetails | None = booking_details

if route_to == AgentType.BOOKING:
    agent_response = _booking_agent.run(
        user_message,
        history,
        existing_details=booking_details or BookingDetails(),
    )
    updated_booking = agent_response.collected
    # If the booking agent detected an emergency mid-collection, re-route
    if updated_booking.is_emergency:
        escalation_response = _escalation_agent.run(
            user_message,
            history,
            classification=classification,
            booking_details=updated_booking,
        )
        agent_response = escalation_response
        decision.route_to = AgentType.ESCALATION

elif route_to == AgentType.FAQ:
    agent_response = _faq_agent.run(user_message, history)
    # If FAQ couldn't answer, escalate
    if not agent_response.answered:
        agent_response = _escalation_agent.run(
            user_message,
            history,
            classification=classification,
            booking_details=booking_details,
        )
        decision.route_to = AgentType.ESCALATION

elif route_to == AgentType.ESCALATION:
    agent_response = _escalation_agent.run(
        user_message,
        history,
        classification=classification,
        booking_details=booking_details,
    )

elif route_to == AgentType.FOLLOW_UP:
    # Follow-up agent is usually triggered by scheduler, but handle in-chat feedback too
    agent_response = _faq_agent.run(user_message, history)

else:
    agent_response = _faq_agent.run(user_message, history)

# 4. Package ChatTurn
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
```