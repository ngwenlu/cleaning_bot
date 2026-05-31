“””
Pydantic models for the cleaning company multiagent chatbot.
All agent outputs are standardised through these schemas.
“””

from **future** import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# —————————————————————————

# Enums

# —————————————————————————

class Intent(str, Enum):
“”“Top-level intent categories produced by the intent classifier.”””
BOOKING_ENQUIRY   = “booking_enquiry”    # wants to make a booking (non-urgent)
EMERGENCY_BOOKING = “emergency_booking”  # same-day or next-day booking -> human immediately
FAQ               = “faq”                # general questions about the service
COMPLAINT         = “complaint”          # unhappy with service -> escalate
FEEDBACK          = “feedback”           # post-service feedback
ESCALATION        = “escalation”         # explicit request for human / unclear danger
OUT_OF_SCOPE      = “out_of_scope”       # unrelated to the business

class Sentiment(str, Enum):
“”“Customer sentiment detected by the intent classifier.”””
POSITIVE  = “positive”
NEUTRAL   = “neutral”
NEGATIVE  = “negative”
URGENT    = “urgent”

class UrgencyLevel(str, Enum):
ROUTINE   = “routine”   # standard booking / FAQ
HIGH      = “high”      # complaint, strong negative sentiment
CRITICAL  = “critical”  # emergency booking, safety, or explicit distress

class ApartmentType(str, Enum):
HDB_1_2_ROOM  = “hdb_1_2_room”
HDB_3_ROOM    = “hdb_3_room”
HDB_4_ROOM    = “hdb_4_room”
HDB_5_ROOM    = “hdb_5_room”
CONDO_STUDIO  = “condo_studio”
CONDO_1_BED   = “condo_1_bed”
CONDO_2_BED   = “condo_2_bed”
CONDO_3_BED   = “condo_3_bed”
LANDED        = “landed”
OFFICE        = “office”
OTHER         = “other”

class AgentType(str, Enum):
ORCHESTRATOR = “orchestrator”
BOOKING      = “booking”
FAQ          = “faq”
FOLLOW_UP    = “follow_up”
ESCALATION   = “escalation”

# —————————————————————————

# Shared primitives

# —————————————————————————

class ConversationMessage(BaseModel):
“”“A single turn in the conversation history.”””
role:       str       = Field(…, description=”‘user’ or ‘assistant’”)
content:    str
timestamp:  datetime  = Field(default_factory=datetime.utcnow)
agent_type: Optional[AgentType] = Field(
None, description=“Which agent produced this message (assistant turns only)”
)

class CustomerInfo(BaseModel):
“”“Basic customer contact details, collected incrementally.”””
name:           Optional[str] = None
phone:          Optional[str] = None
email:          Optional[str] = None

```
@field_validator("phone")
@classmethod
def validate_sg_phone(cls, v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    digits = v.replace(" ", "").replace("-", "").replace("+", "")
    # Accept SG numbers: 8-digit local or 65-prefixed
    if not (digits.isdigit() and len(digits) in (8, 10)):
        raise ValueError("Phone must be a valid Singapore number (8 digits or +65xxxxxxxx).")
    return v
```

# —————————————————————————

# Intent Classifier output

# —————————————————————————

class IntentClassification(BaseModel):
“””
Output produced by the LLM-based intent + sentiment classifier.
This is the first model created on every user message.
“””
intent:           Intent
sentiment:        Sentiment
urgency:          UrgencyLevel
confidence:       float        = Field(…, ge=0.0, le=1.0)
reasoning:        str          = Field(…, description=“Brief LLM explanation of the classification”)
is_emergency:     bool         = Field(
False,
description=“True when the requested date is today or tomorrow. Derived from date if present.”
)
detected_date:    Optional[date] = Field(
None,
description=“Booking date explicitly mentioned by the user, if any”
)

```
@model_validator(mode="after")
def sync_emergency_flag(self) -> "IntentClassification":
    """If the detected date is today or tomorrow, force emergency."""
    if self.detected_date is not None:
        delta = (self.detected_date - date.today()).days
        if delta in (0, 1):
            self.is_emergency = True
            self.intent = Intent.EMERGENCY_BOOKING
            self.urgency = UrgencyLevel.CRITICAL
    return self
```

# —————————————————————————

# Booking Agent output

# —————————————————————————

class BookingDetails(BaseModel):
“””
Lead / booking details collected by the Booking Agent.
All fields are optional – the agent fills them in progressively
across conversation turns. Nothing is confirmed here.
“””
customer:           CustomerInfo  = Field(default_factory=CustomerInfo)
requested_date:     Optional[date]  = None
requested_time:     Optional[time]  = None
address:            Optional[str]   = None
postal_code:        Optional[str]   = None
apartment_type:     Optional[ApartmentType] = None
hours_needed:       Optional[float] = Field(None, gt=0, description=“Estimated cleaning hours”)
num_rooms:          Optional[int]   = Field(None, gt=0)
special_instructions: Optional[str] = None
supplies_confirmed: bool            = Field(
False,
description=“Customer has acknowledged they will provide all cleaning supplies”
)

```
# Computed flag -- set by validator, not user input
is_emergency:       bool = False

@model_validator(mode="after")
def flag_emergency(self) -> "BookingDetails":
    if self.requested_date is not None:
        delta = (self.requested_date - date.today()).days
        self.is_emergency = delta in (0, 1)
    return self

def missing_fields(self) -> list[str]:
    """Return names of required fields that are still None."""
    required = [
        "requested_date", "requested_time", "address",
        "apartment_type", "hours_needed",
    ]
    missing = [f for f in required if getattr(self, f) is None]
    if not self.customer.name:
        missing.insert(0, "customer.name")
    if not (self.customer.phone or self.customer.email):
        missing.append("customer.phone_or_email")
    return missing
```

class BookingAgentResponse(BaseModel):
“”“Output returned by the Booking Agent each turn.”””
agent_type:      AgentType      = AgentType.BOOKING
message:         str            = Field(…, description=“Natural-language reply to the user”)
collected:       BookingDetails = Field(default_factory=BookingDetails)
is_complete:     bool           = Field(
False,
description=“True when all required fields in BookingDetails are filled”
)
next_field_to_ask: Optional[str] = Field(
None,
description=“The next missing field the agent should ask about”
)

# —————————————————————————

# FAQ Agent output

# —————————————————————————

class FAQAgentResponse(BaseModel):
“”“Output returned by the FAQ Agent.”””
agent_type: AgentType = AgentType.FAQ
message:    str
sources:    list[str] = Field(
default_factory=list,
description=“Knowledge-base keys or section titles used to answer”
)
answered:   bool      = Field(
True,
description=“False if the question is outside the FAQ knowledge base”
)

# —————————————————————————

# Escalation Agent output

# —————————————————————————

class EscalationRequest(BaseModel):
“””
Produced whenever the chatbot must hand off to a human.
Covers: emergency bookings, complaints, explicit human requests.
“””
agent_type:            AgentType    = AgentType.ESCALATION
urgency:               UrgencyLevel
reason:                str          = Field(…, description=“Why escalation was triggered”)
customer:              CustomerInfo = Field(default_factory=CustomerInfo)
booking_details:       Optional[BookingDetails] = None
conversation_summary:  str          = Field(
…,
description=“LLM-generated summary of the conversation so far for the human agent”
)
message_to_user:       str          = Field(
…,
description=“What the bot says to the customer while handing off”
)
notify_via:            list[str]    = Field(
default_factory=lambda: [“whatsapp”],
description=“Channels to alert the human agent through, e.g. [‘whatsapp’, ‘email’]”
)

# —————————————————————————

# Follow-up Agent output

# —————————————————————————

class FollowUpAgentResponse(BaseModel):
“”“Output for post-service follow-ups and reminders.”””
agent_type:   AgentType = AgentType.FOLLOW_UP
message:      str
follow_up_type: str     = Field(
…,
description=”‘reminder_24h’ | ‘post_service_feedback’ | ‘rebooking_prompt’”
)
customer:     CustomerInfo

# —————————————————————————

# Orchestrator output

# —————————————————————————

class OrchestratorDecision(BaseModel):
“””
The Orchestrator’s routing decision after classifying intent.
Tells the system which agent to invoke next.
“””
classification:  IntentClassification
route_to:        AgentType
system_note:     Optional[str] = Field(
None,
description=“Internal note explaining the routing decision”
)

```
@model_validator(mode="after")
def enforce_emergency_routing(self) -> "OrchestratorDecision":
    """Emergency bookings must always route to escalation, no exceptions."""
    if self.classification.is_emergency:
        self.route_to = AgentType.ESCALATION
    return self
```

# —————————————————————————

# Top-level chat turn

# —————————————————————————

class ChatTurn(BaseModel):
“””
The full record of one user message + bot response.
Stored in session state and optionally persisted to DB.
“””
session_id:         str
turn_number:        int
user_message:       ConversationMessage
orchestrator:       OrchestratorDecision
agent_response:     BookingAgentResponse | FAQAgentResponse | EscalationRequest | FollowUpAgentResponse
timestamp:          datetime = Field(default_factory=datetime.utcnow)