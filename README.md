# CleanBot

A multiagent chatbot for a part-time home cleaning service in Singapore.
Built with OpenAI GPT-4o-mini and Streamlit.

The bot handles inbound customer enquiries: answering FAQs, collecting booking
details, and routing complaints or emergencies to a human salesperson. It cannot
confirm bookings or check availability — that is always done by a person.

---

## Features

- **FAQ answering** — service info, pricing, hours, supplies, cancellation policy
- **Booking detail collection** — gathers all required info before handing to salesperson
- **Emergency routing** — same-day or next-day bookings escalate to human immediately
- **Complaint handling** — negative sentiment detected and routed to salesperson
- **Date and time validation** — rejects past dates, out-of-hours times, overrunning slots
- **Sidebar progress tracker** — live booking form checklist as fields are collected
- **Debug panel** — shows intent, sentiment, urgency and datetime flags per message

---

## Project Structure

```
cleaning_chatbot/
│
├── knowledge_base.py     <- THE ONLY FILE TO EDIT for business rule changes
├── agents.py             <- All agent logic (classifier, booking, FAQ, escalation)
├── app.py                <- Streamlit UI only, no business logic
├── config.py             <- API credentials and model names
├── requirements.txt
│
└── .streamlit/
    ├── config.toml               <- Theme (warm cream + deep teal)
    └── secrets.toml.example      <- Deployment secrets template
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ngwenlu/cleaning_bot.git
cd cleaning_bot
pip install -r requirements.txt
```

### 2. Set environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
SALESPERSON_PHONE=+6591234567
SALESPERSON_EMAIL=sales@example.com
```

### 3. Run

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. In **Settings → Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
SALESPERSON_PHONE = "+6591234567"
SALESPERSON_EMAIL = "sales@example.com"
```

4. Deploy. The app is live at `[https://yourapp.streamlit.app](https://cleaningbot-a6qgdxzpwjmtmctfkfsfyx.streamlit.app/)`

---

## How It Works

Every user message goes through the same pipeline:

```
User message
    |
    v
classify()          LLM extracts intent + date/time strings
                    Python computes all datetime flags
    |
    v
process_message()   Routes based on Python-computed flags
    |
    +-- emergency (today/tomorrow)  --> run_escalation()
    +-- complaint / explicit human  --> run_escalation()
    +-- booking intent              --> run_booking()
    |       |
    |       +-- escalate=True       --> run_escalation()
    +-- faq / out of scope          --> run_faq()
            |
            +-- not answered        --> run_escalation()
```

**Key design principle:** The LLM extracts natural language (intent, date/time
strings). Python does all arithmetic (is the date in the past? does the time
overrun 9pm?). This separation prevents the common failure mode where LLMs
miscalculate dates.

---

## Booking Rules

| Rule | Value |
|------|-------|
| Service hours | 9am to 9pm, 7 days a week |
| Minimum session | 3 hours |
| Maximum session | 8 hours |
| Latest start time | 6pm (6pm + 3h = 9pm exactly) |
| Emergency window | Same day or next day |
| Furthest advance | 6 months (183 days) |
| Rate | $20/hr flat (no variation by size or location) |
| Cancellation notice | 24 hours (50% fee if late) |
| Payment | Cash, PayNow |
| Supplies | Customer provides everything. Cleaners bring nothing. |

The chatbot collects these 7 fields before handoff:

1. Customer name
2. Address
3. Date
4. Start time
5. Hours needed
6. Whether there are pets
7. Contact (phone or email)

---

## Changing Business Rules

**To change any rule** (hours, pricing, cancellation policy, booking limits):

Open `knowledge_base.py` and edit the `BusinessConfig` dataclass at the top:

```python
CONFIG = BusinessConfig(
    hours    = ServiceHours(start_hour=9, end_hour=21),
    pricing  = Pricing(hourly_rate=20),
    booking  = BookingRules(min_hours=3, max_hours=8, max_advance_days=183),
    cancellation = CancellationPolicy(notice_hours=24, late_fee_pct=50),
)
```

The change propagates automatically to all agent prompts, FAQ answers, and
Python validation on the next deploy. No other file needs touching.

**To add a new FAQ topic:**

Add a `KBEntry` to the `KNOWLEDGE_BASE` list in `knowledge_base.py`:

```python
KBEntry(
    key="public_holidays",
    topics=["public holiday", "PH", "surcharge"],
    answer="We operate on public holidays with a $10 surcharge per session.",
),
```

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| LLM | OpenAI GPT-4o-mini |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |
| Language | Python 3.11+ |
| Dependencies | openai, streamlit, python-dotenv |

---

## Architecture Notes

The project went through a significant refactor during development. The original
design used 8 separate agent files with Pydantic models for output validation.
This was collapsed into a single `agents.py` because:

- Pydantic validation errors from business rules crashed the UI
- Every bug required updating 3-4 files simultaneously
- The abstraction added complexity without benefit at this scale

Business rule validation now lives entirely in `knowledge_base.py` (for prompt
injection) and Python datetime arithmetic (for date/time flags). This means a
single file edit propagates everywhere without risk of validation crashes.
