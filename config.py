"""
Central config. All agents import from here.

Set OPENAI_API_KEY in your .env file or Streamlit secrets.
"""

import os

from openai import OpenAI

# Support Streamlit secrets when running on Streamlit Cloud

try:
    import streamlit as st

    _secrets = st.secrets
except Exception:
    _secrets = {}


def _get(key: str, default: str = "") -> str:
    """
    Read from environment variables first,
    then Streamlit secrets,
    then fall back to default.
    """
    return (
        os.environ.get(key)
        or (_secrets.get(key, default) if _secrets else default)
    )


# ------------------------------------------------------------------
# OpenAI Models
# ------------------------------------------------------------------

# gpt-4o-mini is fast and inexpensive while still handling
# structured JSON extraction very well.

MODEL = "gpt-4o-mini"
CLASSIFIER_MODEL = "gpt-4o-mini"

MAX_TOKENS = 1024
CLASSIFIER_MAX_TOKENS = 300


# ------------------------------------------------------------------
# OpenAI Client
# ------------------------------------------------------------------

client = OpenAI(
    api_key=_get("OPENAI_API_KEY")
)


# ------------------------------------------------------------------
# Business Configuration
# ------------------------------------------------------------------

COMPANY_NAME = "Dad's Cleaning Services"

SALESPERSON_WHATSAPP = _get(
    "SALESPERSON_PHONE",
    "+6591234567",
)

SALESPERSON_EMAIL = _get(
    "SALESPERSON_EMAIL",
    "sales@example.com",
)


# ------------------------------------------------------------------
# Service Hours (Singapore Time, UTC+8)
# ------------------------------------------------------------------

SERVICE_HOURS_START = 8   # 8:00 AM
SERVICE_HOURS_END = 20    # 8:00 PM