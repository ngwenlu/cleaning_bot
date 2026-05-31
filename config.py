"""
Central config. All agents import from here.
Set OPENAI_API_KEY in your .env or Streamlit secrets.
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
    """Read from env first, then Streamlit secrets, then default."""
    return (
        os.environ.get(key)
        or (_secrets.get(key, default) if _secrets else default)
    )


# Model

MODEL = "gpt-5"
MAX_COMPLETION_TOKENS = 1024


# Client (singleton)

client = OpenAI(
    api_key=_get("OPENAI_API_KEY")
)


# Business config

COMPANY_NAME = "Dad’s Cleaning Services"

SALESPERSON_WHATSAPP = _get(
    "SALESPERSON_PHONE",
    "+6591234567"
)

SALESPERSON_EMAIL = _get(
    "SALESPERSON_EMAIL",
    "sales@example.com"
)


# Service hours in Singapore time (SGT = UTC+8)

SERVICE_HOURS_START = 8
SERVICE_HOURS_END = 20