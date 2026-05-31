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
# Models
# ------------------------------------------------------------------

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
# Salesperson Contact Details
# ------------------------------------------------------------------

# Business rules (hours, pricing, etc.) live in knowledge_base.py

SALESPERSON_WHATSAPP = _get(
    "SALESPERSON_PHONE",
    "+6591234567",
)

SALESPERSON_EMAIL = _get(
    "SALESPERSON_EMAIL",
    "sales@example.com",
)