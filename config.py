"""
Central config. All agents import from here.
Set ANTHROPIC_API_KEY in your .env or Streamlit secrets.
"""

import os
from anthropic import Anthropic

# Support Streamlit secrets when running on Streamlit Cloud

try:
import streamlit as st
_secrets = st.secrets
except Exception:
_secrets = {}

def _get(key: str, default: str = “”) -> str:
“”“Read from env first, then Streamlit secrets, then default.”””
return os.environ.get(key) or (_secrets.get(key, default) if _secrets else default)

# – Model ——————————————————————

MODEL = “claude-sonnet-4-20250514”
MAX_TOKENS = 1024

# – Client (singleton) —————————————————–

client = Anthropic(api_key=_get(“ANTHROPIC_API_KEY”))

# – Business config ––––––––––––––––––––––––––––

COMPANY_NAME         = “Dad’s Cleaning Services”
SALESPERSON_WHATSAPP = _get(“SALESPERSON_PHONE”, “+6591234567”)
SALESPERSON_EMAIL    = _get(“SALESPERSON_EMAIL”, “sales@example.com”)

# Service hours in Singapore time (SGT = UTC+8)

SERVICE_HOURS_START = 8   # 8am
SERVICE_HOURS_END   = 20  # 8pm