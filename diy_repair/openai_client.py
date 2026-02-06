"""
OpenAI Client Utility
Centralized configuration for OpenAI client with ChatAnywhere API
"""

import os
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
)


def get_openai_client():
    """Get the configured OpenAI client instance."""
    return client
