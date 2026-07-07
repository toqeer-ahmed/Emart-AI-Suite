"""
Central configuration.

This is intentionally the ONE file you should need to touch to move from
"demo mode" to "connected to real E-Mart infrastructure":

  - DATA_LAYER: swap SQLiteDataLayer for your EmartDataLayer (see
    app/shared/data_layer.py for the interface to implement).
  - ANTHROPIC_API_KEY: set this env var to switch the Assistant and
    Listing Generator from template mode to live LLM mode.
  - CORS_ORIGINS: add E-Mart's actual frontend domain(s) here before
    going to production.
"""
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CORS_ORIGINS = os.environ.get("EMART_AI_CORS_ORIGINS", "*").split(",")
API_KEY_HEADER_NAME = "X-EMart-AI-Key"
GATEWAY_API_KEY = os.environ.get("EMART_AI_GATEWAY_KEY", "")  # empty = auth disabled (dev mode)
