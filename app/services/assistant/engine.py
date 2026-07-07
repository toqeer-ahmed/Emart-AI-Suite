"""
Shopping Assistant Engine
==========================
This is the "LangChain + Dify" role from the proposal: it takes a customer
message, calls Search and Recommendations (tool use), and turns the
combined result into a conversational reply.

Two modes:
  - LLM mode (ANTHROPIC_API_KEY set): calls Claude to phrase the reply
    naturally around the real search/recommendation results.
  - Template mode (no key): assembles a clear, correct reply from a
    template. Always available, zero cost, zero external dependency.

Either way, the *data* backing the reply (which products get mentioned) is
always real - it comes from the Search and Recommendation engines, never
invented by the LLM. This avoids the classic chatbot failure mode of
confidently recommending a product that doesn't exist in your catalog.
"""
from __future__ import annotations
import os
import json
import requests
from typing import List

from app.services.search.engine import get_search_engine
from app.services.recommendations.engine import get_recommendation_engine
from app.shared.data_layer import get_data_layer
from app.shared.schemas import AssistantTurn, AssistantReply, RecommendationResult

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


class AssistantEngine:
    def __init__(self):
        self.search = get_search_engine()
        self.recs = get_recommendation_engine()
        self.dl = get_data_layer()

    def _template_reply(self, message: str, search_hits, recs) -> str:
        if search_hits:
            names = ", ".join(h.name for h in search_hits[:3])
            reply = f"Here's what I found matching '{message}': {names}."
        else:
            reply = f"I couldn't find an exact match for '{message}' in stock right now."
        if recs:
            rec_names = ", ".join(r.name for r in recs[:2])
            reply += f" You might also like: {rec_names}."
        return reply

    def _llm_reply(self, message: str, search_hits, recs) -> str:
        catalog_context = json.dumps(
            {"search_results": [h.model_dump() for h in search_hits],
             "recommendations": [r.model_dump() for r in recs]}
        )
        prompt = (
            f"You are a helpful shopping assistant for an e-commerce marketplace. "
            f"A customer said: \"{message}\". "
            f"Here is the ONLY real product data available "
            f"(you must not invent products not in this list): {catalog_context}\n\n"
            f"Write a short, friendly 2-3 sentence reply recommending from this data only."
        )
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": "claude-sonnet-4-6", "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    def handle_turn(self, turn: AssistantTurn) -> AssistantReply:
        used_services = ["search"]
        search_hits = self.search.search(turn.message, top_k=5)

        recs = self.recs.for_customer(turn.customer_id, top_k=3)
        used_services.append("recommendations")

        if ANTHROPIC_API_KEY:
            try:
                reply_text = self._llm_reply(turn.message, search_hits, recs)
                used_services.append("llm")
            except Exception:
                reply_text = self._template_reply(turn.message, search_hits, recs)
        else:
            reply_text = self._template_reply(turn.message, search_hits, recs)

        suggested: List[RecommendationResult] = [
            RecommendationResult(sku=h.sku, name=h.name, price=h.price, reason="matches your query")
            for h in search_hits[:3]
        ] + recs[:2]

        reply = AssistantReply(reply=reply_text, suggested_products=suggested, used_services=used_services)
        self.dl.log_signal("assistant_turn", {"customer_id": turn.customer_id, "message": turn.message, "reply": reply_text})
        return reply


_engine_singleton: AssistantEngine = None


def get_assistant_engine() -> AssistantEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = AssistantEngine()
    return _engine_singleton
