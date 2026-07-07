"""
Listing Generator Engine
=========================
Generates a marketplace-ready listing (title, bullets, description, SEO
keywords) for a product.

Two modes, chosen automatically:
  1. LLM mode - if ANTHROPIC_API_KEY is set in the environment, calls the
     real Claude API to write genuinely tailored copy.
  2. Template mode - if no key is set, falls back to a rule-based
     template so the endpoint still works out of the box with zero
     external dependencies or cost. This is what runs by default in a
     fresh checkout of this repo.

This mirrors the Dify/LangChain "content generation" role from the
proposal, without forcing you to stand up Dify just to get this one
feature working.
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any

from app.shared.data_layer import get_data_layer
from app.shared.schemas import ListingDraft

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


class ListingGeneratorEngine:
    def __init__(self):
        self.dl = get_data_layer()

    def _template_listing(self, product: Dict[str, Any]) -> ListingDraft:
        name = product["name"]
        category = product.get("category", "product")
        tags = product.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = [t.strip() for t in tags.split(",") if t.strip()]

        title = f"{name} - Premium {category.title()} | Fast Shipping"
        bullets = [
            f"HIGH QUALITY: {name} built for everyday reliability.",
            f"VERSATILE: Great fit for {category} needs, at home or on the go.",
            "FAST SHIPPING: Ships quickly with tracked delivery.",
            "CUSTOMER SUPPORT: Backed by responsive seller support.",
        ]
        description = (
            f"{name} is designed for customers who want dependable {category} "
            f"products without compromise. {product.get('description', '')} "
            f"Order now and enjoy fast, tracked delivery."
        ).strip()
        keywords = list({category, name.split()[0].lower(), *tags})

        return ListingDraft(
            sku=product["sku"], title=title, bullet_points=bullets,
            description=description, seo_keywords=keywords,
        )

    def _llm_listing(self, product: Dict[str, Any]) -> ListingDraft:
        """Real call to Claude via the Anthropic API. Only runs if
        ANTHROPIC_API_KEY is configured in the environment."""
        import requests

        prompt = f"""Write an e-commerce marketplace listing for this product.
Return ONLY valid JSON with keys: title, bullet_points (array of 4 strings),
description (2-3 sentences), seo_keywords (array of 5 strings).

Product name: {product['name']}
Category: {product.get('category', '')}
Raw description: {product.get('description', '')}
Price: {product.get('price', '')}
"""
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        return ListingDraft(
            sku=product["sku"], title=data["title"], bullet_points=data["bullet_points"],
            description=data["description"], seo_keywords=data["seo_keywords"],
        )

    def generate(self, sku: str) -> ListingDraft:
        product = self.dl.get_product(sku)
        if not product:
            raise ValueError(f"Unknown SKU: {sku}")

        if ANTHROPIC_API_KEY:
            try:
                draft = self._llm_listing(product)
            except Exception:
                # Fail safe: never break the endpoint just because the LLM
                # call had an issue - fall back to template mode.
                draft = self._template_listing(product)
        else:
            draft = self._template_listing(product)

        self.dl.log_signal("listing_generated", draft.model_dump())
        return draft


_engine_singleton: ListingGeneratorEngine = None


def get_listing_engine() -> ListingGeneratorEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ListingGeneratorEngine()
    return _engine_singleton
