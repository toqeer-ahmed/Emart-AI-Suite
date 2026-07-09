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
"""
from __future__ import annotations
import os
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, Any, List
import requests

from app.shared.data_layer import get_data_layer
from app.shared.schemas import ListingDraft

class ProductNotFoundError(ValueError):
    """Raised when a product with the requested SKU is not found in the database."""
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

logger = logging.getLogger("app.services.listing_generator.engine")


class ListingGeneratorEngine:
    def __init__(self):
        self.dl = get_data_layer()
        self.session = requests.Session()

    def close(self):
        """Clean up requests session connection pool."""
        try:
            self.session.close()
        except Exception as e:
            logger.warning("Error during session closing: %s", str(e))

    def _extract_json(self, text: str) -> dict:
        """
        Robustly extracts the first valid JSON object from the text response using brace counting.
        Tolerates surrounding conversational text, markdown blocks, and cleans trailing commas.
        """
        if not text.strip():
            raise ValueError("LLM returned an empty response text.")

        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object structure detected in LLM response.")

        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            raise ValueError("Mismatched or incomplete JSON braces in LLM response.")

        raw_json = text[start_idx:end_idx + 1].strip()

        raw_json = re.sub(r",\s*([\]\}])", r"\1", raw_json)

        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON validation and parsing failed: {e}")

    def _map_to_draft(self, product: Dict[str, Any], data: Dict[str, Any]) -> ListingDraft:
        """
        Safely maps arbitrary parsed JSON fields to the structured ListingDraft.
        Tolerates missing keys and unexpected schemas by using aliases and defaults.
        """
        sku = str(product.get("sku") or "").strip()
        product_name = str(product.get("name") or "Unnamed Product").strip()

        title_val = data.get("title", data.get("headline", data.get("product_title", product_name)))
        title = str(title_val or "").strip()
        if not title:
            title = f"{product_name} - Standard Edition"
        
        if len(title) > 80:
            title = title[:77] + "..."

        raw_bullets = data.get("bullet_features", data.get("bullet_points", data.get("bulletPoints", data.get("features", []))))
        if not isinstance(raw_bullets, list):
            raw_bullets = [str(raw_bullets)] if raw_bullets else []
        else:
            raw_bullets = [str(b) for b in raw_bullets if b is not None]

        selling_points = data.get("selling_points", data.get("key_selling_points", []))
        if not isinstance(selling_points, list):
            selling_points = [str(selling_points)] if selling_points else []
        else:
            selling_points = [str(sp) for sp in selling_points if sp is not None]

        bullets: List[str] = []
        for b in raw_bullets:
            if b.strip():
                bullets.append(b.strip())
        for sp in selling_points:
            if sp.strip():
                bullets.append(f"Benefit: {sp.strip()}")

        bullets = [b for b in bullets if b][:6]
        if not bullets:
            bullets = [f"Optimized for use as a {str(product.get('category') or 'product')} utility."]

        short_desc = str(data.get("short_description") or data.get("marketing_description") or "").strip()
        long_desc = str(data.get("long_description") or data.get("product_description") or "").strip()
        cta = str(data.get("call_to_action") or data.get("cta") or "").strip()

        desc_parts = []
        if short_desc:
            desc_parts.append(short_desc)
        if long_desc:
            desc_parts.append(long_desc)
        if cta:
            desc_parts.append(cta)

        description = "\n\n".join(desc_parts)
        if not description:
            description = str(product.get("description") or "")

        keywords = data.get("keywords", data.get("seo_keywords", []))
        if not isinstance(keywords, list):
            keywords = [str(keywords)] if keywords else []
        else:
            keywords = [str(k) for k in keywords if k is not None]

        tags = data.get("tags", data.get("suggested_product_tags", []))
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        else:
            tags = [str(t) for t in tags if t is not None]

        clean_keywords = []
        for k in (keywords + tags):
            val = k.lower().strip()
            if val:
                clean_keywords.append(val)

        seo_keywords = sorted(list(dict.fromkeys(clean_keywords)))

        return ListingDraft(
            sku=sku,
            title=title,
            bullet_points=bullets,
            description=description,
            seo_keywords=seo_keywords
        )

    def _template_listing(self, product: Dict[str, Any]) -> ListingDraft:
        """
        Generates high-quality, non-repetitive descriptions and bullet features
        dynamically from the product specs without generic marketing boilerplate.
        """
        sku = str(product.get("sku") or "").strip()
        name = str(product.get("name") or "Unnamed Product").strip()
        category = str(product.get("category") or "product").strip()
        raw_description = str(product.get("description") or "").strip()

        tags = product.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = [t.strip() for t in tags.split(",") if t.strip()]
        
        clean_tags = [t.lower().strip() for t in tags if isinstance(t, str) and t.strip()]

        if clean_tags:
            title = f"{name} - Engineered for {category.title()} | {clean_tags[0].title()} Focus"
        else:
            title = f"{name} - Optimized for {category.title()} Applications"

        if len(title) > 80:
            title = title[:77] + "..."

        desc_sentences = [s.strip() for s in raw_description.split('.') if s.strip()]
        feature_sentence = desc_sentences[0] if desc_sentences else f"an option within the {category} catalog"

        short_desc = f"The {name} is specifically designed to support {category} functions, utilizing advanced attributes to deliver consistent results."
        long_desc = f"Constructed for practical use, this item handles daily requirements effectively. It highlights key properties related to {feature_sentence.lower()}."
        if clean_tags:
            long_desc += f" Additional characteristics include {', '.join(clean_tags[:3])} features."

        cta = f"Order the {name} to resolve your {category} requirements."
        description = f"{short_desc}\n\n{long_desc}\n\n{cta}".strip()

        bullets: List[str] = [
            f"Integrates directly with {category} operations and workloads.",
            f"Engineered around the core specifications of {name}."
        ]
        if clean_tags:
            bullets.append(f"Features targeted capabilities including {', '.join(clean_tags[:3])}.")
        if product.get("stock", 0) > 0:
            bullets.append(f"Available inventory in stock: {product['stock']} units ready for distribution.")
        if product.get("price"):
            bullets.append(f"Offered at a competitive marketplace value of ${product['price']}.")

        category_benefits = {
            "electronics": "Engineered with modern circuitry for seamless component integration.",
            "home": "Designed for ergonomic space optimization and convenient home integration.",
            "fashion": "Tailored with comfortable materials structured for prolonged wear.",
            "sports": "Built for durability and resilience during athletic and outdoor activities."
        }
        benefit = category_benefits.get(category.lower(), "Provides stable utility matching standard industry specifications.")
        bullets.append(benefit)
        bullets = [b for b in bullets if b][:6]

        keywords_set = {category}
        for word in name.split():
            clean_word = re.sub(r"[^\w]", "", word.lower())
            if clean_word and len(clean_word) > 3:
                keywords_set.add(clean_word)
        for t in clean_tags:
            keywords_set.add(t)

        seo_keywords = sorted(list(dict.fromkeys(keywords_set)))

        return ListingDraft(
            sku=sku,
            title=title,
            bullet_points=bullets,
            description=description,
            seo_keywords=seo_keywords
        )

    def _llm_listing(self, product: Dict[str, Any]) -> ListingDraft:
        """
        Queries Claude using System instructions for professional, structured e-commerce copywriting.
        Uses connection-pooled Session requests with a tight connection/read timeout window.
        """
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic key is missing.")

        system_instruction = "You are an experienced e-commerce copywriter."
        sku = str(product.get("sku") or "").strip()

        prompt = f"""Write a structured marketplace listing for this product.

Product Name: {product.get('name', 'Unnamed Product')}
Category: {product.get('category', '')}
Raw Description: {product.get('description', '')}
Price: {product.get('price', '')}
Tags: {product.get('tags', '')}

You MUST follow these strict guidelines:
1. Generate an SEO-optimized title (maximum 80 characters, no clickbait, naturally include the category, avoid keyword stuffing).
2. Generate natural marketing copy with a professional tone. Avoid exaggerated claims and unverifiable statements. Write naturally. Do not invent product specifications that were not supplied.
3. Use Amazon-style bullets for bullet_features.
4. Use NO emojis, NO markdown formatting, and NO HTML.
5. Return ONLY a valid JSON object with the keys specified below. Do not wrap the JSON object in a conversational intro or outro.

Expected JSON schema:
{{
  "title": "SEO Optimized Title",
  "short_description": "Short marketing description (1 sentence)",
  "long_description": "Detailed description (2-3 sentences)",
  "bullet_features": ["list of up to 6 key features, matching product information quality"],
  "selling_points": ["list of key selling points or benefits"],
  "keywords": ["list of search keywords"],
  "tags": ["list of suggested product tags"],
  "call_to_action": "Call to action text"
}}
"""

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1000,
            "system": system_instruction,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            resp = self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=(5, 20)
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as he:
            status_code = getattr(he.response, "status_code", "Unknown")
            logger.warning("LLM API call failed for SKU %s. HTTP Status Code: %s", sku, status_code)
            raise

        response_data = resp.json()
        text = response_data["content"][0]["text"].strip()

        data = self._extract_json(text)
        if not isinstance(data, dict):
            raise ValueError(f"Extracted JSON block is not a dictionary. Got: {type(data).__name__}")

        return self._map_to_draft(product, data)

    def generate(self, sku: str) -> ListingDraft:
        """
        Coordinates database fetching, mode routing, error boundary transitions,
        and continuous training signal ingestion. Handles logs with strict exclusions.
        """
        start_time = time.time()
        product = self.dl.get_product(sku)
        if not product:
            raise ProductNotFoundError(f"Unknown SKU: {sku}")

        success = False
        mode = "template"
        draft = None

        if ANTHROPIC_API_KEY:
            try:
                logger.info("LLM mode activated for SKU: %s", sku)
                draft = self._llm_listing(product)
                success = True
                mode = "llm"
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "LLM generation failed for SKU: %s. Mode: llm. Elapsed: %.2fs. Success: False. Reason: %s",
                    sku, elapsed, str(type(e).__name__)
                )

        if not draft:
            logger.info("Template mode activated for SKU: %s", sku)
            try:
                draft = self._template_listing(product)
                success = True
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "Template generation failed for SKU: %s. Mode: template. Elapsed: %.2fs. Success: False. Reason: %s",
                    sku, elapsed, str(e)
                )
                raise

        elapsed_time = time.time() - start_time
        logger.info(
            "Listing draft generation complete. SKU: %s, Mode: %s, Success: %s, Elapsed: %.2fs",
            sku, mode, str(success), elapsed_time
        )

        try:
            self.dl.log_signal(
                "listing_generated",
                {
                    "mode": mode,
                    "sku": sku,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.warning("Failed to log continuous learning signal for SKU: %s. Error: %s", sku, str(e))

        return draft


_engine_singleton: ListingGeneratorEngine | None = None


def get_listing_engine() -> ListingGeneratorEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ListingGeneratorEngine()
    return _engine_singleton
