"""
Shopping Assistant Engine
==========================
This is the "LangChain + Dify" role from the proposal: it takes a customer
message, calls Search and Recommendations (tool use), and turns the
combined result into a conversational reply.

Two modes, chosen automatically:
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
import re
import time
import json
import logging
import uuid
import threading
import httpx
import anyio
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.services.search.engine import get_search_engine
from app.services.recommendations.engine import get_recommendation_engine
from app.shared.data_layer import get_data_layer
from app.shared.schemas import AssistantTurn, AssistantReply, RecommendationResult, SearchResult

from app.services.assistant.http_client import HttpClientManager
from app.services.assistant.intent_detector import IntentDetector

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

logger = logging.getLogger("app.services.assistant.engine")


class AssistantConfig:
    MAX_PROMPT_SEARCH = 5
    MAX_PROMPT_RECS = 5
    MAX_TEMPLATE_RESULTS = 3
    MAX_REPLY_TOKENS = 500
    TIMEOUT_CONNECT = 5
    TIMEOUT_READ = 20
    MAX_LLM_REPLY_LENGTH = 1500


@dataclass
class AssistantCandidate:
    sku: str
    name: str
    price: float
    description: str
    category: str
    stock: int
    reason: str
    score: float
    is_search: bool
    is_rec: bool
    original_rank: int


class AssistantEngine:
    def __init__(self):
        """
        Initializes engines and http manager with resilient async connection pool.
        Note: The Data Layer is used ONLY for telemetry continuous learning signals.
        Loads dynamic catalog categories and product names from Search index.
        Enforces absolute thread safety using threading.Lock.
        """
        self._lock = threading.Lock()
        
        self.search = get_search_engine()
        self.recs = get_recommendation_engine()
        self.dl = get_data_layer()
        
        self.http_client = HttpClientManager(
            connect_timeout=AssistantConfig.TIMEOUT_CONNECT,
            read_timeout=AssistantConfig.TIMEOUT_READ
        )
        
        self.catalog_categories = []
        self._catalog_product_names = []
        self._last_catalog_load = 0.0
        self.intent_detector = IntentDetector([])
        
        self._refresh_catalog_metadata()

        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.circuit_open = False
        
        self._cache_task = None
        try:
            loop = asyncio.get_running_loop()
            self._cache_task = loop.create_task(self._periodic_cache_refresh())
        except RuntimeError:
            pass
            
        logger.info("Shopping Assistant engine initialized with Async httpx client.")

    def start_background_tasks(self):
        """Lifespan callback helper to warm/initiate background tasks if loop was missing on init."""
        with self._lock:
            if self._cache_task is None or self._cache_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self._cache_task = loop.create_task(self._periodic_cache_refresh())
                    logger.info("Dynamic background cache refresh worker started successfully.")
                except RuntimeError:
                    pass

    def close(self):
        """Gracefully closes HTTP connection pools and stops background tasks on shutdown."""
        try:
            if self._cache_task:
                self._cache_task.cancel()
        except Exception:
            pass
            
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.http_client.close())
            except RuntimeError:
                pass
        except Exception as e:
            logger.warning("Error during async connection close: %s", str(e))

    def _refresh_catalog_metadata(self):
        """Synchronously populates dynamic catalog categories and product names cache."""
        with self._lock:
            try:
                self.catalog_categories = self.search.list_categories()
                self._catalog_product_names = [p["name"].lower() for p in self.search.list_products()]
                self._last_catalog_load = time.time()
                self.intent_detector = IntentDetector(self.catalog_categories)
                logger.info("Synchronous catalog metadata cached. Loaded %d products.", len(self._catalog_product_names))
            except Exception as e:
                logger.warning("Failed to refresh catalog metadata synchronously: %s", str(e))

    async def _async_refresh_catalog_metadata(self):
        """Asynchronously refreshes dynamic catalog categories and product names (no req path overhead)."""
        try:
            categories = await anyio.to_thread.run_sync(self.search.list_categories)
            products = await anyio.to_thread.run_sync(self.search.list_products)
            
            with self._lock:
                self.catalog_categories = categories
                self._catalog_product_names = [p["name"].lower() for p in products]
                self._last_catalog_load = time.time()
                self.intent_detector = IntentDetector(self.catalog_categories)
                logger.info("Asynchronous catalog metadata cache refreshed. Loaded %d products.", len(self._catalog_product_names))
        except Exception as e:
            logger.warning("Failed to refresh catalog metadata asynchronously: %s", str(e))

    async def _periodic_cache_refresh(self):
        """Asynchronously refreshes the catalog metadata cache every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300.0)
                await self._async_refresh_catalog_metadata()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Periodic cache refresh loop encountered exception: %s", str(e))

    @property
    def catalog_product_names(self) -> List[str]:
        """Thread-safe catalog product names cache getter."""
        with self._lock:
            return list(self._catalog_product_names)

    def _is_llm_healthy(self) -> bool:
        """Applies circuit breaker logic to check Anthropic API health under lock protection."""
        with self._lock:
            if self.circuit_open:
                if time.time() - self.last_failure_time > 60.0:
                    self.circuit_open = False
                    self.consecutive_failures = 0
                    return True
                return False
            return True

    def _record_llm_success(self):
        """Thread-safely resets circuit breaker parameters."""
        with self._lock:
            self.consecutive_failures = 0
            self.circuit_open = False

    def _record_llm_failure(self):
        """Thread-safely records failure and trips circuit breaker if thresholds exceeded."""
        with self._lock:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            if self.consecutive_failures >= 3:
                self.circuit_open = True
                logger.error("Claude API Circuit Breaker tripped. Fallback to templates immediately.")

    def _prepare_request_context(self, turn: AssistantTurn) -> dict:
        """Initializes request-scoped container holding all stage information."""
        intent = self.intent_detector.detect(turn.message)
        return {
            "message": turn.message,
            "intent": intent,
            "search_hits": [],
            "recommendations": [],
            "candidates": [],
            "reply": "",
            "mode": "template",
            "fallback_reason": "none"
        }

    def _execute_search(self, request_context: dict) -> tuple[List[SearchResult], bool]:
        """Executes search query exactly once per assistant request."""
        search_success = True
        try:
            search_hits = self.search.search(request_context["message"], top_k=5)
        except Exception as e:
            logger.warning("Search execution failed. Error: %s", str(e))
            search_hits = []
            search_success = False
        return search_hits, search_success

    def _execute_recommendations(self, customer_id: str) -> tuple[List[RecommendationResult], bool]:
        """Executes recommendation engine query exactly once."""
        recommendation_success = True
        try:
            recs = self.recs.for_customer(customer_id, top_k=5)
        except Exception as e:
            logger.warning("Recommendation retrieval failed. Error: %s", str(e))
            recs = []
            recommendation_success = False
        return recs, recommendation_success

    def _deduplicate(self, search_hits: List[SearchResult], recs: List[RecommendationResult]) -> List[AssistantCandidate]:
        """Merges search results and recommendation list into deterministic candidates."""
        raw_candidates_by_sku: Dict[str, AssistantCandidate] = {}
        original_rank_counter = 0
        
        for h in search_hits:
            raw_candidates_by_sku[h.sku] = AssistantCandidate(
                sku=h.sku,
                name=h.name,
                price=h.price,
                description=h.description,
                category=h.category,
                stock=h.stock,
                reason="",
                score=h.score,
                is_search=True,
                is_rec=False,
                original_rank=original_rank_counter
            )
            original_rank_counter += 1

        for r in recs:
            if r.sku in raw_candidates_by_sku:
                existing = raw_candidates_by_sku[r.sku]
                existing.is_rec = True
                existing.reason = r.reason or ""
            else:
                raw_candidates_by_sku[r.sku] = AssistantCandidate(
                    sku=r.sku,
                    name=r.name,
                    price=r.price,
                    description=r.description,
                    category=r.category,
                    stock=r.stock,
                    reason=r.reason or "",
                    score=0.0,
                    is_search=False,
                    is_rec=True,
                    original_rank=original_rank_counter
                )
                original_rank_counter += 1

        return list(raw_candidates_by_sku.values())

    def _filter_stock(self, candidates: List[AssistantCandidate]) -> List[AssistantCandidate]:
        """Removes out of stock items."""
        return [c for c in candidates if c.stock > 0]

    def _filter_budget(self, candidates: List[AssistantCandidate], budget_limit: Optional[float]) -> List[AssistantCandidate]:
        """Filters candidates exceeding the budget threshold limit."""
        if budget_limit is None:
            return candidates
        return [c for c in candidates if c.price <= budget_limit]

    def _filter_category(self, candidates: List[AssistantCandidate], matched_category: Optional[str], is_category_intent: bool) -> List[AssistantCandidate]:
        """Filters candidates to match category if category intent is active."""
        if not is_category_intent or not matched_category:
            return candidates
        cat_candidates = [c for c in candidates if c.category == matched_category]
        return cat_candidates if cat_candidates else candidates

    def _filter_comparison(self, candidates: List[AssistantCandidate], intent: dict) -> List[AssistantCandidate]:
        """Filters comparison candidates to exact compared subjects using token overlap matching."""
        if not intent["comparison"] or not intent["comparison_subjects"]:
            return candidates

        def get_best_match(subject: str, candidates_list: List[AssistantCandidate]) -> Optional[AssistantCandidate]:
            subject_tokens = set(re.findall(r"\b\w+\b", subject.lower()))
            best_cand = None
            best_overlap = 0
            for c in candidates_list:
                name_tokens = set(re.findall(r"\b\w+\b", c.name.lower()))
                desc_tokens = set(re.findall(r"\b\w+\b", c.description.lower()))
                all_tokens = name_tokens | desc_tokens
                overlap = len(subject_tokens.intersection(all_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cand = c
            return best_cand

        sub1, sub2 = intent["comparison_subjects"]
        match1 = get_best_match(sub1, candidates)
        match2 = get_best_match(sub2, candidates)
        
        if not match1 or not match2 or match1.sku == match2.sku:
            intent["type"] = "recommendation"
            return candidates
        return [match1, match2]

    def _rank_candidates(self, candidates: List[AssistantCandidate], intent: dict) -> List[AssistantCandidate]:
        """Ranks candidates based on intent prioritization rules."""
        if intent["type"] == "recommendation" or intent.get("recommendation"):
            candidates.sort(key=lambda x: (not x.is_rec, x.original_rank))
        elif intent["sort_by_price_asc"]:
            candidates.sort(key=lambda x: (x.price, -x.score, -x.is_rec, x.original_rank))
        else:
            candidates.sort(key=lambda x: (-x.score, -x.is_rec, x.price, x.original_rank))
        return candidates

    def _build_context(self, request_context: dict) -> List[AssistantCandidate]:
        """Coordinates candidate combination workflow by calling modular sub-filter helpers."""
        intent = request_context["intent"]
        
        candidates = self._deduplicate(request_context["search_hits"], request_context["recommendations"])
        candidates = self._filter_stock(candidates)
        candidates = self._filter_budget(candidates, intent["budget_limit"])
        
        if not intent["matched_category"] and request_context["search_hits"]:
            cats = [h.category for h in request_context["search_hits"] if h.category]
            if cats:
                intent["matched_category"] = max(set(cats), key=cats.count)
                if intent["type"] == "general":
                    intent["type"] = "category"
                    
        candidates = self._filter_category(candidates, intent["matched_category"], intent["type"] == "category")
        
        if intent["gift"]:
            gift_candidates = [c for c in candidates if c.is_search]
            if gift_candidates:
                recs_only = [c for c in candidates if not c.is_search]
                candidates = gift_candidates + recs_only
                
        candidates = self._filter_comparison(candidates, intent)
        candidates = self._rank_candidates(candidates, intent)
        
        return candidates

    def _format_price(self, price: float) -> str:
        """Standardized price formatter helper."""
        return f"${price:.2f}"

    def _format_product(self, c: AssistantCandidate) -> str:
        """Standardized product text formatter."""
        return f"{c.name} for {self._format_price(c.price)}"

    def _build_prompt(self, request_context: dict) -> str:
        """Helper responsible for prompt engineering."""
        candidates = request_context["candidates"]

        search_list = [c for c in candidates if c.is_search][:AssistantConfig.MAX_PROMPT_SEARCH]
        recs_list = [c for c in candidates if c.is_rec][:AssistantConfig.MAX_PROMPT_RECS]

        search_sections = []
        for p in search_list:
            truncated_desc = p.description[:150] + "..." if len(p.description) > 150 else p.description
            search_sections.append(
                f"- Name: {p.name}, SKU: {p.sku}, Price: {self._format_price(p.price)}, "
                f"Category: {p.category}, Stock: {p.stock} units. Description: {truncated_desc}"
            )
        search_context_str = "\n".join(search_sections)

        recs_sections = []
        for p in recs_list:
            recs_sections.append(
                f"- Name: {p.name}, SKU: {p.sku}, Price: {self._format_price(p.price)}, Reason: {p.reason}" if p.reason
                else f"- Name: {p.name}, SKU: {p.sku}, Price: {self._format_price(p.price)}"
            )
        recs_context_str = "\n".join(recs_sections)

        escaped_user_query = json.dumps(request_context["message"])

        prompt_parts = [
            "Customer Question",
            "-----------------",
            f"<UserQuery>\n{escaped_user_query}\n</UserQuery>",
            "\nRelevant Products",
            "-----------------",
            search_context_str or "No matching products found.",
            "\nRecommendation Candidates",
            "--------------------------",
            recs_context_str or "No recommendation candidates available.",
            "\nInstructions",
            "------------",
            "1. Recommending items: You must only recommend products listed above. Never invent products, prices, specifications, or details.",
            "2. If no products in the list match the user request, you must explicitly state that you couldn't find one. Never substitute unrelated products just to answer.",
            "3. Explain why the recommended items are suitable based on the product description and price.",
            "4. Keep your response conversational, friendly, and natural. Avoid buzzwords, exaggerated marketing claims, or unverifiable assertions.",
            "5. Response Format: Return ONLY plain text. Do NOT use markdown (no bullet lists using '-', no bold using '**', no headers using '#', no code blocks). Do NOT use HTML tags. Do NOT use emojis."
        ]
        return "\n".join(prompt_parts)

    def _template_budget(self, request_context: dict, candidates: List[AssistantCandidate]) -> str:
        """Renders budget template responses."""
        if candidates:
            names_prices = ", ".join(self._format_product(c) for c in candidates[:AssistantConfig.MAX_TEMPLATE_RESULTS])
            reply = f"For options matching your budget preferences, I recommend looking at: {names_prices}."
            rec_only = [c for c in candidates if c.is_rec and not c.is_search]
            if rec_only:
                rec_names = ", ".join(f"{r.name} ({r.reason})" if r.reason else r.name for r in rec_only[:2])
                reply += f" I also suggest checking out these personalized options: {rec_names}."
            return reply
        else:
            return f"I couldn't find any direct matches in our catalog matching '{request_context['message']}' under your price limit."

    def _template_comparison(self, candidates: List[AssistantCandidate]) -> str:
        """Renders comparison side-by-side responses."""
        c1 = candidates[0]
        c2 = candidates[1]
        c1_clause = f" matches because of: {c1.reason}" if c1.reason else ""
        c2_clause = f" matches because of: {c2.reason}" if c2.reason else ""
        reply = (
            f"Here is a comparison between the {c1.name} ({self._format_price(c1.price)}) and the {c2.name} ({self._format_price(c2.price)}). "
            f"The {c1.name} is described as: {c1.description}.{c1_clause} "
            f"In comparison, the {c2.name} features: {c2.description}.{c2_clause}"
        )
        return reply

    def _template_gift(self, candidates: List[AssistantCandidate]) -> str:
        """Renders gift template responses."""
        if candidates:
            names_prices = ", ".join(f"{c.name} priced at {self._format_price(c.price)}" for c in candidates[:2])
            reply = f"For gift ideas, these items from our catalog make great choices: {names_prices}."
            recs_list = [c for c in candidates if c.is_rec][:2]
            if recs_list:
                rec_names = ", ".join(f"{r.name} ({r.reason})" if r.reason else r.name for r in recs_list)
                reply += f" Based on user history, you could also consider: {rec_names}."
            return reply
        else:
            return "If you are looking for a gift, please let me know which category or budget you prefer so I can recommend options."

    def _template_category(self, matched_category: str, candidates: List[AssistantCandidate]) -> str:
        """Renders category filter template responses."""
        if candidates:
            names_prices = ", ".join(f"{c.name} ({self._format_price(c.price)})" for c in candidates[:AssistantConfig.MAX_TEMPLATE_RESULTS])
            reply = f"In the {matched_category} category, we have several options: {names_prices}."
            other_recs = [c for c in candidates if c.is_rec and c.category != matched_category]
            if other_recs:
                other_names = ", ".join(f"{o.name} ({o.reason})" if o.reason else o.name for o in other_recs[:2])
                reply += f" Additionally, you might also like these recommended alternatives: {other_names}."
            return reply
        else:
            return f"I couldn't find items in the {matched_category} category right now."

    def _template_recommendation(self, request_context: dict, candidates: List[AssistantCandidate]) -> str:
        """Renders pure personalized suggestions templates."""
        rec_candidates = [c for c in candidates if c.is_rec][:AssistantConfig.MAX_TEMPLATE_RESULTS]
        if rec_candidates:
            names_reasons = ", ".join(f"{c.name} ({self._format_price(c.price)}) - {c.reason}" if c.reason
                                      else f"{c.name} ({self._format_price(c.price)})" for c in rec_candidates)
            reply = f"Here are some personalized suggestions: {names_reasons}."
            return reply
        elif candidates:
            names_prices = ", ".join(self._format_product(c) for c in candidates[:AssistantConfig.MAX_TEMPLATE_RESULTS])
            return f"Here are a few popular selections from our catalog: {names_prices}."
        else:
            return "I don't have any custom recommendations for your profile at the moment, but you can explore our full catalog online."

    def _template_general(self, request_context: dict, candidates: List[AssistantCandidate]) -> str:
        """Renders standard general intent template responses."""
        search_candidates = [c for c in candidates if c.is_search][:AssistantConfig.MAX_TEMPLATE_RESULTS]
        rec_candidates = [c for c in candidates if c.is_rec and not c.is_search][:2]
        
        if search_candidates:
            names = ", ".join(c.name for c in search_candidates)
            reply = f"Here is what I found matching '{request_context['message']}': {names}."
            if rec_candidates:
                rec_names = ", ".join(f"{r.name} ({r.reason})" if r.reason else r.name for r in rec_candidates)
                reply += f" You might also be interested in: {rec_names}."
            return reply
        elif rec_candidates:
            rec_names = ", ".join(f"{r.name} ({r.reason})" if r.reason else r.name for r in rec_candidates)
            return f"I couldn't find an exact match for '{request_context['message']}', but I highly recommend these items: {rec_names}."
        else:
            return f"I couldn't find an exact match for '{request_context['message']}' in stock right now."

    def _template_reply(self, request_context: dict) -> str:
        """Constructs rich template replies for fallback."""
        intent = request_context["intent"]
        candidates = request_context["candidates"]

        if intent["type"] == "comparison":
            if len(candidates) < 2:
                intent["type"] = "recommendation"

        if intent["type"] == "budget":
            return self._template_budget(request_context, candidates)
        elif intent["type"] == "comparison":
            return self._template_comparison(candidates)
        elif intent["type"] == "gift":
            return self._template_gift(candidates)
        elif intent["type"] == "category":
            return self._template_category(intent["matched_category"], candidates)
        elif intent["type"] == "recommendation":
            return self._template_recommendation(request_context, candidates)
        else:
            return self._template_general(request_context, candidates)

    async def _llm_reply(self, prompt: str) -> str:
        """Queries Claude Messages API via requests session with timeouts."""
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is missing.")

        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        system_instruction = (
            "You are a professional e-commerce shopping assistant for E-Mart.\n"
            "Guidelines:\n"
            "- Never invent product names, SKUs, prices, or descriptions.\n"
            "- Only recommend catalog items explicitly provided in the product lists.\n"
            "- If no product matches, explicitly state you couldn't find one. Never substitute unrelated products just to answer.\n"
            "- Explain recommendations based on real product specifications.\n"
            "- Avoid exaggerated claims, hyperbole, or unverifiable statements.\n"
            "- Return plain text only. Do NOT use markdown formatting (no bolding like **, no lists like -, no headers like #), do NOT use HTML tags, and do NOT use emojis."
        )

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": model,
            "max_tokens": AssistantConfig.MAX_REPLY_TOKENS,
            "system": system_instruction,
            "messages": [{"role": "user", "content": prompt}]
        }

        resp = await self.http_client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json_data=payload
        )
        resp.raise_for_status()
        
        response_json = resp.json()
        if not isinstance(response_json, dict) or "content" not in response_json:
            raise ValueError("Malformed Anthropic API response payload.")
        content = response_json["content"]
        if not isinstance(content, list) or len(content) == 0:
            raise ValueError("Malformed Anthropic API response: empty content array.")
        first_item = content[0]
        if not isinstance(first_item, dict) or "text" not in first_item:
            raise ValueError("Malformed Anthropic API response: missing text field.")
            
        return first_item["text"].strip()

    def _extract_response(self, response_text: str) -> str:
        """
        Sanitizes output before returning the final plain-text response.
        Removes markdown, HTML, emojis, excessive whitespace, and unsupported formatting.
        Note: Smarter regex is used to only strip leading line bullets (e.g. '- ') and preserve inline hyphens (e.g. 'USB-C', 'Wi-Fi', 'Noise-Cancelling', '2-in-1').
        """
        text = response_text

        text = re.sub(r"<[^>]+>", "", text)

        text = re.sub(r"\#+\s+", "", text)
        text = re.sub(r"\**([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        lines = []
        for line in text.split("\n"):
            cleaned_line = line.strip()
            cleaned_line = re.sub(r"^[-*+•]\s+", "", cleaned_line)
            cleaned_line = re.sub(r"^\d+\.\s+", "", cleaned_line)
            lines.append(cleaned_line)
        text = "\n".join(lines)

        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f|"
            "\U0001f300-\U0001f5ff|"
            "\U0001f680-\U0001f6ff|"
            "\U0001f1e0-\U0001f1ff|"
            "\U00002700-\U000027bf|"
            "\U00002600-\U000026ff|"
            "\U0001f900-\U0001f9ff|"
            "\U0001fa00-\U0001faff"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        
        return text.strip()

    def _is_clean_plain_text(self, text: str) -> bool:
        """Verifies that the text is clean plain text, with no markdown, HTML, or emojis."""
        if not text:
            return False
        if re.search(r"<[^>]+>", text):
            return False
        if re.search(r"\*\*|__", text):
            return False
        if re.search(r"^\s*#+\s+", text, re.M):
            return False
        if re.search(r"^\s*[-*+•]\s+", text, re.M):
            return False
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f|"
            "\U0001f300-\U0001f5ff|"
            "\U0001f680-\U0001f6ff|"
            "\U0001f1e0-\U0001f1ff|"
            "\U00002700-\U000027bf|"
            "\U00002600-\U000026ff|"
            "\U0001f900-\U0001f9ff|"
            "\U0001fa00-\U0001faff"
            "]+", flags=re.UNICODE
        )
        if emoji_pattern.search(text):
            return False
        return True

    def _validate_response(self, response_text: str, request_context: dict) -> bool:
        """
        Performs strict verification of LLM output:
        - Must not be empty.
        - Must only mention products/SKUs present in request_context candidates list.
        - Must not contain markdown formatting or HTML tags.
        - Must not contain hallucinated product names.
        """
        if not response_text:
            return False

        if len(response_text) > AssistantConfig.MAX_LLM_REPLY_LENGTH:
            return False

        if not self._is_clean_plain_text(response_text):
            return False

        candidates = request_context["candidates"]
        
        sku_mentions = re.findall(r"\b[A-Za-z0-9]+-[0-9]+\b", response_text)
        candidate_skus = {c.sku for c in candidates}
        for sku in sku_mentions:
            if sku not in candidate_skus:
                logger.warning("LLM response validation failed: mentioned SKU %s not in candidate list.", sku)
                return False

        response_lower = response_text.lower()
        candidate_names = {c.name.lower() for c in candidates}
        
        non_candidate_names = [name for name in self.catalog_product_names if name not in candidate_names]
        for name in non_candidate_names:
            if name in response_lower:
                logger.warning("LLM response validation failed: mentioned non-candidate product name %s.", name)
                return False

        return True

    async def _try_llm(self, request_context: dict) -> tuple[str, bool]:
        """Attempts LLM reply generation using Claude Messages API with health checks."""
        if not ANTHROPIC_API_KEY or not self._is_llm_healthy():
            request_context["fallback_reason"] = "template_no_api"
            return "", False

        try:
            prompt = self._build_prompt(request_context)
            raw_reply = await self._llm_reply(prompt)
            
            extracted_reply = self._extract_response(raw_reply)
            
            if self._validate_response(extracted_reply, request_context):
                self._record_llm_success()
                return extracted_reply, True
            else:
                request_context["fallback_reason"] = "template_validation"
                self._record_llm_failure()
                return "", False
        except httpx.TimeoutException:
            request_context["fallback_reason"] = "template_timeout"
            self._record_llm_failure()
        except httpx.HTTPStatusError:
            request_context["fallback_reason"] = "template_http"
            self._record_llm_failure()
        except httpx.RequestError:
            request_context["fallback_reason"] = "template_exception"
            self._record_llm_failure()
        except Exception as e:
            logger.error("LLM execution error: %s", str(e))
            request_context["fallback_reason"] = "template_exception"
            self._record_llm_failure()

        return "", False

    def _try_template(self, request_context: dict) -> tuple[str, bool]:
        """Assembles fallback response using predefined deterministic templates."""
        try:
            reply_text = self._template_reply(request_context)
            return reply_text, True
        except Exception as e:
            logger.error("Template mode failed. Error: %s", str(e))
            request_context["fallback_reason"] = "template_exception"
            return "E-Mart catalog services are temporarily unavailable.", False

    def _log_telemetry(self, request_context: dict, start_time: float, success: bool, customer_id: str, correlation_id: str):
        """Logs privacy-safe telemetry logs and continuous learning signals."""
        elapsed_time = time.time() - start_time
        logger.info(
            "Assistant completed. Cust: %s, Mode: %s, Success: %s, Elapsed: %.3fs, Reason: %s, Trace: %s",
            customer_id, request_context["mode"], str(success), elapsed_time, request_context["fallback_reason"], correlation_id
        )

        try:
            self.dl.log_signal(
                "assistant_reply",
                {
                    "customer_id": customer_id,
                    "mode": request_context["mode"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": success,
                    "fallback_reason": request_context["fallback_reason"],
                    "correlation_id": correlation_id
                }
            )
        except Exception as e:
            logger.warning("Failed to log continuous learning signal: %s", str(e))

    async def handle_turn(self, turn: AssistantTurn) -> AssistantReply:
        """Main entry point. Coordinates orchestration workflow asynchronously using request_context."""
        start_time = time.time()
        
        if not turn.message or not isinstance(turn.message, str):
            return AssistantReply(
                reply="Please enter a valid message.",
                suggested_products=[],
                used_services=["template"]
            )
        
        if len(turn.message) > 1000:
            turn.message = turn.message[:1000]
            logger.warning("User query message truncated to 1000 characters for safety.")

        correlation_id = getattr(turn, "session_id", None) or str(uuid.uuid4())
        logger.info("Shopping Assistant turn initiated. Customer ID: %s, Trace: %s", turn.customer_id, correlation_id)

        request_context = self._prepare_request_context(turn)

        search_hits, search_success = await anyio.to_thread.run_sync(self._execute_search, request_context)
        request_context["search_hits"] = search_hits

        recs, recommendation_success = await anyio.to_thread.run_sync(self._execute_recommendations, turn.customer_id)
        request_context["recommendations"] = recs

        if not search_hits and not recs:
            if not search_success or not recommendation_success:
                request_context["fallback_reason"] = "double_engine_failure" if (not search_success and not recommendation_success) else ("search_failed" if not search_success else "recommendation_failed")
                reply_text = "E-Mart catalog services are temporarily unavailable."
            else:
                request_context["fallback_reason"] = "none"
                reply_text = "I couldn't find matching products."
                
            await anyio.to_thread.run_sync(
                self._log_telemetry,
                request_context, start_time, False, turn.customer_id, correlation_id
            )
            return AssistantReply(
                reply=reply_text,
                suggested_products=[],
                used_services=["search", "recommendations", "template"]
            )

        candidates = await anyio.to_thread.run_sync(self._build_context, request_context)
        request_context["candidates"] = candidates

        reply_text, success = await self._try_llm(request_context)
        if success:
            mode = "llm"
        else:
            reply_text, _ = self._try_template(request_context)
            mode = "template"

        request_context["mode"] = mode

        suggested = self._build_suggested_products(request_context)

        await anyio.to_thread.run_sync(
            self._log_telemetry,
            request_context, start_time, success, turn.customer_id, correlation_id
        )

        return AssistantReply(
            reply=reply_text,
            suggested_products=suggested,
            used_services=["search", "recommendations", mode]
        )

    def _build_suggested_products(self, request_context: dict) -> List[RecommendationResult]:
        """Maps candidates in context to final response suggestion schemas."""
        candidates = request_context["candidates"]
        return [
            RecommendationResult(
                sku=c.sku,
                name=c.name,
                price=c.price,
                reason=c.reason,
                description=c.description,
                category=c.category,
                stock=c.stock
            )
            for c in candidates[:5]
        ]


_engine_singleton: AssistantEngine | None = None


def get_assistant_engine() -> AssistantEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = AssistantEngine()
    return _engine_singleton
