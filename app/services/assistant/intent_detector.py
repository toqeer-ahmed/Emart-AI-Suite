"""
Intent Detector Module
======================
Separates intent parsing strategy rules from the main orchestrator engine,
making individual parsing logic isolated, pluggable, and unit-testable.
"""
from __future__ import annotations
import re
from typing import List

class IntentStrategy:
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        """Parses the message and updates intent dict if matched. Returns True if intent type changes."""
        return False


class BudgetIntentStrategy(IntentStrategy):
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        currency_pattern = r"(?:\$|€|£|rs\.?|pkr)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|eur|gbp|pkr|rs)"
        budget_match = re.search(r"\b(?:under|less\s+than|below|max|maximum)\s+" + currency_pattern, msg_lower)
        matched = False
        if budget_match:
            intent["type"] = "budget"
            val = budget_match.group(1) or budget_match.group(2)
            if val:
                intent["budget_limit"] = float(val)
                intent["sort_by_price_asc"] = True
                matched = True

        if not intent["budget_limit"]:
            if any(word in msg_lower for word in ["cheap", "budget", "affordable", "inexpensive"]):
                intent["type"] = "budget"
                intent["sort_by_price_asc"] = True
                matched = True
        return matched


class ComparisonIntentStrategy(IntentStrategy):
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        comparison_patterns = [
            r"compare\s+(.+)\s+and\s+(.+)",
            r"(.+)\s+vs\s+(.+)",
            r"(.+)\s+versus\s+(.+)",
            r"difference\s+between\s+(.+)\s+and\s+(.+)"
        ]
        for pattern in comparison_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                intent["type"] = "comparison"
                intent["comparison"] = True
                sub1 = re.sub(r"(?i)\bcompare\b|\bdifference\s+between\b", "", match.group(1)).strip()
                sub2 = match.group(2).strip()
                if sub1 and sub2:
                    intent["comparison_subjects"] = [sub1, sub2]
                return True

        if not intent["comparison"] and any(word in msg_lower for word in ["better", "difference", "compare", "best", "versus", "vs"]):
            intent["type"] = "comparison"
            intent["comparison"] = True
            return True
        return False


class GiftIntentStrategy(IntentStrategy):
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        if any(word in msg_lower for word in ["gift", "present", "birthday", "anniversary", "christmas"]):
            intent["type"] = "gift"
            intent["gift"] = True
            return True
        return False


class CategoryIntentStrategy(IntentStrategy):
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        def normalize_word(w: str) -> str:
            w = w.lower()
            if w.endswith("ies"):
                return w[:-3] + "y"
            if w.endswith("s") and not w.endswith("ss"):
                return w[:-1]
            return w

        message_words = {normalize_word(w) for w in re.findall(r"\b\w+\b", msg_lower)}
        for cat in catalog_categories:
            cat_norm = normalize_word(cat)
            cat_words = {normalize_word(w) for w in cat.lower().split()}
            if cat_norm in msg_lower or cat_words.issubset(message_words):
                intent["matched_category"] = cat
                if intent["type"] == "general":
                    intent["type"] = "category"
                return True
        return False


class RecommendationIntentStrategy(IntentStrategy):
    def parse(self, msg_lower: str, catalog_categories: List[str], intent: dict) -> bool:
        if any(word in msg_lower for word in ["recommend", "suggest", "options for", "show me", "alternatives"]):
            if intent["type"] == "general":
                intent["type"] = "recommendation"
            intent["recommendation"] = True
            return True
        return False


class IntentDetector:
    def __init__(self, catalog_categories: List[str]):
        self.catalog_categories = catalog_categories
        self.strategies: List[IntentStrategy] = [
            BudgetIntentStrategy(),
            ComparisonIntentStrategy(),
            GiftIntentStrategy(),
            CategoryIntentStrategy(),
            RecommendationIntentStrategy()
        ]

    def detect(self, message: str) -> dict:
        """Parses raw user input message into intent profile parameters."""
        msg_lower = message.lower()
        intent = {
            "type": "general",
            "budget_limit": None,
            "sort_by_price_asc": False,
            "matched_category": None,
            "comparison": False,
            "comparison_subjects": [],
            "gift": False,
            "recommendation": False
        }
        for strategy in self.strategies:
            strategy.parse(msg_lower, self.catalog_categories, intent)
        return intent
