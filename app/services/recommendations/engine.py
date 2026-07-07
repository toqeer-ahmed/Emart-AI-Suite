"""
Recommendation Engine
======================
Implements the same behavior LightFM would give you (personalized when
there's history, popularity-based when there isn't) using order-event
co-occurrence, which works from day one with zero training step.

UPGRADE PATH: once you have enough order_events logged, swap `_related()`'s
body for an actual LightFM model (collaborative + content-based hybrid) -
keep the same function signatures so the Orchestrator/Gateway don't need
changes.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Optional

from app.shared.data_layer import get_data_layer
from app.shared.schemas import RecommendationResult


class RecommendationEngine:
    def __init__(self):
        self.dl = get_data_layer()

    def _popularity_ranking(self, exclude_skus: Optional[set] = None) -> List[RecommendationResult]:
        products = self.dl.list_products()
        exclude_skus = exclude_skus or set()
        # Cold start (Section 10 mitigation): no interaction data yet ->
        # popularity fallback using stock + price as weak proxy signals
        # until real interaction volume builds up.
        ranked = sorted(products, key=lambda p: p.get("stock", 0), reverse=True)
        return [
            RecommendationResult(sku=p["sku"], name=p["name"], price=p["price"], reason="popular")
            for p in ranked if p["sku"] not in exclude_skus
        ][:5]

    def _co_purchase_map(self):
        """Build a simple co-purchase count from logged order events:
        which SKUs tend to appear in the same invoice."""
        by_invoice = defaultdict(set)
        # Pull events across all customers we have signals for
        signal_events = self.dl.get_signals("order_placed", limit=5000)
        for sig in signal_events:
            import json
            payload = json.loads(sig["payload"]) if isinstance(sig["payload"], str) else sig["payload"]
            invoice = payload.get("invoice_no")
            sku = payload.get("sku")
            if invoice and sku:
                by_invoice[invoice].add(sku)

        co_counts = defaultdict(Counter)
        for skus in by_invoice.values():
            skus = list(skus)
            for i, sku_a in enumerate(skus):
                for sku_b in skus:
                    if sku_a != sku_b:
                        co_counts[sku_a][sku_b] += 1
        return co_counts

    def for_customer(self, customer_id: str, top_k: int = 5) -> List[RecommendationResult]:
        history = self.dl.get_customer_events(customer_id)
        if not history:
            return self._popularity_ranking()[:top_k]

        bought_skus = {e["sku"] for e in history}
        co_counts = self._co_purchase_map()

        candidates = Counter()
        for sku in bought_skus:
            for related_sku, count in co_counts.get(sku, {}).items():
                if related_sku not in bought_skus:
                    candidates[related_sku] += count

        if not candidates:
            return self._popularity_ranking(exclude_skus=bought_skus)[:top_k]

        results = []
        for sku, _ in candidates.most_common(top_k):
            product = self.dl.get_product(sku)
            if product:
                results.append(RecommendationResult(
                    sku=sku, name=product["name"], price=product["price"],
                    reason="frequently bought together with items in your history",
                ))
        if len(results) < top_k:
            filler = self._popularity_ranking(exclude_skus=bought_skus | {r.sku for r in results})
            results.extend(filler[: top_k - len(results)])
        return results

    def cart_cross_sell(self, cart_skus: List[str], top_k: int = 3) -> List[RecommendationResult]:
        """Stage 6 'Add to Cart' cross-sell mode."""
        co_counts = self._co_purchase_map()
        candidates = Counter()
        for sku in cart_skus:
            for related_sku, count in co_counts.get(sku, {}).items():
                if related_sku not in cart_skus:
                    candidates[related_sku] += count

        results = []
        for sku, _ in candidates.most_common(top_k):
            product = self.dl.get_product(sku)
            if product:
                results.append(RecommendationResult(
                    sku=sku, name=product["name"], price=product["price"],
                    reason="pairs well with items in your cart",
                ))
        if len(results) < top_k:
            filler = self._popularity_ranking(exclude_skus=set(cart_skus) | {r.sku for r in results})
            results.extend(filler[: top_k - len(results)])
        return results


_engine_singleton: RecommendationEngine = None


def get_recommendation_engine() -> RecommendationEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = RecommendationEngine()
    return _engine_singleton
