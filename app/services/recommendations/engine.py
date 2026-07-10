"""
Recommendation Engine
======================
Implements a Singular Value Decomposition (SVD) Matrix Factorization model
for personalized product recommendations, and uses item embedding cosine similarity
for cart cross-sells. Falls back to co-purchase frequency maps and stock-based
popularity when user-item interaction histories are sparse (cold-start mitigation).
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Optional
import numpy as np

from app.shared.data_layer import get_data_layer
from app.shared.schemas import RecommendationResult


class RecommendationEngine:
    def __init__(self):
        self.dl = get_data_layer()
        self.user_factors = {}
        self.item_factors = {}
        self.item_skus = []
        self.user_ids = []

    def _fit_collaborative_filtering(self) -> bool:
        """Trains pure-Python SVD Matrix Factorization on historical order events."""
        try:
            events = self.dl.get_all_order_events()
            if not events:
                return False

            # Extract unique users and items with valid values
            users = list(set(e["customer_id"] for e in events if e.get("customer_id")))
            items = list(set(e["sku"] for e in events if e.get("sku")))

            # We need a minimum size matrix to perform factorization
            if len(users) < 2 or len(items) < 2:
                return False

            user_to_idx = {u: idx for idx, u in enumerate(users)}
            item_to_idx = {i: idx for idx, i in enumerate(items)}

            # Build interaction matrix (Purchase Counts)
            R = np.zeros((len(users), len(items)))
            for e in events:
                u_idx = user_to_idx.get(e["customer_id"])
                i_idx = item_to_idx.get(e["sku"])
                if u_idx is not None and i_idx is not None:
                    R[u_idx, i_idx] += float(e.get("quantity", 1))

            # Set K latent factors (max 4 or matrix bounds)
            K = min(len(users), len(items), 4)

            # SVD: R = U * Sigma * Vt
            U, Sigma, Vt = np.linalg.svd(R, full_matrices=False)

            # Keep top K components
            U_k = U[:, :K]
            Sigma_k = np.diag(Sigma[:K])
            Vt_k = Vt[:K, :]

            # Compute user and item embeddings
            P = np.dot(U_k, np.sqrt(Sigma_k))
            Q = np.dot(Vt_k.T, np.sqrt(Sigma_k))

            # Cache mappings and factor vectors
            self.user_ids = users
            self.item_skus = items
            self.user_factors = {users[idx]: P[idx] for idx in range(len(users))}
            self.item_factors = {items[idx]: Q[idx] for idx in range(len(items))}
            return True
        except Exception:
            return False

    def _popularity_ranking(self, exclude_skus: Optional[set] = None) -> List[RecommendationResult]:
        products = self.dl.list_products()
        exclude_skus = exclude_skus or set()
        ranked = sorted(products, key=lambda p: p.get("stock", 0), reverse=True)
        return [
            RecommendationResult(
                sku=p["sku"], name=p["name"], price=p["price"], reason="popular",
                description=p.get("description", ""), category=p.get("category", ""),
                stock=p.get("stock", 0)
            )
            for p in ranked if p["sku"] not in exclude_skus
        ][:5]

    def _co_purchase_map(self):
        """Builds a simple co-purchase count from logged order events."""
        by_invoice = defaultdict(set)
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

    def _co_purchase_cross_sell(self, cart_skus: List[str], top_k: int = 3) -> List[RecommendationResult]:
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
                    description=product.get("description", ""), category=product.get("category", ""),
                    stock=product.get("stock", 0)
                ))
        if len(results) < top_k:
            filler = self._popularity_ranking(exclude_skus=set(cart_skus) | {r.sku for r in results})
            results.extend(filler[: top_k - len(results)])
        return results

    def for_customer(self, customer_id: str, top_k: int = 5) -> List[RecommendationResult]:
        # Perform SVD fitting on current catalog transactions
        success = self._fit_collaborative_filtering()
        history = self.dl.get_customer_events(customer_id)

        # Cold start fallback if factorization isn't ready or user has no purchase history
        if not success or not history or customer_id not in self.user_factors:
            return self._popularity_ranking()[:top_k]

        bought_skus = {e["sku"] for e in history}
        user_vector = self.user_factors[customer_id]

        # Calculate scores for all factorization items
        scores = []
        for sku, item_vector in self.item_factors.items():
            if sku in bought_skus:
                continue
            score = float(np.dot(user_vector, item_vector))
            scores.append((sku, score))

        # Sort candidate items descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sku, score in scores[:top_k]:
            product = self.dl.get_product(sku)
            if product:
                results.append(RecommendationResult(
                    sku=sku, name=product["name"], price=product["price"],
                    reason="personalized recommendation based on your purchase history",
                    description=product.get("description", ""), category=product.get("category", ""),
                    stock=product.get("stock", 0)
                ))

        if len(results) < top_k:
            filler = self._popularity_ranking(exclude_skus=bought_skus | {r.sku for r in results})
            results.extend(filler[: top_k - len(results)])
        return results

    def cart_cross_sell(self, cart_skus: List[str], top_k: int = 3) -> List[RecommendationResult]:
        success = self._fit_collaborative_filtering()
        
        # Fall back to co-purchase counts if collaborative filtering is not fit
        if not success or not self.item_factors:
            return self._co_purchase_cross_sell(cart_skus, top_k)

        # Extract embeddings for items in the cart
        valid_cart_vectors = [self.item_factors[sku] for sku in cart_skus if sku in self.item_factors]
        if not valid_cart_vectors:
            return self._co_purchase_cross_sell(cart_skus, top_k)

        # Calculate average cart embedding centroid
        cart_center = np.mean(valid_cart_vectors, axis=0)
        cart_center_norm = np.linalg.norm(cart_center)

        scores = []
        for sku, item_vector in self.item_factors.items():
            if sku in cart_skus:
                continue
            item_norm = np.linalg.norm(item_vector)
            if cart_center_norm > 0 and item_norm > 0:
                # Cosine Similarity between catalog item and cart average embedding
                similarity = float(np.dot(cart_center, item_vector) / (cart_center_norm * item_norm))
            else:
                similarity = 0.0
            scores.append((sku, similarity))

        # Sort items descending by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sku, sim in scores[:top_k]:
            product = self.dl.get_product(sku)
            if product:
                results.append(RecommendationResult(
                    sku=sku, name=product["name"], price=product["price"],
                    reason="frequently bought together with items in your cart",
                    description=product.get("description", ""), category=product.get("category", ""),
                    stock=product.get("stock", 0)
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
