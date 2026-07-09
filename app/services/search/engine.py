"""
Search Engine
=============
This is a genuinely functional TF-IDF + cosine-similarity search over the
shared product catalog - not a mock. It runs entirely offline (no model
download, no external API), which makes it a safe default for a
self-contained add-on.

UPGRADE PATH: swap this for Marqo/Elasticsearch by replacing `search()`'s
body - keep the function signature identical and nothing else in the
package needs to change, since the Orchestrator and Gateway only ever call
`SearchEngine.search(query)`.
"""
from __future__ import annotations
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.shared.data_layer import get_data_layer
from app.shared.schemas import SearchResult


class SearchEngine:
    def __init__(self):
        self.dl = get_data_layer()
        self._vectorizer = None
        self._matrix = None
        self._products = []
        self._build_index()

    def _build_index(self):
        self._products = self.dl.list_products()
        corpus = [
            f"{p['name']} {p.get('description', '')} {p.get('category', '')} {p.get('tags', '')}"
            for p in self._products
        ]
        if not corpus:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def refresh(self):
        """Call after products change (new vendor upload, price update,
        etc.) - cheap enough to call often for a catalog this size; for a
        large catalog, schedule this on a timer or on a product-change
        webhook instead of every request."""
        self._build_index()

    def search(self, query: str, top_k: int = 5, exclude_out_of_stock: bool = True) -> List[SearchResult]:
        if not self._products:
            self.refresh()
        if not self._products or self._vectorizer is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        ranked = sorted(zip(self._products, scores), key=lambda x: x[1], reverse=True)

        results = []
        for product, score in ranked:
            if score <= 0:
                continue
            in_stock = product.get("stock", 0) > 0
            if exclude_out_of_stock and not in_stock:
                # Forecasting -> Search collaboration pattern from the
                # proposal: filter out-of-stock items so customers aren't
                # disappointed.
                continue
            results.append(SearchResult(
                sku=product["sku"], name=product["name"],
                price=product["price"], score=round(float(score), 4),
                in_stock=in_stock,
                description=product.get("description", ""),
                category=product.get("category", ""),
                stock=product.get("stock", 0)
            ))
            if len(results) >= top_k:
                break
        return results

    def list_categories(self) -> List[str]:
        """Returns a sorted list of unique categories available in the catalog."""
        if not self._products:
            self.refresh()
        return sorted(list(set(p.get("category") for p in self._products if p.get("category"))))

    def list_products(self) -> List[dict]:
        """Returns all products currently loaded in the search index."""
        if not self._products:
            self.refresh()
        return self._products


_engine_singleton: SearchEngine = None


def get_search_engine() -> SearchEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SearchEngine()
    return _engine_singleton
