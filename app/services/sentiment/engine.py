"""
Sentiment Engine
=================
Uses VADER (rule-based, no model download, works fully offline) to give a
real per-review sentiment score, plus simple keyword-bucket aspect
extraction (delivery, quality, price, packaging) so you get the
"Great quality, mixed on delivery speed" style summary from the proposal
without needing a GPU or a HuggingFace model download at build time.

UPGRADE PATH: swap `_score_text()` for a HuggingFace
`pipeline("sentiment-analysis")` / fine-tuned RoBERTa when you want
aspect-level nuance beyond keyword buckets - same function signature.
"""
from __future__ import annotations
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.shared.data_layer import get_data_layer
from app.shared.schemas import SentimentSummary

ASPECT_KEYWORDS = {
    "delivery": ["delivery", "shipping", "arrived", "shipment", "courier"],
    "quality": ["quality", "material", "build", "durable", "broke", "sturdy"],
    "price": ["price", "expensive", "cheap", "value", "worth"],
    "packaging": ["packaging", "box", "damaged", "package"],
    "sizing": ["size", "fit", "small", "large", "tight", "loose"],
}


class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.dl = get_data_layer()

    def _score_text(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)["compound"]  # -1..1

    def _label(self, compound: float) -> str:
        if compound >= 0.3:
            return "positive"
        if compound <= -0.3:
            return "negative"
        return "neutral"

    def _extract_aspects(self, reviews: List[str]) -> Dict[str, str]:
        aspects = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            matching_sentences = [
                r for r in reviews if any(k in r.lower() for k in keywords)
            ]
            if matching_sentences:
                avg_score = sum(self._score_text(s) for s in matching_sentences) / len(matching_sentences)
                aspects[aspect] = self._label(avg_score)
        return aspects

    def summarize_product(self, sku: str) -> SentimentSummary:
        reviews = self.dl.get_reviews(sku)
        texts = [r["text"] for r in reviews]

        if not texts:
            return SentimentSummary(
                sku=sku, overall_sentiment="neutral", positive_ratio=0.0,
                aspects={}, sample_size=0,
            )

        scores = [self._score_text(t) for t in texts]
        positive_count = sum(1 for s in scores if s >= 0.3)
        overall_score = sum(scores) / len(scores)

        result = SentimentSummary(
            sku=sku,
            overall_sentiment=self._label(overall_score),
            positive_ratio=round(positive_count / len(texts), 2),
            aspects=self._extract_aspects(texts),
            sample_size=len(texts),
        )
        # Recommendation re-weighting signal (Section 5 of the proposal):
        # declining sentiment can suppress a product in ranking later.
        self.dl.log_signal("sentiment_summary", result.model_dump())
        return result

    def score_single_review(self, text: str) -> Dict[str, float]:
        return self.analyzer.polarity_scores(text)


_engine_singleton: SentimentEngine = None


def get_sentiment_engine() -> SentimentEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SentimentEngine()
    return _engine_singleton
