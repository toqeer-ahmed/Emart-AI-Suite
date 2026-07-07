"""
Sentiment Engine
=================
Uses a Hugging Face pre-trained sentiment analysis model (DistilBERT SST-2)
to give a real per-review sentiment score, plus simple keyword-bucket aspect
extraction (delivery, quality, price, packaging).
"""
from __future__ import annotations
from typing import List, Dict
from transformers import pipeline

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
        # Load the pre-trained Hugging Face sentiment-analysis pipeline (DistilBERT)
        self.pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.dl = get_data_layer()

    def _score_text(self, text: str) -> float:
        if not text.strip():
            return 0.0
        try:
            res = self.pipeline(text)[0]
            label = res["label"].upper()
            score = res["score"]
            # Map label and confidence score to range [-1.0, 1.0]
            if label == "POSITIVE":
                return float(score)
            elif label == "NEGATIVE":
                return -float(score)
            return 0.0
        except Exception:
            return 0.0

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
        score = self._score_text(text)
        return {
            "neg": max(-score, 0.0),
            "neu": 1.0 - abs(score),
            "pos": max(score, 0.0),
            "compound": score
        }


_engine_singleton: SentimentEngine = None


def get_sentiment_engine() -> SentimentEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SentimentEngine()
    return _engine_singleton

