from fastapi import APIRouter
from pydantic import BaseModel
from app.services.sentiment.engine import get_sentiment_engine
from app.shared.schemas import SentimentSummary
from app.shared.data_layer import get_data_layer

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])


class ReviewSubmission(BaseModel):
    sku: str
    text: str
    rating: int


@router.get("/product/{sku}", response_model=SentimentSummary)
def product_sentiment(sku: str):
    return get_sentiment_engine().summarize_product(sku)


@router.post("/reviews")
def submit_review(review: ReviewSubmission):
    get_data_layer().add_review(review.sku, review.text, review.rating)
    return {"status": "review recorded"}
