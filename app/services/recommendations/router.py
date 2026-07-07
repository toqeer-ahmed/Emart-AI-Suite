from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from app.services.recommendations.engine import get_recommendation_engine
from app.shared.schemas import RecommendationResult

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


class CartCrossSellRequest(BaseModel):
    cart_skus: List[str]


@router.get("/for-customer/{customer_id}", response_model=List[RecommendationResult])
def for_customer(customer_id: str, top_k: int = 5):
    return get_recommendation_engine().for_customer(customer_id, top_k=top_k)


@router.post("/cart-cross-sell", response_model=List[RecommendationResult])
def cart_cross_sell(req: CartCrossSellRequest):
    return get_recommendation_engine().cart_cross_sell(req.cart_skus)
