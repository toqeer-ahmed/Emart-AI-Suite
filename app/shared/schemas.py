"""
Shared request/response contracts.

Every service in the suite imports from here instead of defining its own
one-off models. This is what lets the Orchestrator pass a single
`OrderContext` through Search -> Recommendations -> Fraud -> Assistant
without each service inventing its own shape for "a customer" or "an order".
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Product(BaseModel):
    sku: str
    name: str
    description: str = ""
    category: str = ""
    price: float
    stock: int = 0
    vendor_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class CartItem(BaseModel):
    sku: str
    quantity: int
    unit_price: float


class Customer(BaseModel):
    customer_id: str
    country: Optional[str] = None
    account_age_days: Optional[int] = None
    is_returning: bool = False


class OrderEvent(BaseModel):
    """One line-item event, matching the shape the fraud model was trained on
    (UCI-style e-commerce transaction log): invoice, sku, qty, unit price, customer, date."""
    invoice_no: str
    customer_id: str
    sku: str
    description: str = ""
    quantity: int
    unit_price: float
    invoice_date: datetime
    country: Optional[str] = None


class OrderContext(BaseModel):
    """The single object the Orchestrator threads through every stage of
    checkout (Stage 6-9 of the customer workflow)."""
    customer_id: str
    cart: List[CartItem]
    country: Optional[str] = None
    session_query: Optional[str] = None  # last search/chat query, for context


class FraudCheckResult(BaseModel):
    customer_id: str
    is_anomaly: bool
    anomaly_score: float
    risk_tier: str  # "low" | "medium" | "high"
    decision: str   # "auto_approve" | "step_up_verification" | "hold_for_review"
    contributing_factors: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    sku: str
    name: str
    price: float
    score: float
    in_stock: bool
    description: str = ""
    category: str = ""
    stock: int = 0


class RecommendationResult(BaseModel):
    sku: str
    name: str
    price: float
    reason: str  # e.g. "popular", "frequently bought with X", "similar to Y"
    description: str = ""
    category: str = ""
    stock: int = 0


class SentimentSummary(BaseModel):
    sku: str
    overall_sentiment: str  # "positive" | "mixed" | "negative"
    positive_ratio: float
    aspects: Dict[str, str] = Field(default_factory=dict)  # e.g. {"delivery": "negative"}
    sample_size: int


class ForecastPoint(BaseModel):
    period: str
    forecast_units: float
    lower_bound: float
    upper_bound: float


class ListingDraft(BaseModel):
    sku: str
    title: str
    bullet_points: List[str]
    description: str
    seo_keywords: List[str] = Field(default_factory=list)


class PhotoEnhanceRequest(BaseModel):
    sku: str
    image_url: str
    style: str = "studio_white"  


class PhotoEnhanceResult(BaseModel):
    sku: str
    original_url: str
    enhanced_url: str
    style: str
    status: str


class AssistantTurn(BaseModel):
    customer_id: str
    message: str
    session_id: Optional[str] = None


class AssistantReply(BaseModel):
    reply: str
    suggested_products: List[RecommendationResult] = Field(default_factory=list)
    used_services: List[str] = Field(default_factory=list)
