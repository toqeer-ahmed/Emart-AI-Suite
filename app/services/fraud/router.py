from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.fraud.engine import get_fraud_engine
from app.shared.schemas import FraudCheckResult, OrderEvent
from app.shared.data_layer import get_data_layer

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])


class QuickCheckRequest(BaseModel):
    customer_id: str
    cart_total: float
    item_count: int


@router.get("/score/{customer_id}", response_model=FraudCheckResult)
def score_customer(customer_id: str):
    """Behavioral anomaly score for a customer, using the trained
    Isolation Forest. Use this for account-level risk review / Stage 9
    ongoing monitoring."""
    try:
        return get_fraud_engine().score_customer(customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-check")
def quick_check(req: QuickCheckRequest):
    """Instant rule-based check for the live checkout moment (Stage 8).
    Pair this with /score/{customer_id} for the full picture: quick-check
    catches transaction-shaped anomalies right now, /score catches
    behavioral patterns building up over time."""
    return get_fraud_engine().quick_transaction_check(
        req.customer_id, req.cart_total, req.item_count
    )


@router.post("/events")
def add_event(event: OrderEvent):
    """Ingest a raw order/return event into the shared data layer. This is
    how E-Mart's checkout/refund flow feeds the model new data - call this
    on every order placed and every return processed."""
    dl = get_data_layer()
    dl.add_order_event(event.model_dump())
    dl.log_signal("order_placed", {"invoice_no": event.invoice_no, "sku": event.sku})
    return {"status": "logged"}
