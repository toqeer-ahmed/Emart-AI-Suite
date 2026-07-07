"""
AI Orchestrator
================
This is the piece the whole package is built around: it's what makes this
a *unified* AI tool instead of eight separate scripts. Every cross-service
workflow from the proposal lives here - the individual service routers
stay simple and single-purpose; this module is the only place that calls
more than one service and combines the results.

Two orchestrated flows are implemented:

  1. `run_checkout()` - Stages 6-9 (cart -> checkout -> fraud -> confirmation)
  2. `landing_page()` - Stage 1 (recognition + personalized homepage,
     cold-start-safe)

Both follow the "Decision routing" responsibility described in the
proposal: the Orchestrator decides which service(s) to call and what to do
with disagreeing signals (e.g. fraud says hold, but it's a first-time
customer - decide accordingly), rather than any one service deciding alone.
"""
from __future__ import annotations
from typing import Dict, Any

from app.services.search.engine import get_search_engine
from app.services.recommendations.engine import get_recommendation_engine
from app.services.fraud.engine import get_fraud_engine
from app.services.sentiment.engine import get_sentiment_engine
from app.services.forecasting.engine import get_forecasting_engine
from app.shared.data_layer import get_data_layer
from app.shared.schemas import OrderContext


class Orchestrator:
    def __init__(self):
        self.search = get_search_engine()
        self.recs = get_recommendation_engine()
        self.fraud = get_fraud_engine()
        self.sentiment = get_sentiment_engine()
        self.forecasting = get_forecasting_engine()
        self.dl = get_data_layer()

    # ---------------------------------------------------------------
    # Stage 1: Landing & Recognition
    # ---------------------------------------------------------------
    def landing_page(self, customer_id: str) -> Dict[str, Any]:
        history = self.dl.get_customer_events(customer_id)
        is_returning = len(history) > 0
        recommendations = self.recs.for_customer(customer_id, top_k=6)
        return {
            "customer_id": customer_id,
            "is_returning_customer": is_returning,
            "homepage_recommendations": [r.model_dump() for r in recommendations],
            "mode": "personalized" if is_returning else "cold_start_popularity",
        }

    # ---------------------------------------------------------------
    # Stage 5: Product page (search result + sentiment + stock-aware)
    # ---------------------------------------------------------------
    def product_detail(self, sku: str) -> Dict[str, Any]:
        product = self.dl.get_product(sku)
        if not product:
            return {"error": "unknown sku"}
        sentiment = self.sentiment.summarize_product(sku)
        reorder = self.forecasting.reorder_alert(sku)
        return {
            "product": product,
            "sentiment_summary": sentiment.model_dump(),
            "inventory_outlook": reorder,
        }

    # ---------------------------------------------------------------
    # Stages 6-9: Cart -> Checkout -> Fraud -> Confirmation
    # ---------------------------------------------------------------
    def run_checkout(self, order: OrderContext) -> Dict[str, Any]:
        cart_total = sum(item.quantity * item.unit_price for item in order.cart)
        item_count = sum(item.quantity for item in order.cart)

        # Stage 6: cart-context cross-sell (shown to customer before they
        # finish checkout, doesn't block the flow below)
        cross_sell = self.recs.cart_cross_sell([i.sku for i in order.cart])

        # Stage 7/8: passive risk signal collection + fraud scoring.
        # This is the "decision routing" responsibility from the proposal:
        # combine the instant transaction-shaped check with the
        # customer's behavioral history, and don't let either signal alone
        # make the final call.
        quick_check = self.fraud.quick_transaction_check(order.customer_id, cart_total, item_count)
        behavioral = self.fraud.score_customer(order.customer_id)

        final_decision = self._combine_fraud_signals(quick_check, behavioral)

        # Log the order as events for future fraud/recommendation training
        # (Section 7 continuous learning loop).
        import uuid
        invoice_no = f"INV-{uuid.uuid4().hex[:10]}"
        for item in order.cart:
            self.dl.add_order_event({
                "invoice_no": invoice_no, "customer_id": order.customer_id,
                "sku": item.sku, "description": "", "quantity": item.quantity,
                "unit_price": item.unit_price, "invoice_date": __import__("datetime").datetime.utcnow().isoformat(),
                "country": order.country,
            })
            self.dl.log_signal("order_placed", {"invoice_no": invoice_no, "sku": item.sku})

        # Stage 9: confirmation copy, tone depends on the fraud decision
        if final_decision["decision"] == "auto_approve":
            message = f"Your order {invoice_no} is confirmed and being prepared for shipment."
        elif final_decision["decision"] == "step_up_verification":
            message = f"Your order {invoice_no} needs a quick verification step before we can ship it."
        else:
            message = f"Your order {invoice_no} has been placed and is under review before shipment."

        return {
            "invoice_no": invoice_no,
            "cart_total": round(cart_total, 2),
            "cross_sell_suggestions": [r.model_dump() for r in cross_sell],
            "fraud_check": final_decision,
            "confirmation_message": message,
        }

    def _combine_fraud_signals(self, quick_check: Dict[str, Any], behavioral) -> Dict[str, Any]:
        """Decision routing: escalate to the stricter of the two signals,
        but never silently auto-approve a first-time high-value order just
        because the behavioral model has no history for that customer yet."""
        quick_risk = quick_check["risk"]
        behavioral_decision = behavioral.decision

        severity = {"auto_approve": 0, "low": 0, "step_up_verification": 1, "medium": 1, "hold_for_review": 2, "high": 2}

        quick_sev = severity.get(quick_risk, 0)
        behavioral_sev = severity.get(behavioral_decision, 0)

        if max(quick_sev, behavioral_sev) == 0:
            decision = "auto_approve"
        elif max(quick_sev, behavioral_sev) == 1:
            decision = "step_up_verification"
        else:
            decision = "hold_for_review"

        return {
            "decision": decision,
            "quick_check": quick_check,
            "behavioral_check": behavioral.model_dump(),
        }


_orchestrator_singleton: Orchestrator = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = Orchestrator()
    return _orchestrator_singleton
