"""
Fraud Detection Engine
=======================
Wraps the actual trained artifacts from your notebook
(`Ecommerce_Abnormal_Purchase_Refund_Detection.ipynb`):

  - fraud_detection_model.pkl -> sklearn IsolationForest
        contamination=0.05, n_estimators=100, random_state=42
  - scaler.pkl -> sklearn StandardScaler fit on the same 12 features

IMPORTANT - what this model actually is:
This is a CUSTOMER-level behavioral anomaly model (it was trained on
aggregated per-customer purchase/return history from the UCI e-commerce
dataset), NOT a per-transaction real-time fraud score. That means:

  - It answers "does this customer's overall buying/returning pattern look
    abnormal", which fits Stage 9/continuous-risk-monitoring and the
    "customer risk tier" idea in your proposal.
  - It does NOT by itself answer "is THIS specific checkout fraudulent" the
    way a per-transaction PyOD model would for Stage 8. For that, this
    module also exposes a lightweight rule-based real-time check
    (`quick_transaction_check`) that runs instantly at checkout and can be
    swapped for a proper per-transaction PyOD/XGBoost model later without
    changing the API contract.

The 12 features, in the exact order the scaler/model expect:
    total_transactions, total_items_bought, total_items_returned,
    total_amount_spent, total_amount_returned, unique_products_bought,
    avg_basket_value, avg_item_price, return_rate, refund_to_spend_ratio,
    tenure_days, returns_per_day
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from app.shared.data_layer import get_data_layer
from app.shared.schemas import FraudCheckResult

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

FEATURE_ORDER = [
    "total_transactions", "total_items_bought", "total_items_returned",
    "total_amount_spent", "total_amount_returned", "unique_products_bought",
    "avg_basket_value", "avg_item_price", "return_rate",
    "refund_to_spend_ratio", "tenure_days", "returns_per_day",
]


class FraudEngine:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_model.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.dl = get_data_layer()

    # ---------- feature engineering (mirrors the training notebook) ----------
    def _build_customer_features(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        if not events:
            return None

        bought = [e for e in events if e["quantity"] > 0]
        returned = [e for e in events if e["quantity"] < 0 or str(e["invoice_no"]).startswith("C")]

        total_items_bought = sum(e["quantity"] for e in bought)
        total_items_returned = abs(sum(e["quantity"] for e in returned))
        total_amount_spent = sum(e["quantity"] * e["unit_price"] for e in bought)
        total_amount_returned = abs(sum(e["quantity"] * e["unit_price"] for e in returned))

        invoices = set(e["invoice_no"] for e in events)
        skus = set(e["sku"] for e in events)

        dates = [datetime.fromisoformat(e["invoice_date"]) if isinstance(e["invoice_date"], str) else e["invoice_date"] for e in events]
        tenure_days = max((max(dates) - min(dates)).days, 0) if len(dates) > 1 else 0

        total_items_handled = total_items_bought + total_items_returned
        return_rate = (total_items_returned / total_items_handled) if total_items_handled else 0.0
        refund_to_spend_ratio = total_amount_returned / (total_amount_spent + 0.01)
        avg_basket_value = (total_amount_spent / len(invoices)) if invoices else 0.0
        avg_item_price = (sum(e["unit_price"] for e in events) / len(events)) if events else 0.0
        returns_per_day = total_items_returned / (tenure_days + 1)

        return {
            "total_transactions": len(invoices),
            "total_items_bought": total_items_bought,
            "total_items_returned": total_items_returned,
            "total_amount_spent": total_amount_spent,
            "total_amount_returned": total_amount_returned,
            "unique_products_bought": len(skus),
            "avg_basket_value": avg_basket_value,
            "avg_item_price": avg_item_price,
            "return_rate": round(return_rate, 4),
            "refund_to_spend_ratio": round(refund_to_spend_ratio, 4),
            "tenure_days": tenure_days,
            "returns_per_day": round(returns_per_day, 2),
        }

    def _risk_tier(self, is_anomaly: bool, score: float) -> tuple[str, str]:
        # score: lower (more negative) = more anomalous, per IsolationForest.decision_function
        if is_anomaly and score < -0.05:
            return "high", "hold_for_review"
        if is_anomaly:
            return "medium", "step_up_verification"
        return "low", "auto_approve"

    def score_customer(self, customer_id: str) -> FraudCheckResult:
        events = self.dl.get_customer_events(customer_id)
        features = self._build_customer_features(events)

        if features is None:
            # Cold start: no history yet -> can't be scored by a behavioral
            # model. Fall back to "low risk, monitor" per the proposal's
            # cold-start mitigation strategy (never silently auto-approve
            # blind, never silently block a first-time buyer either).
            result = FraudCheckResult(
                customer_id=customer_id,
                is_anomaly=False,
                anomaly_score=0.0,
                risk_tier="unknown_new_customer",
                decision="step_up_verification",
                contributing_factors=["no purchase history available yet"],
            )
            self.dl.log_signal("fraud_decision", result.model_dump())
            return result

        vector_df = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        scaled = self.scaler.transform(vector_df)
        pred = self.model.predict(scaled)[0]          # 1 = normal, -1 = anomaly
        score = float(self.model.decision_function(scaled)[0])
        is_anomaly = pred == -1

        tier, decision = self._risk_tier(is_anomaly, score)

        factors = []
        if features["return_rate"] > 0.5:
            factors.append(f"high return rate ({features['return_rate']*100:.0f}% of items)")
        if features["refund_to_spend_ratio"] > 1.0:
            factors.append("refunds exceed total spend (financial clawback pattern)")
        if features["returns_per_day"] > 1:
            factors.append("high-velocity returns")
        if not factors and is_anomaly:
            factors.append("behavioral profile statistically unusual vs customer base")

        result = FraudCheckResult(
            customer_id=customer_id,
            is_anomaly=bool(is_anomaly),
            anomaly_score=round(score, 4),
            risk_tier=tier,
            decision=decision,
            contributing_factors=factors,
        )
        # Continuous learning hook (Section 7 of the proposal): every
        # decision is logged so the model can be retrained on outcomes later.
        self.dl.log_signal("fraud_decision", result.model_dump())
        return result

    def quick_transaction_check(self, customer_id: str, cart_total: float, item_count: int) -> Dict[str, Any]:
        """Instant rule-based check for the checkout moment (Stage 8),
        meant to run alongside (not instead of) the behavioral model above.
        Swap this function's body for a real per-transaction PyOD/XGBoost
        model when you have labeled transaction-level fraud data - the
        return contract stays the same so nothing upstream needs to change."""
        flags = []
        if cart_total > 1000:
            flags.append("unusually large order value")
        if item_count > 50:
            flags.append("unusually high item count in single order")

        customer_history = self.dl.get_customer_events(customer_id)
        if not customer_history and cart_total > 300:
            flags.append("first-time customer with high-value order")

        risk = "high" if len(flags) >= 2 else ("medium" if flags else "low")
        return {"risk": risk, "flags": flags}


_engine_singleton: FraudEngine = None


def get_fraud_engine() -> FraudEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = FraudEngine()
    return _engine_singleton
