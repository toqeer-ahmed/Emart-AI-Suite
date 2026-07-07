"""
Demand Forecasting Engine
==========================
Uses a simple linear-trend + seasonal-naive blend over daily unit sales
logged in order_events. This needs no heavy dependency (Prophet pulls in
cmdstanpy/pystan and is a fairly large install), so the add-on stays
lightweight by default.

UPGRADE PATH: once you have a few months of real E-Mart sales history,
swap `forecast()`'s body for actual Prophet:

    from prophet import Prophet
    m = Prophet()
    m.fit(daily_df)  # requires columns 'ds' and 'y'
    future = m.make_future_dataframe(periods=periods)
    forecast_df = m.predict(future)

Keep the return shape (List[ForecastPoint]) identical and nothing else
needs to change.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

from app.shared.data_layer import get_data_layer
from app.shared.schemas import ForecastPoint


class ForecastingEngine:
    def __init__(self):
        self.dl = get_data_layer()

    def _daily_units(self, sku: str) -> dict:
        # Pull from order_events via the signals log for simplicity in the
        # SQLite demo layer; a real adapter would query order_events
        # directly filtered by sku.
        all_products = self.dl.list_products()
        daily = defaultdict(int)
        # NOTE: in this reference implementation we approximate using
        # ai_signals("order_placed") entries logged at checkout time, since
        # SQLiteDataLayer.get_customer_events is keyed by customer, not sku.
        # A production adapter should query order_events directly by sku
        # and date for accurate daily aggregates.
        signals = self.dl.get_signals("order_placed", limit=5000)
        import json
        for s in signals:
            payload = json.loads(s["payload"]) if isinstance(s["payload"], str) else s["payload"]
            if payload.get("sku") == sku:
                day = s["created_at"][:10]
                daily[day] += 1
        return daily

    def forecast(self, sku: str, periods: int = 7) -> List[ForecastPoint]:
        daily = self._daily_units(sku)

        if len(daily) < 2:
            # Not enough history yet: flat forecast using current stock
            # velocity assumption of 1 unit/day as a safe placeholder,
            # clearly marked as low-confidence via wide bounds.
            product = self.dl.get_product(sku)
            base = 1.0
            points = []
            for i in range(periods):
                day = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                points.append(ForecastPoint(period=day, forecast_units=base, lower_bound=0, upper_bound=base * 3))
            return points

        values = list(daily.values())
        n = len(values)
        avg = sum(values) / n
        # simple linear trend via least squares on index
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = avg
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs) or 1
        slope = num / den

        points = []
        for i in range(periods):
            day = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            point_forecast = max(avg + slope * (n + i), 0)
            points.append(ForecastPoint(
                period=day,
                forecast_units=round(point_forecast, 2),
                lower_bound=round(max(point_forecast * 0.6, 0), 2),
                upper_bound=round(point_forecast * 1.4, 2),
            ))
        return points

    def reorder_alert(self, sku: str) -> dict:
        """Inventory -> Forecasting collaboration pattern: flag SKUs whose
        projected demand will outrun current stock within the forecast
        window (Stage 2/Search 'filter OOS' + Section 5 shared data usage)."""
        product = self.dl.get_product(sku)
        if not product:
            return {"sku": sku, "error": "unknown sku"}
        forecast_points = self.forecast(sku, periods=7)
        projected_demand = sum(p.forecast_units for p in forecast_points)
        stock = product.get("stock", 0)
        needs_reorder = projected_demand > stock
        return {
            "sku": sku,
            "current_stock": stock,
            "projected_7day_demand": round(projected_demand, 1),
            "needs_reorder": needs_reorder,
        }


_engine_singleton: ForecastingEngine = None


def get_forecasting_engine() -> ForecastingEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ForecastingEngine()
    return _engine_singleton
