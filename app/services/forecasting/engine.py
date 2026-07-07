"""
Demand Forecasting Engine
==========================
Uses Meta's Prophet model to generate 7-day demand forecasts from daily unit sales
logged in order_events. When the order history contains fewer than 5 data points,
it falls back to a naive forecast to ensure cold-start safety.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
import logging
from typing import List
import pandas as pd
from prophet import Prophet

from app.shared.data_layer import get_data_layer
from app.shared.schemas import ForecastPoint

# Suppress cmdstanpy verbose logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


class ForecastingEngine:
    def __init__(self):
        self.dl = get_data_layer()

    def _daily_units(self, sku: str) -> dict:
        # Pull from order_events via the signals log for simplicity in the
        # SQLite dev layer
        daily = defaultdict(int)
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

        # Prophet needs at least a few historical points to compute trend and seasonality.
        # Fall back to naive forecast if history is too sparse.
        if len(daily) < 5:
            product = self.dl.get_product(sku)
            base = 1.0
            points = []
            for i in range(periods):
                day = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                points.append(ForecastPoint(
                    period=day, 
                    forecast_units=base, 
                    lower_bound=0.0, 
                    upper_bound=base * 3
                ))
            return points

        # Prepare DataFrame for Prophet (requires columns 'ds' and 'y')
        df_data = [{"ds": datetime.strptime(date, "%Y-%m-%d"), "y": float(qty)} for date, qty in daily.items()]
        df = pd.DataFrame(df_data)

        # Fit Prophet model
        # Disable yearly/daily seasonality for sparse data, keep weekly seasonality
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)

        # Predict future
        future = m.make_future_dataframe(periods=periods, include_history=False)
        forecast_df = m.predict(future)

        points = []
        for _, row in forecast_df.iterrows():
            period_str = row["ds"].strftime("%Y-%m-%d")
            yhat = float(row["yhat"])
            # Bound predictions to positive values
            forecast_units = max(round(yhat, 2), 0.0)
            lower_bound = max(round(float(row["yhat_lower"]), 2), 0.0)
            upper_bound = max(round(float(row["yhat_upper"]), 2), 0.0)
            
            points.append(ForecastPoint(
                period=period_str,
                forecast_units=forecast_units,
                lower_bound=lower_bound,
                upper_bound=upper_bound
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

