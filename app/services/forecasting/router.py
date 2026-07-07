from fastapi import APIRouter
from typing import List
from app.services.forecasting.engine import get_forecasting_engine
from app.shared.schemas import ForecastPoint

router = APIRouter(prefix="/forecasting", tags=["Demand Forecasting"])


@router.get("/{sku}", response_model=List[ForecastPoint])
def forecast(sku: str, periods: int = 7):
    return get_forecasting_engine().forecast(sku, periods=periods)


@router.get("/{sku}/reorder-alert")
def reorder_alert(sku: str):
    return get_forecasting_engine().reorder_alert(sku)
