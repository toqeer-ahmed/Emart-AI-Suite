from fastapi import APIRouter
from app.orchestrator.orchestrator import get_orchestrator
from app.shared.schemas import OrderContext

router = APIRouter(prefix="/workflow", tags=["Orchestrated Workflow"])


@router.get("/landing/{customer_id}")
def landing(customer_id: str):
    """Stage 1: personalized homepage / cold-start popularity fallback."""
    return get_orchestrator().landing_page(customer_id)


@router.get("/product/{sku}")
def product_detail(sku: str):
    """Stage 5: product page with sentiment summary + inventory outlook."""
    return get_orchestrator().product_detail(sku)


@router.post("/checkout")
def checkout(order: OrderContext):
    """Stages 6-9: cross-sell, fraud decision routing, order confirmation -
    the single endpoint E-Mart's checkout flow should call."""
    return get_orchestrator().run_checkout(order)
