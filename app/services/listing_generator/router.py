from fastapi import APIRouter, HTTPException
from app.services.listing_generator.engine import get_listing_engine
from app.shared.schemas import ListingDraft

router = APIRouter(prefix="/listing-generator", tags=["Listing Generator"])


@router.get("/{sku}", response_model=ListingDraft)
def generate_listing(sku: str):
    try:
        return get_listing_engine().generate(sku)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
