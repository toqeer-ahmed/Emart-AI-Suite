from fastapi import APIRouter, HTTPException
import logging
import time
from app.services.listing_generator.engine import get_listing_engine, ProductNotFoundError
from app.shared.schemas import ListingDraft

router = APIRouter(prefix="/listing-generator", tags=["Listing Generator"])
logger = logging.getLogger("app.services.listing_generator.router")


@router.get("/{sku}", response_model=ListingDraft)
def generate_listing(sku: str):
    logger.info("Listing generation request received for SKU: %s", sku)
    start_time = time.time()
    try:
        draft = get_listing_engine().generate(sku)
        elapsed = time.time() - start_time
        logger.info("Listing generation request succeeded for SKU: %s. Elapsed: %.2fs", sku, elapsed)
        return draft
    except ProductNotFoundError as e:
        logger.warning("Listing generation SKU not found for SKU: %s. Error: %s", sku, str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Listing generation request failed for SKU: %s. Error: %s", sku, str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


