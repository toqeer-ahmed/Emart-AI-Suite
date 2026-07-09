"""
E-Mart AI Suite - Gateway
==========================
Single entry point for every AI capability described in the Unified AI
Marketplace Platform proposal, running as one deployable add-on:

    Search . Recommendations . Shopping Assistant . Fraud Detection
    . Demand Forecasting . Sentiment Analysis . Photo Studio . Listing Generator

Run it:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs covering
every endpoint below.

HOW THIS PLUGS INTO E-MART
---------------------------
This whole package is designed to run as a SIDECAR service next to your
existing Laravel/Firebase/MySQL stack - it does not replace anything.
E-Mart's backend calls this service over HTTP (see README.md for the
concrete integration points: product sync, checkout hook, review sync).
Nothing here requires you to change E-Mart's existing database schema.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os

from app.config import CORS_ORIGINS, GATEWAY_API_KEY, API_KEY_HEADER_NAME
from app.shared.seed_data import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed()  # safe to run repeatedly; upserts demo catalog + reviews
    try:
        from app.services.assistant.engine import get_assistant_engine
        get_assistant_engine().start_background_tasks()
    except Exception:
        pass
    yield
    # Clean up the listing generator engine singleton session on shutdown
    try:
        from app.services.listing_generator.engine import get_listing_engine
        get_listing_engine().close()
    except Exception:
        pass
    # Clean up the shopping assistant engine singleton session on shutdown
    try:
        from app.services.assistant.engine import get_assistant_engine
        get_assistant_engine().close()
    except Exception:
        pass


from app.services.fraud.router import router as fraud_router
from app.services.search.router import router as search_router
from app.services.recommendations.router import router as recommendations_router
from app.services.sentiment.router import router as sentiment_router
from app.services.forecasting.router import router as forecasting_router
from app.services.assistant.router import router as assistant_router
from app.services.photo_studio.router import router as photo_studio_router
from app.services.listing_generator.router import router as listing_router
from app.orchestrator.router import router as orchestrator_router

app = FastAPI(
    title="E-Mart AI Suite",
    description="Unified AI add-on: Search, Recommendations, Shopping Assistant, "
                 "Fraud Detection, Demand Forecasting, Sentiment Analysis, "
                 "Photo Studio, and Listing Generator behind one gateway.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "services", "photo_studio", "output")
os.makedirs(PHOTO_OUTPUT_DIR, exist_ok=True)
app.mount("/static/photo-studio", StaticFiles(directory=PHOTO_OUTPUT_DIR), name="photo-studio-static")


@app.middleware("http")
async def gateway_auth(request: Request, call_next):
    """AI Gateway responsibility from the proposal: single point of auth
    enforcement before any request reaches an AI service. Disabled by
    default (GATEWAY_API_KEY unset) for local dev/demo; set
    EMART_AI_GATEWAY_KEY in production and have E-Mart's backend send it
    on every call."""
    if GATEWAY_API_KEY and request.url.path not in ("/", "/docs", "/openapi.json", "/redoc", "/health"):
        provided = request.headers.get(API_KEY_HEADER_NAME)
        if provided != GATEWAY_API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid AI gateway key"})
    return await call_next(request)


@app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
def get_dashboard():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "emart-ai-suite"}


# Every AI capability, mounted behind this one gateway:
app.include_router(search_router)
app.include_router(recommendations_router)
app.include_router(assistant_router)
app.include_router(fraud_router)
app.include_router(forecasting_router)
app.include_router(sentiment_router)
app.include_router(photo_studio_router)
app.include_router(listing_router)
app.include_router(orchestrator_router)
