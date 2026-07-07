# E-Mart AI Suite

A single, self-contained AI add-on that unifies every module discussed for
E-Mart's AI roadmap — Search, Recommendations, Shopping Assistant, Fraud
Detection, Demand Forecasting, Sentiment Analysis, Photo Studio, and
Listing Generator — behind **one gateway**, following the orchestrator
architecture from the original proposal.

It runs as a **sidecar service**: E-Mart's existing Laravel/Firebase/MySQL
stack keeps doing what it does, and calls this service over plain HTTP for
AI features. Nothing here requires touching E-Mart's schema or rewriting
any existing code.

```
                     ┌─────────────────────────────────────────┐
                     │           E-Mart AI Suite (this repo)     │
                     │                                            │
  E-Mart backend --> │  Gateway (auth, CORS)                     │
  (Laravel/Firebase/ │      │                                     │
   MySQL - unchanged)│  Orchestrator (workflow_router.py)         │
                     │      │                                     │
                     │  ┌───┴────────────────────────────────┐   │
                     │  │ Search │ Recs │ Assistant │ Fraud   │   │
                     │  │ Forecasting │ Sentiment │ Photo     │   │
                     │  │ Studio │ Listing Generator          │   │
                     │  └───┬────────────────────────────────┘   │
                     │      │                                     │
                     │  Shared Data Layer (SQLite by default,     │
                     │  swap for your MySQL/Firestore adapter)    │
                     └─────────────────────────────────────────┘
```

## What's real vs. what's a documented upgrade path

Being direct about this, since it matters for how you plan the rollout:

| Service | What's running right now | Upgrade path when ready |
|---|---|---|
| **Fraud Detection** | Your actual trained `IsolationForest` + `StandardScaler` from the uploaded notebook, wired to real feature engineering | Add a per-transaction PyOD/XGBoost model for real-time Stage-8 scoring (see `quick_transaction_check`) |
| **Search** | Real TF-IDF + cosine similarity over the shared product catalog | Swap for Marqo/Elasticsearch vector search |
| **Recommendations** | Real popularity + co-purchase mining from logged orders, cold-start safe | Swap for a trained LightFM model once you have enough interaction volume |
| **Sentiment** | Real VADER sentiment scoring + keyword-based aspect extraction | Swap for a fine-tuned HuggingFace Transformer for finer aspect nuance |
| **Forecasting** | Real linear-trend forecast from logged order events | Swap for Prophet once you have a few months of real sales history |
| **Shopping Assistant** | Real orchestration of Search + Recommendations into a reply; template-based by default, or live Claude API calls if you set `ANTHROPIC_API_KEY` | Add LangChain/Dify if you want a visual flow builder for non-developers |
| **Photo Studio** | Real image processing (auto-contrast, sharpening, white-balance, square canvas) via Pillow | Swap for `rembg` or a hosted background-removal API for full AI background swap |
| **Listing Generator** | Template-based copywriting by default, or live Claude API calls if you set `ANTHROPIC_API_KEY` | No change needed — this already calls a real LLM when a key is configured |

Every "upgrade path" is a single function body swap inside one `engine.py`
file per service — the router, the schema, and everything that calls it
stays the same. That's the entire point of the modular design.

## Running it

```bash
cd emart-ai-suite
cp .env.example .env         # fill in ANTHROPIC_API_KEY if you want live LLM mode
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

Then open **http://localhost:8000/docs** — every endpoint below is live
and callable from there immediately, seeded with demo products so you can
test without connecting real data first.

Run the test suite:

```bash
pytest tests/test_smoke.py -v
```

## Endpoint reference

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/search?q=...` | Product search |
| GET | `/recommendations/for-customer/{id}` | Personalized or cold-start recs |
| POST | `/recommendations/cart-cross-sell` | Cart-context cross-sell (Stage 6) |
| POST | `/assistant/chat` | Shopping assistant reply (Stage 4) |
| GET | `/fraud/score/{customer_id}` | Behavioral anomaly score (your trained model) |
| POST | `/fraud/quick-check` | Instant rule-based checkout check (Stage 8) |
| POST | `/fraud/events` | Log a raw order/return event for the fraud model to learn from |
| GET | `/forecasting/{sku}` | 7-day demand forecast |
| GET | `/forecasting/{sku}/reorder-alert` | Stock-vs-demand reorder flag |
| GET | `/sentiment/product/{sku}` | Aspect-level review sentiment summary |
| POST | `/sentiment/reviews` | Submit a new review |
| POST | `/photo-studio/enhance` | Clean up a vendor product photo |
| GET | `/listing-generator/{sku}` | Generate marketplace listing copy |
| GET | `/workflow/landing/{customer_id}` | **Orchestrated** Stage 1 homepage |
| GET | `/workflow/product/{sku}` | **Orchestrated** Stage 5 product page (sentiment + inventory) |
| POST | `/workflow/checkout` | **Orchestrated** Stages 6-9: cross-sell + fraud decision + confirmation |

The `/workflow/*` endpoints are the ones that actually demonstrate
"unified" — they call multiple services and combine the results, exactly
like the AI Orchestrator described in the original architecture proposal.

## Integration points for E-Mart specifically

Given E-Mart's Firebase/MySQL hybrid, here's where this plugs in without
touching your existing schema:

1. **Product sync** — whenever a vendor creates/updates a listing in
   E-Mart, call `PUT`-style upsert into the AI suite's catalog (add a thin
   `POST /admin/sync-product` endpoint if you want this pushed, or point
   `app/shared/data_layer.py`'s `EmartDataLayer` directly at your MySQL
   `products` table — see the docstring in that file for the exact
   interface to implement).

2. **Checkout hook** — replace whatever ad-hoc fraud/cross-sell logic
   exists in E-Mart's checkout controller with one call to
   `POST /workflow/checkout`. It returns cross-sell suggestions, the fraud
   decision, and confirmation copy in one response.

3. **Review sync** — when a customer submits a review in E-Mart, forward
   it to `POST /sentiment/reviews` so sentiment summaries build up
   automatically; show `GET /sentiment/product/{sku}` on the product page.

4. **Vendor dashboard** — wire `POST /photo-studio/enhance` and
   `GET /listing-generator/{sku}` into the vendor upload flow as
   "Enhance Photo" / "Generate Listing" buttons — this is the direct
   equivalent of the resale-photo-enhancer / Amazon-listing-generator
   tools discussed earlier, but native to E-Mart instead of a separate
   third-party SaaS.

5. **Auth** — set `EMART_AI_GATEWAY_KEY` in production and have E-Mart's
   backend send it as the `X-EMart-AI-Key` header. The suite's own
   middleware rejects any request without it, so this AI layer is never
   directly exposed to end users — only your backend talks to it.

## Swapping in E-Mart's real database

Everything in this suite programs against the abstract `DataLayer`
interface in `app/shared/data_layer.py`, never a specific database. To
connect real E-Mart data instead of the SQLite demo:

```python
# app/shared/data_layer.py
class EmartDataLayer(DataLayer):
    def __init__(self, mysql_conn, firestore_client):
        self.mysql = mysql_conn
        self.fs = firestore_client

    def get_product(self, sku):
        # SELECT * FROM products WHERE sku = %s
        ...

    def log_signal(self, signal_type, payload):
        # write to a Firestore `ai_signals` collection or a MySQL
        # `ai_events` table, whichever you already use for analytics
        ...
```

Then change one line in `app/shared/data_layer.py`'s `get_data_layer()` to
return `EmartDataLayer(...)` instead of `SQLiteDataLayer()`. No other file
in the package needs to change — every router and engine only ever calls
the abstract interface.

## Continuous learning loop

Every AI decision (fraud verdict, sentiment summary, listing generated,
photo enhanced, assistant reply) gets written to the `ai_signals` table via
`DataLayer.log_signal()`. That's the hook for the "improve every 30 days"
principle from the original proposal — once you're ready, a scheduled job
can read these signals back out (`get_signals(signal_type)`) and retrain
whichever model needs it, without any change to how the live services work
in the meantime.
