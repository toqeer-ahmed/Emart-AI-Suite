"""
Quick smoke test - not a full test suite, but enough to prove every
service starts, responds, and the orchestrator can chain them together.
Run with: python -m pytest tests/test_smoke.py -v
"""
from fastapi.testclient import TestClient
from app.main import app
from app.shared.seed_data import seed

seed()  # TestClient doesn't reliably fire startup events across versions;
        # seed explicitly here so tests are deterministic either way.
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_search():
    r = client.get("/search", params={"q": "wireless earbuds"})
    assert r.status_code == 200
    assert len(r.json()) > 0
    assert r.json()[0]["sku"] == "EM-1001"


def test_recommendations_cold_start():
    r = client.get("/recommendations/for-customer/BRAND_NEW_CUSTOMER")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_sentiment():
    r = client.get("/sentiment/product/EM-1001")
    assert r.status_code == 200
    body = r.json()
    assert body["sample_size"] > 0


def test_forecasting():
    r = client.get("/forecasting/EM-1001")
    assert r.status_code == 200
    assert len(r.json()) == 7


def test_fraud_score_new_customer():
    r = client.get("/fraud/score/UNKNOWN_CUST")
    assert r.status_code == 200
    assert r.json()["risk_tier"] == "unknown_new_customer"


def test_listing_generator():
    r = client.get("/listing-generator/EM-1002")
    assert r.status_code == 200
    assert "title" in r.json()


def test_assistant_chat():
    r = client.post("/assistant/chat", json={"customer_id": "C1", "message": "running shoes"})
    assert r.status_code == 200
    assert "reply" in r.json()


def test_orchestrated_checkout():
    order = {
        "customer_id": "C99",
        "cart": [{"sku": "EM-1001", "quantity": 2, "unit_price": 24.99}],
        "country": "Pakistan",
    }
    r = client.post("/workflow/checkout", json=order)
    assert r.status_code == 200
    body = r.json()
    assert "invoice_no" in body
    assert body["fraud_check"]["decision"] in ("auto_approve", "step_up_verification", "hold_for_review")


def test_landing_page():
    r = client.get("/workflow/landing/C99")
    assert r.status_code == 200
    assert "homepage_recommendations" in r.json()
