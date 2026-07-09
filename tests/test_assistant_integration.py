"""
Enterprise Integration Tests for Shopping Assistant
===================================================
Tests full horizontal integration paths:
- Search + Assistant (reading from seeded SQLite index)
- Recommendations + Assistant (reading from seeded SQLite profile)
- LLM + Response Validation Pipeline
- Fallback modes & Circuit Breakers
- Gateway Integration Middleware
Mocks ONLY external HTTP API calls to Anthropic.
"""
from __future__ import annotations
import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.shared.seed_data import seed
from app.services.assistant.engine import get_assistant_engine

# Seed database before running integration test suite
seed()

# Configure TestClient with raise_server_exceptions=False so that HTTPExceptions
# raised in Starlette BaseHTTPMiddleware are correctly returned as 401 responses.
client = TestClient(app, raise_server_exceptions=False)


def test_integration_search_and_recommendations_flow():
    """Verify that Assistant uses the real SQLite search & recs engines without mocking."""
    # Reset engine state to healthy
    engine = get_assistant_engine()
    engine.consecutive_failures = 0
    engine.circuit_open = False
    
    r = client.post("/assistant/chat", json={
        "customer_id": "C1",
        "message": "wireless earbuds"
    })
    
    assert r.status_code == 200
    body = r.json()
    assert "reply" in body
    assert "suggested_products" in body
    assert len(body["suggested_products"]) > 0
    
    # Ensure real product data from SQLite seed is populated
    top_product = body["suggested_products"][0]
    assert "EM-1001" in [p["sku"] for p in body["suggested_products"]]
    assert top_product["name"] is not None
    assert top_product["price"] > 0.0


def test_integration_llm_mode_validation_success():
    """Verify LLM success mode and output validations using mock external API."""
    engine = get_assistant_engine()
    engine.consecutive_failures = 0
    engine.circuit_open = False

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    # Include real product EM-1001 that search will find for "wireless earbuds"
    mock_resp.json.return_value = {
        "content": [{
            "text": "I suggest looking at the Wireless Earbuds (EM-1001) for great audio quality."
        }]
    }

    # Patch ONLY the external HTTP post call to Anthropic and API key global check
    with patch("app.services.assistant.engine.ANTHROPIC_API_KEY", "fake-key"), \
         patch.object(engine.http_client, "post", AsyncMock(return_value=mock_resp)):
        
        r = client.post("/assistant/chat", json={
            "customer_id": "C1",
            "message": "wireless earbuds"
        })
        
        assert r.status_code == 200
        body = r.json()
        assert "Wireless Earbuds" in body["reply"]
        assert "EM-1001" in body["reply"]
        assert body["used_services"][-1] == "llm"


def test_integration_llm_validation_reject_fallback():
    """Verify that LLM replies recommending non-existent SKUs are caught and fallback to Template."""
    engine = get_assistant_engine()
    engine.consecutive_failures = 0
    engine.circuit_open = False

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    # Mentioning an unauthorized SKU (FAKE-SKU-999) not in catalog index
    mock_resp.json.return_value = {
        "content": [{
            "text": "I suggest looking at this fake device (FAKE-SKU-999)."
        }]
    }

    with patch("app.services.assistant.engine.ANTHROPIC_API_KEY", "fake-key"), \
         patch.object(engine.http_client, "post", AsyncMock(return_value=mock_resp)):
        
        r = client.post("/assistant/chat", json={
            "customer_id": "C1",
            "message": "wireless earbuds"
        })
        
        # Verify the validation rejected it and fell back to Template mode response
        assert r.status_code == 200
        body = r.json()
        assert "FAKE-SKU-999" not in body["reply"]
        assert body["used_services"][-1] == "template"


def test_integration_circuit_breaker_degradation():
    """Verify that multiple external API timeouts trip the circuit breaker and use Template mode."""
    engine = get_assistant_engine()
    engine.consecutive_failures = 0
    engine.circuit_open = False

    # Simulate Anthropic API throwing connect timeouts
    with patch("app.services.assistant.engine.ANTHROPIC_API_KEY", "fake-key"), \
         patch.object(engine.http_client, "post", AsyncMock(side_effect=httpx.ConnectTimeout("Connect failure"))):
        
        # 1st failure
        r = client.post("/assistant/chat", json={"customer_id": "C1", "message": "earbuds"})
        assert r.json()["used_services"][-1] == "template"
        assert engine.consecutive_failures == 1
        
        # 2nd failure
        client.post("/assistant/chat", json={"customer_id": "C1", "message": "earbuds"})
        assert engine.consecutive_failures == 2
        
        # 3rd failure - trips breaker
        client.post("/assistant/chat", json={"customer_id": "C1", "message": "earbuds"})
        assert engine.circuit_open is True
        assert engine.consecutive_failures == 3

        # 4th request - should immediately bypass LLM API call entirely
        with patch.object(engine.http_client, "post", AsyncMock()) as mock_post:
            r = client.post("/assistant/chat", json={"customer_id": "C1", "message": "earbuds"})
            assert r.json()["used_services"][-1] == "template"
            mock_post.assert_not_called()


def test_integration_gateway_authentication():
    """Verify integration gateway key header rules enforcement."""
    # Temporarily set mock gateway key
    with patch("app.main.GATEWAY_API_KEY", "prod-secret-api-key"):
        
        # Call without header -> 401 Unauthorized
        r = client.post("/assistant/chat", json={
            "customer_id": "C1",
            "message": "earbuds"
        })
        assert r.status_code == 401
        assert r.json()["detail"] == "Missing or invalid AI gateway key"
        
        # Call with incorrect key header -> 401 Unauthorized
        r = client.post("/assistant/chat", json={
            "customer_id": "C1",
            "message": "earbuds"
        }, headers={"X-EMart-AI-Key": "wrong-key"})
        assert r.status_code == 401
        
        # Call with valid key header -> 200 OK
        r = client.post("/assistant/chat", json={
            "customer_id": "C1",
            "message": "earbuds"
        }, headers={"X-EMart-AI-Key": "prod-secret-api-key"})
        assert r.status_code == 200
