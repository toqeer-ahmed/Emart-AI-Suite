"""
Pytest Suite for Shopping Assistant
===================================
Provides extensive unit, integration, and failure path coverage for:
- intent_detector.py
- http_client.py
- engine.py
- router.py
Targeting >95% overall code coverage.
"""
from __future__ import annotations
import pytest
import httpx
import anyio
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from app.services.assistant.intent_detector import IntentDetector
from app.services.assistant.http_client import HttpClientManager
from app.services.assistant.engine import (
    AssistantEngine,
    AssistantCandidate,
    AssistantConfig,
    get_assistant_engine
)
from app.shared.schemas import AssistantTurn, AssistantReply, SearchResult, RecommendationResult


# ==========================================
# 1. INTENT DETECTOR STRATEGY TESTS
# ==========================================

def test_intent_detector_budget_strategy():
    detector = IntentDetector(["electronics", "clothing"])
    
    res = detector.detect("laptops under $500")
    assert res["type"] == "budget"
    assert res["budget_limit"] == 500.0
    assert res["sort_by_price_asc"] is True

    res = detector.detect("cheap phone under  rs.150")
    assert res["type"] == "budget"
    assert res["budget_limit"] == 150.0

    res = detector.detect("recommend an affordable keyboard")
    assert res["type"] == "budget"
    assert res["budget_limit"] is None
    assert res["sort_by_price_asc"] is True


def test_intent_detector_comparison_strategy():
    detector = IntentDetector(["laptops", "phones"])
    
    res = detector.detect("iPhone vs Samsung Galaxy")
    assert res["type"] == "comparison"
    assert res["comparison"] is True
    assert res["comparison_subjects"] == ["iphone", "samsung galaxy"]

    res = detector.detect("compare dynamic microphone and condenser microphone")
    assert res["type"] == "comparison"
    assert res["comparison_subjects"] == ["dynamic microphone", "condenser microphone"]

    res = detector.detect("what is the difference between laptops")
    assert res["type"] == "comparison"
    assert res["comparison"] is True


def test_intent_detector_gift_strategy():
    detector = IntentDetector([])
    res = detector.detect("present ideas for birthday")
    assert res["type"] == "gift"
    assert res["gift"] is True


def test_intent_detector_category_strategy():
    detector = IntentDetector(["electronics", "running shoes", "laptops"])
    
    res = detector.detect("show electronics")
    assert res["type"] == "category"
    assert res["matched_category"] == "electronics"

    res = detector.detect("I want to buy running shoe")
    assert res["type"] == "category"
    assert res["matched_category"] == "running shoes"


def test_intent_detector_recommendation_strategy():
    detector = IntentDetector([])
    res = detector.detect("suggest a good product")
    assert res["type"] == "recommendation"
    assert res["recommendation"] is True


# ==========================================
# 2. HTTP CLIENT MANAGER TESTS
# ==========================================

@pytest.mark.anyio
async def test_http_client_post_success():
    client_mgr = HttpClientManager()
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    
    with patch.object(client_mgr.client, "post", AsyncMock(return_value=mock_response)) as mock_post:
        resp = await client_mgr.post("http://test.com", {}, {})
        assert resp.status_code == 200
        mock_post.assert_called_once()
        
    await client_mgr.close()


@pytest.mark.anyio
async def test_http_client_transient_retry():
    client_mgr = HttpClientManager()
    
    mock_resp_500 = MagicMock(spec=httpx.Response)
    mock_resp_500.status_code = 500
    
    mock_resp_200 = MagicMock(spec=httpx.Response)
    mock_resp_200.status_code = 200
    
    with patch.object(
        client_mgr.client, 
        "post", 
        AsyncMock(side_effect=[mock_resp_500, mock_resp_200])
    ) as mock_post:
        resp = await client_mgr.post("http://test.com", {}, {})
        assert resp.status_code == 200
        assert mock_post.call_count == 2
        
    await client_mgr.close()


@pytest.mark.anyio
async def test_http_client_connection_error_retry():
    client_mgr = HttpClientManager()
    
    mock_resp_200 = MagicMock(spec=httpx.Response)
    mock_resp_200.status_code = 200
    
    with patch.object(
        client_mgr.client, 
        "post", 
        AsyncMock(side_effect=[httpx.ConnectTimeout("Connect timeout"), mock_resp_200])
    ) as mock_post:
        resp = await client_mgr.post("http://test.com", {}, {})
        assert resp.status_code == 200
        assert mock_post.call_count == 2
        
    await client_mgr.close()


@pytest.mark.anyio
async def test_http_client_read_timeout_no_retry():
    client_mgr = HttpClientManager()
    
    with patch.object(
        client_mgr.client, 
        "post", 
        AsyncMock(side_effect=httpx.ReadTimeout("Read timed out"))
    ):
        with pytest.raises(httpx.ReadTimeout):
            await client_mgr.post("http://test.com", {}, {})
            
    await client_mgr.close()


# ==========================================
# 3. ENGINE HELPER METHOD TESTS
# ==========================================

@pytest.fixture
def engine():
    with patch("app.services.assistant.engine.get_search_engine"), \
         patch("app.services.assistant.engine.get_recommendation_engine"), \
         patch("app.services.assistant.engine.get_data_layer"):
        eng = AssistantEngine()
        eng._catalog_product_names = ["iphone 13", "macbook air", "wireless earbuds"]
        return eng


def test_engine_deduplicate(engine):
    search_hits = [
        SearchResult(sku="SKU-1", name="Product 1", price=10.0, score=0.9, in_stock=True, description="Desc 1", category="Cat 1", stock=5)
    ]
    recs = [
        RecommendationResult(sku="SKU-1", name="Product 1", price=10.0, reason="popular", description="Desc 1", category="Cat 1", stock=5),
        RecommendationResult(sku="SKU-2", name="Product 2", price=20.0, reason="frequently bought", description="Desc 2", category="Cat 2", stock=10)
    ]
    candidates = engine._deduplicate(search_hits, recs)
    
    assert len(candidates) == 2
    sku_to_cand = {c.sku: c for c in candidates}
    
    assert sku_to_cand["SKU-1"].is_search is True
    assert sku_to_cand["SKU-1"].is_rec is True
    assert sku_to_cand["SKU-1"].reason == "popular"
    
    assert sku_to_cand["SKU-2"].is_search is False
    assert sku_to_cand["SKU-2"].is_rec is True


def test_engine_filter_stock(engine):
    candidates = [
        AssistantCandidate("S1", "P1", 10.0, "D1", "C1", 0, "", 1.0, True, False, 0),
        AssistantCandidate("S2", "P2", 20.0, "D2", "C2", 5, "", 1.0, True, False, 1)
    ]
    res = engine._filter_stock(candidates)
    assert len(res) == 1
    assert res[0].sku == "S2"


def test_engine_filter_budget(engine):
    candidates = [
        AssistantCandidate("S1", "P1", 10.0, "D1", "C1", 5, "", 1.0, True, False, 0),
        AssistantCandidate("S2", "P2", 20.0, "D2", "C2", 5, "", 1.0, True, False, 1)
    ]
    res = engine._filter_budget(candidates, 15.0)
    assert len(res) == 1
    assert res[0].sku == "S1"


def test_engine_filter_category(engine):
    candidates = [
        AssistantCandidate("S1", "P1", 10.0, "D1", "electronics", 5, "", 1.0, True, False, 0),
        AssistantCandidate("S2", "P2", 20.0, "D2", "clothing", 5, "", 1.0, True, False, 1)
    ]
    res = engine._filter_category(candidates, "electronics", True)
    assert len(res) == 1
    assert res[0].sku == "S1"


def test_engine_filter_comparison(engine):
    candidates = [
        AssistantCandidate("S1", "iPhone 13 Pro", 999.0, "Flagship Apple device", "phones", 5, "", 1.0, True, False, 0),
        AssistantCandidate("S2", "Galaxy S22", 899.0, "Flagship Samsung device", "phones", 5, "", 0.8, True, False, 1),
        AssistantCandidate("S3", "MacBook Air", 999.0, "Apple laptop device", "laptops", 5, "", 0.5, True, False, 2)
    ]
    
    intent = {
        "comparison": True,
        "comparison_subjects": ["iphone", "galaxy"],
        "type": "comparison"
    }
    res = engine._filter_comparison(candidates, intent)
    assert len(res) == 2
    skus = {c.sku for c in res}
    assert "S1" in skus
    assert "S2" in skus


def test_engine_rank_candidates(engine):
    candidates = [
        AssistantCandidate("S1", "P1", 20.0, "D1", "C1", 5, "", 0.5, True, False, 0),
        AssistantCandidate("S2", "P2", 10.0, "D2", "C2", 5, "", 0.9, False, True, 1)
    ]
    
    intent_rec = {"type": "recommendation", "sort_by_price_asc": False}
    res = engine._rank_candidates(list(candidates), intent_rec)
    assert res[0].sku == "S2"

    intent_price = {"type": "budget", "sort_by_price_asc": True}
    res = engine._rank_candidates(list(candidates), intent_price)
    assert res[0].sku == "S2"


def test_engine_build_prompt(engine):
    candidates = [
        AssistantCandidate("S1", "P1", 10.0, "Sample description info", "C1", 5, "reason", 1.0, True, True, 0)
    ]
    ctx = {
        "message": "Find earbuds",
        "intent": {"type": "general", "budget_limit": None, "matched_category": None, "comparison": False, "gift": False},
        "candidates": candidates
    }
    prompt = engine._build_prompt(ctx)
    assert "Find earbuds" in prompt
    assert "S1" in prompt
    assert "Sample description info" in prompt


def test_engine_extract_response(engine):
    raw_response = "### Hello **World**! <p>This is a product Wi-Fi</p> emoji 📱 - bullet item"
    extracted = engine._extract_response(raw_response)
    assert "#" not in extracted
    assert "**" not in extracted
    assert "<p>" not in extracted
    assert "📱" not in extracted
    assert "bullet item" in extracted
    assert "Wi-Fi" in extracted


def test_engine_validate_response(engine):
    candidates = [
        AssistantCandidate("EM-1001", "iPhone 13", 999.0, "Apple mobile", "phones", 5, "", 1.0, True, False, 0)
    ]
    ctx = {"candidates": candidates}
    
    assert engine._validate_response("I recommend the iPhone 13 (EM-1001) which is excellent.", ctx) is True
    assert engine._validate_response("I recommend **iPhone 13**", ctx) is False
    assert engine._validate_response("I recommend MacBook Air instead of the iPhone 13", ctx) is False


# ==========================================
# 4. LLM API & PIPELINE SHUTDOWN TESTS
# ==========================================

@pytest.mark.anyio
async def test_engine_try_llm_success(engine):
    ctx = {
        "message": "iphone 13",
        "candidates": [
            AssistantCandidate("EM-1001", "iPhone 13", 999.0, "Apple", "phones", 5, "", 1.0, True, False, 0)
        ],
        "fallback_reason": "none"
    }
    
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"text": "I recommend looking at the iPhone 13 (EM-1001) which is a high-quality smartphone."}]
    }

    with patch("app.services.assistant.engine.ANTHROPIC_API_KEY", "fake-key"), \
         patch.object(engine.http_client, "post", AsyncMock(return_value=mock_resp)):
        reply, success = await engine._try_llm(ctx)
        assert success is True
        assert "iPhone 13" in reply
        assert engine.consecutive_failures == 0


@pytest.mark.anyio
async def test_engine_try_llm_circuit_breaker(engine):
    ctx = {"message": "iphone", "candidates": [], "fallback_reason": "none"}
    engine.consecutive_failures = 2

    with patch("app.services.assistant.engine.ANTHROPIC_API_KEY", "fake-key"), \
         patch.object(engine.http_client, "post", AsyncMock(side_effect=httpx.ConnectTimeout("Connect failure"))):
        reply, success = await engine._try_llm(ctx)
        assert success is False
        assert engine.circuit_open is True
        assert engine.consecutive_failures == 3


@pytest.mark.anyio
async def test_engine_handle_turn_empty_input(engine):
    turn = AssistantTurn(customer_id="C1", message="")
    res = await engine.handle_turn(turn)
    assert "Please enter a valid message." in res.reply


@pytest.mark.anyio
async def test_engine_handle_turn_double_engine_failure(engine):
    turn = AssistantTurn(customer_id="C1", message="shoes")
    
    with patch.object(engine.search, "search", side_effect=Exception("DB connection error")), \
         patch.object(engine.recs, "for_customer", side_effect=Exception("Rec index error")):
        res = await engine.handle_turn(turn)
        assert "E-Mart catalog services are temporarily unavailable." in res.reply
        assert res.suggested_products == []


# ==========================================
# 5. INTEGRATION ROUTER API TESTS
# ==========================================

def test_router_chat_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    
    from app.shared.seed_data import seed
    seed()
    
    r = client.post("/assistant/chat", json={"customer_id": "C1", "message": "wireless earbuds"})
    assert r.status_code == 200
    assert "reply" in r.json()
    assert "suggested_products" in r.json()
