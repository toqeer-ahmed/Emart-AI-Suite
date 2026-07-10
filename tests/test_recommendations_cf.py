import pytest
from app.shared.data_layer import get_data_layer
from app.services.recommendations.engine import RecommendationEngine

@pytest.fixture(autouse=True)
def clean_database():
    """Ensure database state is clean of our test-specific entries before each test."""
    dl = get_data_layer()
    test_skus = ["SKU-1", "SKU-2", "SKU-3", "SKU-4"]
    test_users = ["USER-A", "USER-B", "USER-C", "USER-NEW"]
    with dl._conn() as c:
        for sku in test_skus:
            c.execute("DELETE FROM order_events WHERE sku = ?", (sku,))
            c.execute("DELETE FROM ai_signals WHERE payload LIKE ?", (f"%{sku}%",))
            c.execute("DELETE FROM products WHERE sku = ?", (sku,))
        for user in test_users:
            c.execute("DELETE FROM order_events WHERE customer_id = ?", (user,))
        
        # Seed basic test products
        c.execute("INSERT INTO products (sku, name, description, category, price, stock) VALUES (?, ?, ?, ?, ?, ?)",
                  ("SKU-1", "Product One", "Description 1", "Electronics", 10.0, 100))
        c.execute("INSERT INTO products (sku, name, description, category, price, stock) VALUES (?, ?, ?, ?, ?, ?)",
                  ("SKU-2", "Product Two", "Description 2", "Electronics", 20.0, 80))
        c.execute("INSERT INTO products (sku, name, description, category, price, stock) VALUES (?, ?, ?, ?, ?, ?)",
                  ("SKU-3", "Product Three", "Description 3", "Apparel", 30.0, 50))
        c.execute("INSERT INTO products (sku, name, description, category, price, stock) VALUES (?, ?, ?, ?, ?, ?)",
                  ("SKU-4", "Product Four", "Description 4", "Apparel", 40.0, 30))
    yield
    # Clean up after test
    with dl._conn() as c:
        for sku in test_skus:
            c.execute("DELETE FROM order_events WHERE sku = ?", (sku,))
            c.execute("DELETE FROM ai_signals WHERE payload LIKE ?", (f"%{sku}%",))
            c.execute("DELETE FROM products WHERE sku = ?", (sku,))
        for user in test_users:
            c.execute("DELETE FROM order_events WHERE customer_id = ?", (user,))

def test_cold_start_popularity():
    """Verifies that the engine defaults to stock-based popularity when there is no data."""
    engine = RecommendationEngine()
    recs = engine.for_customer("USER-NEW", top_k=2)
    assert len(recs) == 2
    assert all(r.reason == "popular" for r in recs)

def test_collaborative_filtering_factorization():
    """Verifies that the model fits successfully when there is sufficient order history."""
    dl = get_data_layer()
    
    # Seed order events across multiple users/items to build a matrix
    # User A buys SKU-1 and SKU-3
    dl.add_order_event({"invoice_no": "INV1", "customer_id": "USER-A", "sku": "SKU-1", "description": "", "quantity": 1, "unit_price": 10.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV1", "customer_id": "USER-A", "sku": "SKU-3", "description": "", "quantity": 2, "unit_price": 30.0, "invoice_date": "2026-07-10", "country": "US"})
    
    # User B buys SKU-2 and SKU-4
    dl.add_order_event({"invoice_no": "INV2", "customer_id": "USER-B", "sku": "SKU-2", "description": "", "quantity": 3, "unit_price": 20.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV2", "customer_id": "USER-B", "sku": "SKU-4", "description": "", "quantity": 1, "unit_price": 40.0, "invoice_date": "2026-07-10", "country": "US"})

    # User C buys SKU-1 and SKU-4
    dl.add_order_event({"invoice_no": "INV3", "customer_id": "USER-C", "sku": "SKU-1", "description": "", "quantity": 2, "unit_price": 10.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV3", "customer_id": "USER-C", "sku": "SKU-4", "description": "", "quantity": 2, "unit_price": 40.0, "invoice_date": "2026-07-10", "country": "US"})

    engine = RecommendationEngine()
    success = engine._fit_collaborative_filtering()
    
    assert success is True
    assert "USER-A" in engine.user_factors
    assert "USER-B" in engine.user_factors
    assert "SKU-1" in engine.item_factors
    
    expected_k = min(len(engine.user_ids), len(engine.item_skus), 4)
    assert len(engine.user_factors["USER-A"]) == expected_k

    # Test personalized recommendations (excluding bought skus)
    # USER-A bought SKU-1 and SKU-3, so they should be recommended either SKU-2 or SKU-4
    recs = engine.for_customer("USER-A", top_k=1)
    assert len(recs) == 1
    assert recs[0].sku in ["SKU-2", "SKU-4"]
    assert recs[0].reason == "personalized recommendation based on your purchase history"
    assert recs[0].description == "Description 2" or recs[0].description == "Description 4"

def test_cart_cross_sell_embeddings():
    """Verifies that cross-sell recommendations utilize cosine similarity on item factors."""
    dl = get_data_layer()
    
    # Seed transactions to build embeddings
    dl.add_order_event({"invoice_no": "INV1", "customer_id": "USER-A", "sku": "SKU-1", "description": "", "quantity": 2, "unit_price": 10.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV1", "customer_id": "USER-A", "sku": "SKU-2", "description": "", "quantity": 2, "unit_price": 20.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV2", "customer_id": "USER-B", "sku": "SKU-3", "description": "", "quantity": 3, "unit_price": 30.0, "invoice_date": "2026-07-10", "country": "US"})
    dl.add_order_event({"invoice_no": "INV2", "customer_id": "USER-B", "sku": "SKU-4", "description": "", "quantity": 3, "unit_price": 40.0, "invoice_date": "2026-07-10", "country": "US"})

    engine = RecommendationEngine()
    success = engine._fit_collaborative_filtering()
    assert success is True

    # Get cross-sell for SKU-1
    cross_sell = engine.cart_cross_sell(["SKU-1"], top_k=1)
    assert len(cross_sell) == 1
    # SKU-1 co-occurred with SKU-2, so its factor should align closely with SKU-2
    assert cross_sell[0].sku == "SKU-2"
    assert cross_sell[0].reason == "frequently bought together with items in your cart"
