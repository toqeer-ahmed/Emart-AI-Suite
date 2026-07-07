"""Seeds a small demo catalog + review set so every endpoint has something
real to respond with immediately after `docker compose up`. Safe to call
multiple times (upserts). Delete/replace this once E-Mart's real product
feed is wired into the DataLayer."""
from app.shared.data_layer import get_data_layer

DEMO_PRODUCTS = [
    {"sku": "EM-1001", "name": "Wireless Bluetooth Earbuds", "description": "Noise-cancelling wireless earbuds with 30h battery life", "category": "electronics", "price": 24.99, "stock": 120, "vendor_id": "V1", "tags": ["audio", "wireless", "bluetooth"]},
    {"sku": "EM-1002", "name": "Stainless Steel Water Bottle", "description": "Insulated 750ml water bottle keeps drinks cold for 24h", "category": "home", "price": 12.50, "stock": 300, "vendor_id": "V2", "tags": ["bottle", "insulated", "eco"]},
    {"sku": "EM-1003", "name": "Men's Running Shoes", "description": "Lightweight breathable running shoes for daily training", "category": "fashion", "price": 39.99, "stock": 80, "vendor_id": "V3", "tags": ["shoes", "running", "sportswear"]},
    {"sku": "EM-1004", "name": "Kids Educational Tablet", "description": "7-inch tablet with parental controls and learning apps", "category": "electronics", "price": 59.00, "stock": 45, "vendor_id": "V1", "tags": ["tablet", "kids", "education"]},
    {"sku": "EM-1005", "name": "Non-Stick Cookware Set", "description": "5-piece non-stick cookware set, dishwasher safe", "category": "home", "price": 44.99, "stock": 60, "vendor_id": "V2", "tags": ["kitchen", "cookware"]},
    {"sku": "EM-1006", "name": "Women's Denim Jacket", "description": "Classic fit denim jacket, machine washable", "category": "fashion", "price": 29.99, "stock": 95, "vendor_id": "V3", "tags": ["jacket", "denim", "casual"]},
    {"sku": "EM-1007", "name": "Smart LED Desk Lamp", "description": "Adjustable brightness LED lamp with USB charging port", "category": "home", "price": 18.75, "stock": 150, "vendor_id": "V2", "tags": ["lamp", "led", "desk"]},
    {"sku": "EM-1008", "name": "Fitness Resistance Bands Set", "description": "5 resistance levels for home workouts", "category": "sports", "price": 15.99, "stock": 200, "vendor_id": "V3", "tags": ["fitness", "bands", "home-gym"]},
]

DEMO_REVIEWS = [
    ("EM-1001", "Sound quality is amazing and battery lasts all day", 5),
    ("EM-1001", "Great earbuds but delivery took way too long", 4),
    ("EM-1001", "Connection keeps dropping, quite disappointing", 2),
    ("EM-1003", "Super comfortable for daily runs, highly recommend", 5),
    ("EM-1003", "Runs a bit small, had to exchange for a bigger size", 3),
    ("EM-1006", "Love the fit and the material feels premium", 5),
    ("EM-1006", "Color was slightly different from the photos", 3),
    ("EM-1004", "My kid loves it but the delivery box arrived damaged", 4),
]


def seed():
    dl = get_data_layer()
    for p in DEMO_PRODUCTS:
        dl.upsert_product(p)
    for sku, text, rating in DEMO_REVIEWS:
        dl.add_review(sku, text, rating)
    return {"products_seeded": len(DEMO_PRODUCTS), "reviews_seeded": len(DEMO_REVIEWS)}


if __name__ == "__main__":
    print(seed())
