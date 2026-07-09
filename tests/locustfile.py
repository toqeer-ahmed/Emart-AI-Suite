"""
Locust Performance Load Testing Script
======================================
Simulates concurrent user behavior hitting the Shopping Assistant endpoints.
Measures latency, throughput, failures, and timeout behaviors under load.

Execute locally using:
    locust -f tests/locustfile.py --host http://localhost:8000
"""
import os
import random
from locust import HttpUser, task, between  # type: ignore

GATEWAY_KEY = os.environ.get("EMART_AI_GATEWAY_KEY", "")


class ShoppingAssistantPerformanceUser(HttpUser):
    wait_time = between(1.0, 3.0)

    def on_start(self):
        self.headers = {}
        if GATEWAY_KEY:
            self.headers["X-EMart-AI-Key"] = GATEWAY_KEY

    @task(4)
    def post_general_query(self):
        """Simulates standard product queries."""
        payload = {
            "customer_id": f"C_{random.randint(1, 1000)}",
            "message": random.choice([
                "wireless earbuds",
                "running shoes",
                "mechanical keyboard",
                "portable charger"
            ])
        }
        self.client.post("/assistant/chat", json=payload, headers=self.headers, name="/assistant/chat [General]")

    @task(3)
    def post_budget_query(self):
        """Simulates budget intent queries (triggers price sorting and limits)."""
        payload = {
            "customer_id": f"C_{random.randint(1, 1000)}",
            "message": random.choice([
                "cheap earbuds under $50",
                "laptops under $600",
                "affordable keyboards below $100"
            ])
        }
        self.client.post("/assistant/chat", json=payload, headers=self.headers, name="/assistant/chat [Budget]")

    @task(2)
    def post_comparison_query(self):
        """Simulates product comparison intents."""
        payload = {
            "customer_id": f"C_{random.randint(1, 1000)}",
            "message": random.choice([
                "EM-1001 vs EM-1002",
                "compare wireless earbuds and keyboard",
                "difference between laptop and phone"
            ])
        }
        self.client.post("/assistant/chat", json=payload, headers=self.headers, name="/assistant/chat [Comparison]")

    @task(1)
    def post_gift_query(self):
        """Simulates gift intent queries."""
        payload = {
            "customer_id": f"C_{random.randint(1, 1000)}",
            "message": "looking for a nice birthday gift"
        }
        self.client.post("/assistant/chat", json=payload, headers=self.headers, name="/assistant/chat [Gift]")

    @task(1)
    def post_unauthorized_gateway_query(self):
        """Tests gateway verification rules bypass under load (if key is set)."""
        headers = {"X-EMart-AI-Key": "malicious-bypass-key"}
        payload = {
            "customer_id": "C_BAD",
            "message": "bypass gateway"
        }
        with self.client.post("/assistant/chat", json=payload, headers=headers, catch_response=True, name="/assistant/chat [Gateway Bypass]") as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
