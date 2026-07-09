"""
Enterprise HttpClientManager Module
===================================
Provides asynchronous connection pooling, persistent limits, structured timeout
management, and resilient retry logic for outgoing HTTP integration calls.
"""
from __future__ import annotations
import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger("app.services.assistant.http_client")


class HttpClientManager:
    def __init__(self, connect_timeout: float = 5.0, read_timeout: float = 20.0):
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=50, keepalive_expiry=30.0)
        timeout = httpx.Timeout(20.0, connect=connect_timeout, read=read_timeout)
        self.client = httpx.AsyncClient(limits=limits, timeout=timeout)
        logger.info("Enterprise HttpClientManager initialized with persistent AsyncClient.")

    async def close(self):
        """Closes connection pools gracefully on system shutdown."""
        await self.client.aclose()
        logger.info("HttpClientManager connection pools closed successfully.")

    async def post(self, url: str, headers: Dict[str, str], json_data: Dict[str, Any]) -> httpx.Response:
        """Sends an asynchronous POST request with resilient retries for transient status codes."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self.client.post(url, headers=headers, json=json_data)
                
                if resp.status_code in [429, 500, 502, 503, 504] and attempt < max_attempts:
                    logger.warning("Transient HTTP %d encountered. Attempt %d/%d, retrying...", resp.status_code, attempt, max_attempts)
                    continue
                    
                return resp
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                if attempt == max_attempts:
                    raise e
                logger.warning("Connection failure. Attempt %d/%d, retrying...", attempt, max_attempts)
            except httpx.ReadTimeout as e:
                raise e
        raise httpx.HTTPStatusError("Max attempts exceeded for POST request.", request=None, response=None)
