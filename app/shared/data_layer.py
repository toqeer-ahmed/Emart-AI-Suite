"""
Shared Data Layer
==================
Every AI service reads from and writes signals back to ONE data layer,
per the "no duplicate pipelines" design principle in the proposal.

For this add-on package, the default implementation is SQLite (zero-setup,
works out of the box, good enough to demo the whole pipeline end to end).

TO CONNECT TO E-MART'S REAL DATABASES:
---------------------------------------
E-Mart runs a Firebase/MySQL hybrid. You do NOT need to migrate anything.
Implement `DataLayer` (below) once against your real stores, e.g.:

    class EmartDataLayer(DataLayer):
        def __init__(self, mysql_conn, firestore_client):
            self.mysql = mysql_conn
            self.fs = firestore_client

        def get_product(self, sku):
            # read from MySQL `products` table
            ...

        def log_event(self, event_type, payload):
            # write to Firestore `ai_signals` collection, or a MySQL
            # `ai_events` table - whichever you already use for
            # analytics/logging
            ...

Then in app/config.py, point `get_data_layer()` at `EmartDataLayer` instead
of `SQLiteDataLayer`. Every service in this package only ever calls the
`DataLayer` interface, never a specific database - so this is a one-file
change to go from "demo mode" to "wired into E-Mart production data".
"""
from __future__ import annotations
import sqlite3
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.environ.get("EMART_AI_DB_PATH", os.path.join(os.path.dirname(__file__), "emart_ai.db"))


class DataLayer(ABC):
    """Abstract contract every service programs against."""

    @abstractmethod
    def get_product(self, sku: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_products(self, category: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def upsert_product(self, product: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get_customer_events(self, customer_id: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def add_order_event(self, event: Dict[str, Any]) -> None: ...

    @abstractmethod
    def add_review(self, sku: str, text: str, rating: int) -> None: ...

    @abstractmethod
    def get_reviews(self, sku: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def log_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        """Continuous-learning hook: every AI decision writes a signal back
        here so it can retrain models later (Section 7 of the proposal)."""
        ...

    @abstractmethod
    def get_signals(self, signal_type: str, limit: int = 1000) -> List[Dict[str, Any]]: ...


class SQLiteDataLayer(DataLayer):
    """Default, zero-config implementation. Good for local dev, demos, and
    as a reference for what the real E-Mart adapter needs to implement."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY, name TEXT, description TEXT, category TEXT,
                price REAL, stock INTEGER, vendor_id TEXT, tags TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_no TEXT, customer_id TEXT,
                sku TEXT, description TEXT, quantity INTEGER, unit_price REAL,
                invoice_date TEXT, country TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT, text TEXT,
                rating INTEGER, created_at TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, signal_type TEXT,
                payload TEXT, created_at TEXT
            )""")

    def get_product(self, sku: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
            return dict(row) if row else None

    def list_products(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as c:
            if category:
                rows = c.execute("SELECT * FROM products WHERE category = ?", (category,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM products").fetchall()
            return [dict(r) for r in rows]

    def upsert_product(self, product: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("""INSERT INTO products (sku, name, description, category, price, stock, vendor_id, tags)
                         VALUES (:sku, :name, :description, :category, :price, :stock, :vendor_id, :tags)
                         ON CONFLICT(sku) DO UPDATE SET
                            name=excluded.name, description=excluded.description,
                            category=excluded.category, price=excluded.price,
                            stock=excluded.stock, vendor_id=excluded.vendor_id, tags=excluded.tags""",
                      {**product, "tags": json.dumps(product.get("tags", []))})

    def get_customer_events(self, customer_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM order_events WHERE customer_id = ?", (customer_id,)).fetchall()
            return [dict(r) for r in rows]

    def add_order_event(self, event: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("""INSERT INTO order_events
                (invoice_no, customer_id, sku, description, quantity, unit_price, invoice_date, country)
                VALUES (:invoice_no, :customer_id, :sku, :description, :quantity, :unit_price, :invoice_date, :country)""",
                event)

    def add_review(self, sku: str, text: str, rating: int) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO reviews (sku, text, rating, created_at) VALUES (?, ?, ?, ?)",
                      (sku, text, rating, datetime.utcnow().isoformat()))

    def get_reviews(self, sku: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM reviews WHERE sku = ?", (sku,)).fetchall()
            return [dict(r) for r in rows]

    def log_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO ai_signals (signal_type, payload, created_at) VALUES (?, ?, ?)",
                      (signal_type, json.dumps(payload, default=str), datetime.utcnow().isoformat()))

    def get_signals(self, signal_type: str, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM ai_signals WHERE signal_type = ? ORDER BY id DESC LIMIT ?",
                              (signal_type, limit)).fetchall()
            return [dict(r) for r in rows]


_data_layer_singleton: Optional[DataLayer] = None


def get_data_layer() -> DataLayer:
    global _data_layer_singleton
    if _data_layer_singleton is None:
        _data_layer_singleton = SQLiteDataLayer()
    return _data_layer_singleton
