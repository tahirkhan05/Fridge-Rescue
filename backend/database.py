"""
Module: backend/database.py
Purpose: SQLite Database management for Fridge Rescue scans, inventory, and history
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "fridge_rescue.db"

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: Fridge Scans
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT,
        annotated_image_path TEXT,
        total_items INTEGER,
        rescue_now_count INTEGER,
        use_soon_count INTEGER,
        stable_count INTEGER,
        raw_inventory_json TEXT,
        rescue_plan_json TEXT
    )
    """)

    # Table 2: Fridge Items (Current personal fridge inventory)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 1,
        detection_confidence REAL,
        freshness TEXT,
        freshness_confidence REAL,
        expiry_date TEXT,
        days_remaining INTEGER,
        perishability TEXT,
        priority TEXT,
        tier TEXT,
        score INTEGER,
        user_adjusted INTEGER DEFAULT 0,
        rescued INTEGER DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table 3: Rescue History (Logged rescue actions)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rescue_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        recipe_title TEXT,
        items_rescued_json TEXT,
        action_note TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_scan(
    image_path: str,
    annotated_image_path: str,
    ranked_inventory: Dict[str, Any],
    rescue_plan: Dict[str, Any]
) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    items = ranked_inventory.get("items", [])
    rescue_now = len([it for it in items if it.get("tier") == "RESCUE NOW"])
    use_soon = len([it for it in items if it.get("tier") == "USE SOON"])
    stable = len([it for it in items if it.get("tier") == "STABLE"])

    cursor.execute("""
    INSERT INTO scans (
        image_path, annotated_image_path, total_items, 
        rescue_now_count, use_soon_count, stable_count,
        raw_inventory_json, rescue_plan_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_path,
        annotated_image_path,
        len(items),
        rescue_now,
        use_soon,
        stable,
        json.dumps(ranked_inventory),
        json.dumps(rescue_plan)
    ))

    scan_id = cursor.lastrowid

    # Update or insert current active inventory
    for it in items:
        cursor.execute("""
        INSERT OR REPLACE INTO inventory (
            id, name, quantity, detection_confidence, freshness,
            freshness_confidence, expiry_date, days_remaining,
            perishability, priority, tier, score, user_adjusted, rescued, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
        """, (
            it.get("id"),
            it.get("name"),
            it.get("quantity", 1),
            it.get("detection_confidence", 0.0),
            it.get("freshness"),
            it.get("freshness_confidence"),
            it.get("expiry_date"),
            it.get("days_remaining"),
            it.get("perishability", "moderate"),
            it.get("priority", "LOW"),
            it.get("tier", "STABLE"),
            it.get("score", 0),
            1 if it.get("user_adjusted") else 0
        ))

    conn.commit()
    conn.close()
    return scan_id

def update_inventory_item(item_id: str, updates: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()

    fields = []
    values = []
    for k, v in updates.items():
        if k in ["name", "quantity", "expiry_date", "freshness", "rescued"]:
            fields.append(f"{k} = ?")
            values.append(v)

    if fields:
        fields.append("user_adjusted = 1")
        fields.append("last_updated = CURRENT_TIMESTAMP")
        values.append(item_id)
        query = f"UPDATE inventory SET {', '.join(fields)} WHERE id = ?"
        cursor.execute(query, tuple(values))
        conn.commit()

    conn.close()

def get_recent_scans(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, scan_time, image_path, total_items, rescue_now_count, use_soon_count, stable_count
    FROM scans ORDER BY scan_time DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_current_inventory() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM inventory WHERE rescued = 0 ORDER BY score DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize database tables on import
init_db()
