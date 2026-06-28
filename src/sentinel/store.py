# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""SQLite incident store (local, no external DB)."""
import json
import sqlite3
import time
from . import config


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS incidents(
        id TEXT PRIMARY KEY, data TEXT NOT NULL,
        severity TEXT, status TEXT, created_at REAL, updated_at REAL)""")
    return c


def save(incident: dict) -> dict:
    incident["updated_at"] = time.time()
    with _conn() as c:
        c.execute(
            "REPLACE INTO incidents(id,data,severity,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (incident["id"], json.dumps(incident), incident.get("severity"),
             incident.get("status"), incident.get("created_at"), incident["updated_at"]),
        )
    return incident


def get(incident_id: str):
    with _conn() as c:
        row = c.execute("SELECT data FROM incidents WHERE id=?", (incident_id,)).fetchone()
    return json.loads(row[0]) if row else None


def list_all():
    with _conn() as c:
        rows = c.execute("SELECT data FROM incidents ORDER BY created_at DESC").fetchall()
    return [json.loads(r[0]) for r in rows]
