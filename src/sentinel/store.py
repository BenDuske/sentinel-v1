# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""SQLite incident store (local, no external DB)."""
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from . import config

_log = logging.getLogger("uvicorn.error")


def _loads(row):
    """Parse a stored ``data`` blob, returning None (with a logged warning) on corruption.

    The ``data`` column holds free JSON text with no schema/validity constraint, so a row can
    carry an unparseable blob — a write truncated by a crash or full disk, or a hand-edit /
    partial migration / foreign writer (the SAME non-schema corruption path report.to_markdown
    and the /analyze endpoint already defend against). A bare ``json.loads`` over such a row 500s
    the endpoint; worse, in ``list_all()`` ONE bad row raises and blanks the ENTIRE incident
    history. Returning None keeps every good incident readable and leaves an operator breadcrumb
    instead of taking the dashboard down.
    """
    try:
        return json.loads(row[0])
    except (ValueError, TypeError) as e:
        _log.warning("sentinel.store: skipping unreadable incident row (corrupt JSON): %s", e)
        return None


@contextmanager
def _conn():
    """Yield a connection inside a transaction, then ALWAYS close it.

    A sqlite3 Connection used as a context manager (``with conn:``) commits/rolls back the
    transaction but does NOT close the connection — so a bare ``with sqlite3.connect(...)`` leaks
    the connection until GC reaps it (surfaces as a ResourceWarning). Wrapping it here guarantees
    close() runs while preserving commit-on-success / rollback-on-error via the inner ``with c``.
    """
    c = sqlite3.connect(config.DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS incidents(
        id TEXT PRIMARY KEY, data TEXT NOT NULL,
        severity TEXT, status TEXT, created_at REAL, updated_at REAL)""")
    try:
        with c:
            yield c
    finally:
        c.close()


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
    # A corrupt single row degrades to None (→ the app's clean 404) rather than 500ing the
    # get/analyze/patch/report endpoints; a genuinely absent id is None the same way.
    return _loads(row) if row else None


def list_all():
    with _conn() as c:
        rows = c.execute("SELECT data FROM incidents ORDER BY created_at DESC").fetchall()
    # Skip any unreadable row so a single corrupt blob can't blank the whole dashboard listing.
    return [inc for inc in (_loads(r) for r in rows) if inc is not None]
