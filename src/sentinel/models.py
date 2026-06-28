# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Incident data model."""
import time
import uuid

SEVERITIES = ["low", "medium", "high", "critical"]
STATUSES = ["open", "reviewing", "resolved"]


def new_incident(title: str, description: str = "") -> dict:
    now = time.time()
    return {
        "id": uuid.uuid4().hex[:12],
        "title": title.strip(),
        "description": (description or "").strip(),
        "severity": "unscored",
        "severity_rationale": "",      # transparency (HCAI): why this score
        "summary": "",
        "recommended_actions": [],
        "status": "open",
        "evidence": [],               # uploaded filenames
        "ai_generated": False,        # flips true after analyze; human can then edit
        "created_at": now,
        "updated_at": now,
    }
