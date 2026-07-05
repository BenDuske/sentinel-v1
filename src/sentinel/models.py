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
        # Coerce None -> "" symmetrically with description below: the author already guards a None
        # description ((description or "")), but title had a bare title.strip() that raised
        # AttributeError on None. The HTTP API validates title as a required str, but new_incident is
        # a public data-model helper (seed scripts, CLI, bulk/foreign imports call it directly), so a
        # missing title cell would crash the model constructor instead of degrading like description.
        "title": (title or "").strip(),
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
