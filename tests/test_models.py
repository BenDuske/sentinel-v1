# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Direct tests for the incident data model (models.new_incident).

new_incident is the input boundary for every incident dict — it was exercised indirectly by the
flow/store tests but had no direct coverage of its defensive coercion. In particular the None
handling was ASYMMETRIC: description was guarded ((description or "")) but title was a bare
title.strip() that raised AttributeError on a None title. These pin the symmetric coercion plus
the invariants the rest of the app relies on (defaults, stripping, unique ids).
"""
from sentinel.models import new_incident, SEVERITIES, STATUSES


def test_none_title_coerces_not_crashes():
    # Regression: a None title used to raise AttributeError ('NoneType' has no attribute 'strip'),
    # while a None description already degraded to "". Now both coerce to "".
    inc = new_incident(None)
    assert inc["title"] == ""
    assert inc["description"] == ""


def test_none_description_coerces():
    inc = new_incident("Broken pipe", None)
    assert inc["title"] == "Broken pipe"
    assert inc["description"] == ""


def test_title_and_description_are_stripped():
    inc = new_incident("  Fire in server room  ", "  smoke reported  ")
    assert inc["title"] == "Fire in server room"
    assert inc["description"] == "smoke reported"


def test_whitespace_only_title_strips_to_empty():
    # Reachable via the normal API (a whitespace-only title strips to ""); the report export
    # already degrades an empty title to "(untitled)".
    assert new_incident("   ")["title"] == ""


def test_defaults_are_sane():
    inc = new_incident("Incident")
    assert inc["severity"] == "unscored"
    assert inc["status"] == "open" and inc["status"] in STATUSES
    assert inc["summary"] == "" and inc["severity_rationale"] == ""
    assert inc["recommended_actions"] == [] and inc["evidence"] == []
    assert inc["ai_generated"] is False
    assert isinstance(inc["created_at"], float)
    assert inc["created_at"] == inc["updated_at"]
    assert "unscored" not in SEVERITIES  # sanity: seed severity is deliberately outside the enum


def test_ids_are_unique_and_short():
    ids = {new_incident("x")["id"] for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 12 for i in ids)
