"""Severity reconciliation: rule layer vs LLM -> the HIGHER wins (floor logic).

The LLM is monkeypatched (no network). Verifies both directions plus the offline behavior where
a missing LLM never lowers the rule-layer floor.
"""
from sentinel import risk, ai


def _inc(title, desc=""):
    return {"title": title, "description": desc}


def test_llm_raises_above_rule_floor(monkeypatch):
    # Rule layer would see no signal (low); LLM says critical -> final critical.
    monkeypatch.setattr(ai, "llm_severity", lambda text: "critical")
    sev, rationale = risk.score(_inc("Unusual situation", "something odd happened"))
    assert sev == "critical"
    assert "Rule layer → low" in rationale
    assert "AI judgment → critical" in rationale
    assert "Final = higher of the two → critical" in rationale


def test_rule_floor_holds_when_llm_underestimates(monkeypatch):
    # Grounded rule layer says critical (fire); LLM lowballs to low -> floor wins.
    monkeypatch.setattr(ai, "llm_severity", lambda text: "low")
    sev, rationale = risk.score(_inc("Warehouse fire", "building ablaze, two injured"))
    assert sev == "critical"
    assert "Rule layer → critical" in rationale
    assert "AI judgment → low" in rationale


def test_offline_llm_does_not_lower_floor(monkeypatch):
    # LLM unreachable -> llm_severity returns "" -> rule floor governs, rationale says so.
    monkeypatch.setattr(ai, "llm_severity", lambda text: "")
    sev, rationale = risk.score(_inc("Burst pipe", "water damage in the server room"))
    assert sev == "high"  # water/flood high floor
    assert "unavailable (offline)" in rationale
    assert "rule layer governs" in rationale.lower()


def test_rationale_shows_both_sides(monkeypatch):
    monkeypatch.setattr(ai, "llm_severity", lambda text: "medium")
    sev, rationale = risk.score(_inc("Theft", "equipment stolen overnight"))
    assert "Rule layer →" in rationale and "AI judgment →" in rationale
