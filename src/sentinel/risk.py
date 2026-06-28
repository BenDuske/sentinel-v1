"""Grounded risk scoring — deterministic rule layer reconciled with the LLM.

Severity is NOT just "the AI said so." A transparent rule layer sets a defensible floor from
keywords/signals; the LLM adds judgment; the final score is the HIGHER of the two, with a
rationale that shows BOTH. That auditability is what insurers and technical judges want.
"""
from . import ai

_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_INV = {v: k for k, v in _RANK.items()}

# keyword -> severity floor it implies
_RULES = {
    "critical": ["fire", "flood", "injury", "injured", "gas leak", "breach", "ransomware",
                 "intrusion", "weapon", "collapse", "explosion", "casualty", "data breach"],
    "high":     ["smoke", "water damage", "theft", "stolen", "unauthorized", "malware",
                 "outage", "down", "broken into", "assault", "accident", "leak"],
    "medium":   ["alarm", "suspicious", "minor", "slip", "trip", "error", "failed login",
                 "dent", "scratch", "vandalism"],
    "low":      ["note", "observation", "routine", "test", "info", "reminder"],
}


def rule_layer(text: str):
    t = (text or "").lower()
    best, reasons = -1, []
    for sev, kws in _RULES.items():
        hits = [k for k in kws if k in t]
        if hits and _RANK[sev] > best:
            best = _RANK[sev]
        if hits:
            reasons.append(f"{sev}: matched {', '.join(hits)}")
    if best < 0:
        return "low", ["no strong risk signals matched (default floor: low)"]
    return _INV[best], reasons


def score(incident: dict):
    """Return (severity, rationale). Combines rule floor + LLM, takes the higher."""
    text = f"{incident.get('title','')}. {incident.get('description','')}"
    rule_sev, rule_reasons = rule_layer(text)
    llm_sev = ai.llm_severity(text) or rule_sev
    final = _INV[max(_RANK[rule_sev], _RANK.get(llm_sev, 0))]
    rationale = (f"Rule layer → {rule_sev} ({'; '.join(rule_reasons)}). "
                 f"AI judgment → {llm_sev}. "
                 f"Final = higher of the two → {final}.")
    return final, rationale
