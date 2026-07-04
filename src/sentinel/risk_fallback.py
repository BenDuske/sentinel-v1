# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Deterministic, LLM-free summary + recommendation fallbacks.

When no LLM is reachable, Sentinel must STILL produce a clean summary and concrete next steps —
the LLM is enrichment, not a hard dependency. These functions reuse the grounded risk taxonomy
(risk.rule_layer) so the offline output is category-aware and defensible, not a stub.
"""
from . import risk

# Category-specific next steps, surfaced when a category's signals are present in the incident.
_CATEGORY_ACTIONS = {
    "injury/medical": [
        "Ensure anyone injured has received appropriate medical care; call emergency services if needed.",
        "Document injuries, witnesses, and the time/location for any insurance or OSHA claim.",
    ],
    "fire/smoke": [
        "Confirm the area is evacuated and the fire is fully out before re-entry.",
        "Notify the fire department / facilities and preserve the scene for cause investigation.",
    ],
    "water/flood": [
        "Shut off the water source and begin extraction/drying to limit secondary damage.",
        "Photograph affected areas and damaged property before cleanup for the claim record.",
    ],
    "electrical/power": [
        "De-energize the affected circuit and keep personnel clear until a qualified electrician inspects it.",
        "Tag out the hazard and schedule a licensed electrical inspection.",
    ],
    "gas/chemical": [
        "Evacuate, ventilate, and avoid ignition sources; call the utility / hazmat line.",
        "Do not re-enter until the leak/spill is confirmed contained and the air is tested safe.",
    ],
    "structural": [
        "Cordon off the affected structure and restrict access until a structural engineer assesses it.",
        "Photograph the damage and obtain an engineering evaluation before reoccupancy.",
    ],
    "security/intrusion": [
        "Secure the premises/accounts and preserve logs and footage as evidence.",
        "Notify law enforcement and your security/IT lead; rotate any compromised credentials.",
    ],
    "theft": [
        "File a police report and compile an itemized list of missing property with values.",
        "Preserve access logs and camera footage; notify your insurer for the claim.",
    ],
    "outage": [
        "Engage on-call / facilities to restore service and identify the root cause.",
        "Record start time, scope, and impact for the post-incident review.",
    ],
    "weather": [
        "Ensure personnel safety and shelter; assess property damage once conditions are safe.",
        "Document storm damage and contact your insurer to begin a claim.",
    ],
}

_GENERIC_ACTIONS = [
    "Verify everyone is safe and the immediate hazard is contained.",
    "Document the incident with photos, times, locations, and witness names.",
    "Notify the responsible owner (facilities, security, or IT) and your insurer if a claim may follow.",
    "Have a human reviewer confirm the severity and approve next steps before acting.",
]


def _matched_categories(text: str) -> list:
    """Return the taxonomy categories whose signals appear in the text, most-severe first.

    Uses the SAME precompiled whole-word matchers as the scoring layer (``risk._MATCHERS``) so the
    offline summary/actions name exactly the categories the audited score's rule layer found. Raw
    substring matching here would resurrect the false positives already fixed in ``risk.rule_layer``
    (e.g. "armed" inside "unarmed", "fire" inside "firearm") and make the offline output disagree
    with the score's own rationale.
    """
    t = (text or "").lower()
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    hits = []
    for category, levels in risk.TAXONOMY.items():
        best = -1
        for sev, kws in levels.items():
            if any(risk._MATCHERS[k].search(t) for k in kws):
                best = max(best, rank[sev])
        if best >= 0:
            hits.append((best, category))
    hits.sort(reverse=True)
    return [c for _, c in hits]


def fallback_summary(incident: dict) -> str:
    """A concise, professional summary built from the incident fields and matched categories."""
    title = (incident.get("title") or "Incident").strip()
    desc = (incident.get("description") or "").strip()
    text = f"{title}. {desc}"
    cats = _matched_categories(text)
    sev = incident.get("severity") or "unscored"

    lead = f"Reported incident: {title}."
    if desc:
        lead += f" {desc.rstrip('.')}."
    cat_line = ""
    if cats:
        cat_line = (" Identified risk factors: " + ", ".join(cats) +
                    f". Grounded severity assessed as {sev}.")
    else:
        cat_line = (f" No specific risk taxonomy signals were detected; "
                    f"severity assessed as {sev} pending review.")
    tail = (" This is a deterministic, AI-assisted summary generated without a language model; "
            "a human should review and edit before the report is used.")
    return (lead + cat_line + tail).strip()


def fallback_actions(incident: dict) -> list:
    """Category-aware recommended next steps; always returns at least 3 concrete items."""
    text = f"{incident.get('title', '')}. {incident.get('description', '')}"
    cats = _matched_categories(text)
    actions = []
    for c in cats:
        for a in _CATEGORY_ACTIONS.get(c, []):
            if a not in actions:
                actions.append(a)
    for a in _GENERIC_ACTIONS:
        if len(actions) >= 5:
            break
        if a not in actions:
            actions.append(a)
    return actions[:5]
