"""Grounded rule-layer taxonomy — each category maps to its expected severity FLOOR.

Pure, deterministic, no LLM/network. This is the differentiator: a defensible, auditable floor.
"""
from sentinel import risk

_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _atleast(actual, floor):
    return _RANK[actual] >= _RANK[floor]


def test_no_signal_defaults_to_low():
    sev, reasons = risk.rule_layer("Routine note: replaced the lobby light bulb.")
    assert sev == "low"
    assert reasons and "no risk taxonomy signals" in reasons[0].lower()


# (text, minimum-expected-severity, category-substring-expected-in-rationale)
CASES = [
    ("Two workers injured, one hospitalized after a fall", "high", "injury/medical"),
    ("Fatality on site after equipment failure", "critical", "injury/medical"),
    # Lay synonyms for the acute-emergency critical floor: "heart attack" (== "cardiac arrest")
    # and anaphylaxis previously scored LOW while "cardiac arrest" scored CRITICAL.
    ("Employee is having a heart attack at their desk", "critical", "injury/medical"),
    ("Anaphylaxis after a bee sting; epinephrine administered", "critical", "injury/medical"),
    ("Structure fire in the warehouse, building ablaze", "critical", "fire/smoke"),
    # "flames" is the lay word for an active fire and must reach the same critical floor as "fire"
    # — "in flames" / "visible flames" (no literal "fire" token) previously dropped to LOW.
    ("The building is in flames on the east side", "critical", "fire/smoke"),
    ("Visible flames on the roof of the annex", "critical", "fire/smoke"),
    ("Smoke detected near the electrical panel", "high", "fire/smoke"),
    ("Server room flooded, equipment submerged", "critical", "water/flood"),
    ("Burst pipe caused water damage to the ceiling", "high", "water/flood"),
    ("Exposed wiring sparking in the breaker box", "high", "electrical/power"),
    ("Gas leak reported; carbon monoxide alarm triggered", "critical", "gas/chemical"),
    # A gas ODOR is a HIGH floor regardless of word order — the noun-compound and "natural gas"
    # phrasings a person actually writes must reach the same floor as "smell of gas"/"odor of gas".
    ("Strong gas smell reported in the mechanical room", "high", "gas/chemical"),
    ("Tenant reports a gas odor near the boiler", "high", "gas/chemical"),
    ("Natural gas smell throughout the east wing", "high", "gas/chemical"),
    ("Smell of natural gas by the loading dock", "high", "gas/chemical"),
    # Verb-order gas-odor reports must reach the same HIGH floor as the noun-order forms above —
    # "I smell gas" / "smells like gas" is how a person actually reports it, and previously dropped
    # to LOW purely on word order.
    ("Tenant called: I smell gas in the second-floor hallway", "high", "gas/chemical"),
    ("Staff report they smell natural gas near the meter", "high", "gas/chemical"),
    ("It smells like gas in the mechanical room", "high", "gas/chemical"),
    ("Partial roof collapse; structural failure observed", "critical", "structural"),
    ("Crack in the load-bearing wall is widening", "high", "structural"),
    ("Break-in overnight; forced entry through side door", "high", "security/intrusion"),
    ("Active shooter reported, armed individual on site", "critical", "security/intrusion"),
    ("Shots fired in the lobby; shooter fled the scene", "critical", "security/intrusion"),
    ("Gunshots heard in the parking garage", "critical", "security/intrusion"),
    ("Reports of gunfire near the loading dock", "critical", "security/intrusion"),
    ("A shooting occurred at the north entrance", "critical", "security/intrusion"),
    ("Theft of equipment; inventory stolen from the dock", "high", "theft"),
    ("Site-wide outage; all systems down", "critical", "outage"),
    ("Power outage; the server is down", "high", "outage"),
    ("Tornado warning; high winds and a fallen tree", "critical", "weather"),
]


def test_each_category_hits_expected_floor():
    for text, floor, category in CASES:
        sev, reasons = risk.rule_layer(text)
        assert _atleast(sev, floor), f"{text!r} -> {sev}, expected >= {floor}"
        joined = " ".join(reasons)
        assert category in joined, f"{text!r} missing category {category} in {joined!r}"


def test_rationale_lists_matched_terms():
    sev, reasons = risk.rule_layer("Gas leak with toxic fumes near the boiler")
    assert sev == "critical"
    assert any("matched:" in r for r in reasons)


# Benign phrases that embed a taxonomy keyword inside a larger word. Substring matching wrongly
# scored these CRITICAL ("armed" in "unarmed", "fire" in "firearm"); whole-word matching must not.
NO_FALSE_POSITIVE = [
    ("Unarmed guard completed a routine patrol; all clear.", "armed"),
    ("Employee cleaned the firearm display case in the lobby.", "fire"),
    # The firearm/shooting terms added to security/intrusion must not fire from inside benign
    # words: "shooting" in "troubleshooting" (extremely common in a facilities/IT incident log),
    # "shooter" in "troubleshooter"/"sharpshooter". Word boundaries (\b) must hold the line.
    ("Troubleshooting the printer took most of the morning.", "shooting"),
    ("The troubleshooter reset the panel; nothing else to note.", "shooter"),
    # The new gas-odor phrasings must stay whole-word: a benign mention of the word "gas" with no
    # odor/leak (a filled gas tank, a gas station errand) must NOT fire the gas/chemical floor.
    ("Refueled the generator; the gas tank is now full.", "gas smell"),
    ("Drove to the gas station for supplies; nothing to report.", "gas odor"),
    # The verb-order forms are adjacent-word phrases, so a bare "gas" errand with no smell report
    # (gas prices, a gas can) must NOT fire — "smell" and "gas" are not adjacent here.
    ("Gas prices went up again; refilled the gas can.", "smell gas"),
    # The new acute-medical terms must stay whole-word: a figurative "heart" mention (heartfelt
    # thanks, "at the heart of the issue") must NOT fire the injury/medical critical floor.
    ("A heartfelt thank-you note was left; nothing else to report.", "heart attack"),
    # "flames" is whole-word: the singular "flame" inside "flame-retardant" (and "inflames") must
    # NOT fire the fire/smoke critical floor — only the plural incident-word "flames" does.
    ("Inspected the flame-retardant coating on the ducts; nothing to report.", "flames"),
]


def test_embedded_substring_does_not_fire_floor():
    for text, embedded_kw in NO_FALSE_POSITIVE:
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev} (false positive from {embedded_kw!r})"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_higher_floor_wins_across_categories():
    # injury(high) + fire(critical) in one report -> critical floor.
    sev, reasons = risk.rule_layer("Fire broke out and one worker was injured")
    assert sev == "critical"
    # most-severe reason should be listed first
    assert "critical" in reasons[0]
