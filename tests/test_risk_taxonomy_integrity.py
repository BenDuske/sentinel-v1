"""Structural integrity guards for the rule-layer taxonomy.

These tests assert nothing about *individual* incident scoring (that's test_risk_taxonomy.py).
They protect invariants of the TAXONOMY table itself so a future edit can't silently degrade the
grounded, auditable floor that is Sentinel's differentiator. All pure, deterministic, no LLM/network.
"""
from sentinel import risk, risk_fallback

_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def test_every_keyword_matches_itself():
    """Round-trip guard for the whole-word matchers (a22d1ea).

    Word-boundary matching (`\\bKW\\b`) fails to fire if a keyword begins or ends with a
    non-word character — such a keyword would be *dead* (unfireable) and silently lower the
    floor. Assert every taxonomy signal still matches its own text.
    """
    dead = [kw for kw, rx in risk._MATCHERS.items() if not rx.search(kw.lower())]
    assert not dead, f"whole-word matcher never fires for these keywords: {dead}"


def test_no_duplicate_keyword_within_category():
    """A keyword listed at two severities inside one category makes the lower one dead code
    (the floor always takes the higher). Keep each category's keyword set unambiguous."""
    dupes = []
    for category, levels in risk.TAXONOMY.items():
        seen = {}
        for sev, kws in levels.items():
            for kw in kws:
                if kw in seen:
                    dupes.append(f"{category}: {kw!r} in both {seen[kw]!r} and {sev!r}")
                seen[kw] = sev
    assert not dupes, "duplicate keyword(s) across severities: " + "; ".join(dupes)


def _shadowed():
    """A keyword is *shadowed* when a higher-severity keyword in the SAME category is a
    whole-word subset of it, so the author's intended (lower) floor for the specific phrase can
    never govern — the broader keyword always co-fires and wins. Returns {(category, keyword)}."""
    out = set()
    for category, levels in risk.TAXONOMY.items():
        entries = [(sev, kw) for sev, kws in levels.items() for kw in kws]
        for sev, kw in entries:
            for sev2, kw2 in entries:
                if kw == kw2:
                    continue
                if _RANK[sev2] > _RANK[sev] and risk._MATCHERS[kw2].search(kw.lower()):
                    out.add((category, kw))
                    break
    return out


# Known shadowed phrases as of 2026-07-02. Each is a PRECISION issue: the author placed the
# specific phrase at a lower severity, but a broader same-category keyword over-escalates it
# ("fire alarm" -> critical via "fire"; "mild fumes" -> critical via "fumes"). Honoring the
# author's declared floor needs most-specific-match-wins logic, which changes safety-scoring
# semantics and is deliberately queued for the human taxonomy/hardening review, not auto-applied.
# This guard pins the KNOWN set so any *new* shadowing introduced by a future edit fails loudly.
KNOWN_SHADOWED = {
    ("injury/medical", "minor injury"),   # -> high via "injury"
    ("fire/smoke", "fire alarm"),         # -> critical via "fire"
    ("water/flood", "minor leak"),        # -> high via "leak"
    ("gas/chemical", "mild fumes"),       # -> critical via "fumes"
    ("theft", "petty theft"),             # -> high via "theft"
    ("outage", "partial outage"),         # -> high via "outage"
}


def test_fallback_actions_cover_every_category():
    """Offline-path parity guard for the category->actions table.

    When no LLM is reachable, ``risk_fallback.fallback_actions`` maps each matched TAXONOMY
    category to concrete next steps via ``_CATEGORY_ACTIONS``. That table is keyed by category
    name, so a future rename/addition in ``risk.TAXONOMY`` would silently drop the affected
    category to the generic actions with no test failing. Pin exact key parity in both directions
    so the offline recommendations stay category-aware and defensible. Pure/structural — asserts
    nothing about individual scoring semantics.
    """
    taxonomy_cats = set(risk.TAXONOMY)
    action_cats = set(risk_fallback._CATEGORY_ACTIONS)
    missing = taxonomy_cats - action_cats
    orphan = action_cats - taxonomy_cats
    assert not missing, (
        "TAXONOMY category has no offline actions in _CATEGORY_ACTIONS (would fall back to "
        f"generic-only next steps): {sorted(missing)}"
    )
    assert not orphan, (
        "_CATEGORY_ACTIONS names a category not in risk.TAXONOMY (dead entry / stale name): "
        f"{sorted(orphan)}"
    )


def test_no_new_shadowed_keywords():
    current = _shadowed()
    new = current - KNOWN_SHADOWED
    resolved = KNOWN_SHADOWED - current
    assert not new, (
        "new shadowed keyword(s) introduced — a broader same-category keyword over-escalates "
        f"the author's declared floor: {sorted(new)}. Either rephrase or make the escalation "
        "intentional and update KNOWN_SHADOWED."
    )
    # If a shadow is fixed/removed, tighten the pin so it can't silently regress later.
    assert not resolved, (
        f"these previously-shadowed keywords no longer shadow — remove from KNOWN_SHADOWED: {sorted(resolved)}"
    )
