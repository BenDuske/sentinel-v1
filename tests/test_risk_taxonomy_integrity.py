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


def _effective_shadowed():
    """A keyword is *effectively* shadowed when the REAL scoring path over-escalates it.

    ``_shadowed()`` above only inspects subsets WITHIN a single category, but ``risk.rule_layer``
    takes the floor as the max across ALL categories — so a broader keyword in a *different*
    category can silently over-escalate a phrase and the same-category guard never sees it
    (e.g. injury/medical "collapsed" -> critical via the structural category's "collapsed").
    This runs the actual ``rule_layer`` on each keyword in isolation and flags every phrase whose
    effective severity exceeds its declared floor, regardless of which category caused it.
    Returns {(category, keyword, declared_sev, effective_sev)}."""
    out = set()
    for category, levels in risk.TAXONOMY.items():
        for sev, kws in levels.items():
            for kw in kws:
                eff, _ = risk.rule_layer(kw)
                if _RANK[eff] > _RANK[sev]:
                    out.add((category, kw, sev, eff))
    return out


# Effective (cross-category-aware) shadow set as of 2026-07-08. Superset of KNOWN_SHADOWED: the
# same 6 in-category cases PLUS two CROSS-category escalations the same-category guard is blind to
# — injury/medical "collapsed" and water/flood "ceiling collapse from water" both hit the
# structural category's "collapse"/"collapsed" (critical). Same precision class as the documented
# 6; the semantic fix (most-specific-match-wins) stays queued for the human taxonomy review. This
# pin makes the regression net COMPLETE so a future edit can't slip in a new cross-category shadow.
KNOWN_EFFECTIVE_SHADOWED = {
    ("injury/medical", "minor injury", "medium", "high"),
    ("injury/medical", "collapsed", "high", "critical"),        # CROSS -> structural "collapsed"
    ("fire/smoke", "fire alarm", "high", "critical"),
    ("water/flood", "minor leak", "medium", "high"),
    ("water/flood", "ceiling collapse from water", "high", "critical"),  # CROSS -> structural "collapse"
    ("gas/chemical", "mild fumes", "medium", "critical"),
    ("theft", "petty theft", "medium", "high"),
    ("outage", "partial outage", "medium", "high"),
}


def test_no_new_effective_shadowed_keywords():
    """Cross-category-aware companion to test_no_new_shadowed_keywords.

    Guards the ACTUAL scoring path (``rule_layer`` maxes across all categories), not just
    same-category subsets. Any new phrase whose effective floor exceeds its declared floor — via
    any category — fails here even if the same-category guard stays green."""
    current = _effective_shadowed()
    new = current - KNOWN_EFFECTIVE_SHADOWED
    resolved = KNOWN_EFFECTIVE_SHADOWED - current
    assert not new, (
        "new effective (possibly cross-category) shadow(s) — a broader keyword in some category "
        f"over-escalates the declared floor via rule_layer: {sorted(new)}. Rephrase, or make the "
        "escalation intentional and update KNOWN_EFFECTIVE_SHADOWED."
    )
    assert not resolved, (
        "these previously effective-shadowed keywords no longer over-escalate — remove from "
        f"KNOWN_EFFECTIVE_SHADOWED so it can't silently regress: {sorted(resolved)}"
    )


def test_effective_shadow_set_supersets_same_category():
    """Sanity: every same-category shadow is also an effective shadow (the real path can only add
    escalations, never remove them). Catches drift between the two guards' keyword spellings."""
    same_cat_keys = {(c, kw) for (c, kw) in _shadowed()}
    eff_keys = {(c, kw) for (c, kw, _, _) in _effective_shadowed()}
    missing = same_cat_keys - eff_keys
    assert not missing, (
        f"same-category shadow not reflected in the effective set (guard drift): {sorted(missing)}"
    )


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
