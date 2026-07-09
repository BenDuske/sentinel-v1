"""Boundary + subordination guards for the safety/consent layer.

These pin two invariants in policy.py that the happy-path and fail-closed suites
leave unguarded, and that a refactor could silently break in a *safety-relevant*
direction:

1. screen()'s zero-tolerance minor-safety block turns on an AGE cutoff at the legal
   age of majority. `_MINOR_TERMS` matches ages 0-17 (`1[0-7]|[0-9]`). The existing
   tests only exercise one under-age case ("12 year old") and the word-based terms
   ("child"/"kid"); the 17-vs-18 numeric boundary itself is untested. If that bound
   ever shifts, the failure is invisible to the current suite and dangerous in BOTH
   directions: narrowing lets a minor slip through the zero-tolerance gate; widening
   falsely blocks legitimate ADULT incident reports (assault/injury of a 19- or
   45-year-old), which is the app's core purpose.

2. ethics_preamble()'s header must textually SUBORDINATE org ethics to the baseline
   safety policy ("...unless it conflicts with the SAFETY POLICY above"). The layering
   test only checks positional order; it would still pass if that clause were dropped,
   leaving the model no in-band signal that ethics config cannot loosen the baseline.

All keyless, no network, no real user data.
"""
import re

from sentinel import policy


# --- screen(): the minor-safety age boundary is exactly < 18 ---------------------------------
def test_minor_age_boundary_blocks_under_18_with_sexual_proximity():
    # Every age from 0..17, in proximity to a sexual term, must hard-block minor_safety.
    for age in (0, 7, 12, 15, 16, 17):
        allowed, cat, msg = policy.screen(f"a {age} year old, nude")
        assert not allowed and cat == "minor_safety" and msg, f"age {age} should hard-block"
    # The compact "17yo" / "7 y/o" spellings must block too (the age-clause allows yo | y/o).
    for text in ("17yo, sexual", "a 7 y/o, explicit"):
        allowed, cat, _ = policy.screen(text)
        assert not allowed and cat == "minor_safety", f"{text!r} should hard-block"


def test_adult_ages_are_not_minor_blocked():
    # 18 and up must NOT trip the minor gate — adult incident reports are the whole point.
    # (screen() runs in default/non-strict mode here, so these pass through entirely.)
    for age in (18, 19, 21, 45, 90):
        allowed, cat, _ = policy.screen(f"{age} year old victim, sexual assault reported")
        assert allowed and cat is None, f"age {age} must not be treated as a minor"


def test_adult_sexual_incident_report_still_classifies():
    # A factual adult-crime report (the legitimate use case) must pass the screen unblocked.
    allowed, cat, _ = policy.screen(
        "Sexual assault of a 34-year-old woman reported outside the venue; suspect fled."
    )
    assert allowed and cat is None


def test_minor_age_boundary_is_load_bearing_at_the_regex():
    # Prove the guard is not vacuous: the age alternation really encodes the 0-17 window.
    # If someone widened it to include 18 (e.g. 1[0-8]), test_adult_ages_are_not_minor_blocked
    # would fail; this asserts the source bound is the age-of-majority cutoff, not 18+.
    assert "1[0-7]" in policy._MINOR_TERMS
    # And the live behavior agrees: 17 blocks, 18 does not, via the SAME sexual proximity.
    b17, _, _ = policy.screen("17 year old, nude")
    b18, _, _ = policy.screen("18 year old, nude")
    assert b17 is False and b18 is True


# --- ethics_preamble(): header subordinates org ethics to the baseline safety policy ---------
def test_ethics_header_subordinates_to_safety_policy():
    pre = policy.ethics_preamble({"organization": "Northwind Mutual"})
    # The org block must announce, in-band, that it yields to the SAFETY POLICY. Positional
    # ordering alone (checked elsewhere) does not tell the model ethics cannot loosen safety.
    assert "SAFETY POLICY" in pre
    assert re.search(r"unless it conflicts with the SAFETY POLICY", pre), (
        "ethics header lost its subordination clause"
    )


def test_empty_ethics_emits_no_subordination_header():
    # No org config -> no ORGANIZATION CONTEXT block at all (nothing to subordinate).
    assert policy.ethics_preamble({}) == ""
    pre = policy.system_preamble("PERSONA-X")
    assert "ORGANIZATION CONTEXT" not in pre
