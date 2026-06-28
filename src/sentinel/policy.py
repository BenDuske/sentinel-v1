"""Safety, consent, and ethics layer — stdlib only.

Three responsibilities, all local and dependency-free:

1. BASELINE_POLICY  — the non-negotiable safety rules, prepended to every LLM system prompt.
2. Ethics onboarding — optional org-supplied mission/voice/boundaries layered ABOVE the user
   request but BELOW the baseline (config can tighten, never loosen).
3. Consent gate     — first-run agreement to LICENSE/ToS/AUP/Privacy/Disclaimer, recorded
   locally with a timestamp + policy version.

This is defense-in-depth, not a guarantee: the screen() heuristic hard-blocks only the most
clearly prohibited inputs (zero-tolerance: sexualization of minors) and is deliberately narrow
so that legitimate incident reports — fires, injuries, assaults, break-ins, crime — are still
classified and summarized normally. The BASELINE_POLICY steers the model, and a human stays in
the loop for anything consequential (see docs/legal/DISCLAIMER.md).
"""
import os
import re
import time

from . import config

POLICY_VERSION = "2026-06-28"

# --- 1. Baseline safety policy (always on, cannot be overridden by ethics config) ----------
BASELINE_POLICY = (
    "SAFETY POLICY (highest priority, non-negotiable):\n"
    "- Operate only within the law: U.S. federal law and the operator's own state and local "
    "law. Where laws differ, follow the most restrictive that applies.\n"
    "- Refuse to produce: sexual content, nudity, or explicit adult role-play; and ALWAYS "
    "refuse, with zero tolerance, anything sexualizing a minor.\n"
    "- Refuse to help with content that enables serious physical harm (weapons of mass harm, "
    "explosives, illicit drug/toxin synthesis), unauthorized intrusion/malware, fraud, "
    "harassment, or privacy violations.\n"
    "- This is incident-intelligence software: documenting, classifying, and summarizing real "
    "incidents (including violence, injury, theft, fire, and other crime) is the legitimate "
    "purpose and is fully supported. Describe such incidents factually and professionally.\n"
    "- AI output (severity, summary, recommended actions) is assistive risk decision-support, "
    "not an authoritative underwriting, legal, or safety determination, and is not professional "
    "advice. Encourage human review for consequential decisions.\n"
    "- When a request crosses these lines, briefly refuse and, where possible, offer a safe, "
    "lawful alternative. Do not reveal how to circumvent these rules."
)

# A short, user-facing refusal the caller can surface when screen() hard-blocks.
REFUSAL_MESSAGE = (
    "This input was blocked by Sentinel's safety policy "
    "(see docs/legal/ACCEPTABLE_USE_POLICY.md). Sentinel documents and classifies real "
    "incidents, but it will not process content that sexualizes a minor or is otherwise "
    "prohibited. Please revise the incident description toward a lawful, factual report."
)

# --- 2. Best-effort input screen --------------------------------------------------------------
# Heuristic only. The model + BASELINE_POLICY are the primary gate; this is a coarse net for the
# most clearly prohibited inputs. ONLY the ZERO-TOLERANCE category (sexualization of minors) is
# hard-blocked by default, so that legitimate incident text — "fire broke out, two injured",
# "assault in the parking lot", "break-in overnight" — classifies normally. Strict adult-sexual
# screening is opt-in via SENTINEL_STRICT_SCREEN.
_MINOR_TERMS = r"(child|children|minor|underage|preteen|pre-teen|infant|toddler|kid|kids|\b(?:1[0-7]|[0-9])\s*(?:yo|y/o|year[\s-]?old)s?)"
_SEXUAL_TERMS = r"(sexual|sex\b|nude|naked|porn|explicit|nsfw|erotic|fellatio|genital|aroused)"
_ZERO_TOLERANCE = re.compile(_MINOR_TERMS + r".{0,40}" + _SEXUAL_TERMS, re.I | re.S)
_ZERO_TOLERANCE_REV = re.compile(_SEXUAL_TERMS + r".{0,40}" + _MINOR_TERMS, re.I | re.S)

_STRICT = os.environ.get("SENTINEL_STRICT_SCREEN", "") not in ("", "0", "false", "False")
_SEXUAL_ONLY = re.compile(r"\b(porn|nsfw|erotic|sexually explicit|adult role[\s-]?play)\b", re.I)


def screen(text: str):
    """Return (allowed: bool, category: str|None, message: str|None).

    Hard-blocks any apparent sexualization of a minor unconditionally. Other categories are
    allowed by default (and blocked only under SENTINEL_STRICT_SCREEN) so that legitimate
    incident reports — violence, injury, crime — are still classified; the BASELINE_POLICY in
    the system prompt handles the rest.
    """
    if not text:
        return True, None, None
    if _ZERO_TOLERANCE.search(text) or _ZERO_TOLERANCE_REV.search(text):
        return False, "minor_safety", REFUSAL_MESSAGE
    if _STRICT and _SEXUAL_ONLY.search(text):
        return False, "sexual_content", REFUSAL_MESSAGE
    return True, None, None


# --- 3. Ethics onboarding (optional, layered ABOVE the request, BELOW the baseline) ----------
_ETHICS_KEYS = (
    "organization", "mission", "values", "voice", "goals",
    "boundaries", "required_disclosures",
)


def load_ethics(path: str = None) -> dict:
    """Parse the flat key: value ethics file (a tiny YAML subset; stdlib only)."""
    path = path or os.environ.get("SENTINEL_ETHICS_FILE", "")
    if not path or not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.lstrip().startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key in _ETHICS_KEYS:
                    out[key] = val.strip().strip('"').strip("'")
    except Exception:
        return {}
    return out


def ethics_preamble(ethics: dict = None) -> str:
    ethics = ethics if ethics is not None else load_ethics()
    if not ethics:
        return ""
    lines = ["ORGANIZATION CONTEXT (applies unless it conflicts with the SAFETY POLICY above):"]
    label = {
        "organization": "Organization", "mission": "Mission", "values": "Values",
        "voice": "Voice/tone", "goals": "Goals", "boundaries": "Boundaries",
        "required_disclosures": "Always disclose",
    }
    for k in _ETHICS_KEYS:
        if ethics.get(k):
            lines.append(f"- {label[k]}: {ethics[k]}")
    return "\n".join(lines)


def system_preamble(persona: str = "") -> str:
    """Baseline safety policy + optional org ethics + persona/instructions, in precedence order.

    Precedence (highest first): baseline safety policy -> org ethics -> the caller's persona or
    task instructions. The org ethics file can make Sentinel more specific or more restrictive;
    nothing in it can loosen the baseline.
    """
    blocks = [BASELINE_POLICY]
    eth = ethics_preamble()
    if eth:
        blocks.append(eth)
    if persona:
        blocks.append(persona)
    return "\n\n".join(blocks)


# --- consent gate ----------------------------------------------------------------------------
def _data_dir() -> str:
    """Local Sentinel data dir for the consent record.

    Derived from the configured SQLite DB location so the consent marker lives alongside the
    operator's incident data; falls back to ~/.sentinel when the DB path has no directory.
    """
    db = getattr(config, "DB_PATH", "") or ""
    d = os.path.dirname(os.path.abspath(os.path.expanduser(db))) if db else ""
    if not d:
        d = os.path.join(os.path.expanduser("~"), ".sentinel")
    return d


def _consent_path() -> str:
    return os.path.join(_data_dir(), ".sentinel_consent.json")


def has_consent() -> bool:
    if os.environ.get("SENTINEL_ASSUME_CONSENT", "") not in ("", "0", "false", "False"):
        return True
    p = _consent_path()
    if not os.path.exists(p):
        return False
    try:
        import json
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh).get("version") == POLICY_VERSION
    except Exception:
        return False


def record_consent() -> None:
    import json
    p = _consent_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"version": POLICY_VERSION, "ts": int(time.time()),
                   "agreed": True}, fh)


CONSENT_PROMPT = (
    "\n=== Sentinel v1 — agreement required before first use ===\n"
    "By continuing you agree to the MIT License, Terms of Service, Acceptable Use Policy,\n"
    "Privacy Policy, and AI Output & Warranty Disclaimer (see the repo's LICENSE,\n"
    "CONSENT.md, and docs/legal/). In short:\n"
    "  - Lawful use only (U.S. federal + your state/local law).\n"
    "  - No sexual/explicit content, and zero tolerance for sexualizing minors.\n"
    "  - Incident data and uploaded evidence stay local on your machine.\n"
    "  - AI output may be wrong and is NOT professional advice — you verify and decide.\n"
    "  - Provided 'as is', no warranty, at your own risk.\n"
    f"  Policy version: {POLICY_VERSION}\n"
    "Type AGREE to accept, anything else to exit: "
)

# Short banner the web/server layer can log at startup (points to CONSENT.md).
STARTUP_BANNER = (
    "Sentinel v1 — local-first incident intelligence. "
    f"By using this software you accept the terms in CONSENT.md (policy {POLICY_VERSION}): "
    "lawful use only, no sexual/abusive content, AI output is decision-support not "
    "authoritative, provided as-is at your own risk. Incident data stays on this machine."
)


def ensure_consent(input_fn=input, output_fn=print) -> bool:
    """Interactive first-run gate (for CLI/automation). Returns True if consent is on file."""
    if has_consent():
        return True
    try:
        answer = input_fn(CONSENT_PROMPT)
    except (EOFError, KeyboardInterrupt):
        output_fn("\nNo agreement recorded — exiting.")
        return False
    if answer.strip().upper() == "AGREE":
        record_consent()
        output_fn("Agreement recorded. Welcome to Sentinel.\n")
        return True
    output_fn("You did not agree — exiting. (Nothing was changed.)")
    return False
