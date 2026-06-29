# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Seed the local Sentinel store with a handful of varied, realistic demo incidents.

Runs FULLY OFFLINE: it forces the LLM unreachable, so severity comes from the grounded rule
layer and summaries/actions come from the deterministic, category-aware fallbacks. No network,
no Ollama, no API key — perfect for a HackTitan demo or a clean screenshot.

Usage (from the repo root):

    python scripts/seed_demo.py            # seed into ./sentinel.db (or $SENTINEL_DB)
    python scripts/seed_demo.py --reset    # wipe existing incidents first
    SENTINEL_DB=./demo.db python scripts/seed_demo.py

Stdlib + the sentinel package only. Safe to re-run (each run inserts fresh incidents).
"""
import argparse
import os
import sys

# Make `import sentinel` work whether run from the repo root or elsewhere (src/ layout).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Force the LLM offline for deterministic, dependency-free seeding. Pointing the base at an
# unused localhost port makes ai.chat()/llm_severity() degrade to the rule layer + fallbacks.
os.environ.setdefault("SENTINEL_LLM_BASE", "http://127.0.0.1:1/v1")
os.environ.setdefault("SENTINEL_ASSUME_CONSENT", "1")

from sentinel import ai, store, risk  # noqa: E402
from sentinel.models import new_incident  # noqa: E402

# Five varied, realistic incidents spanning the taxonomy an insurer / facilities / security
# reviewer cares about: fire, water, injury, intrusion, outage.
DEMO_INCIDENTS = [
    (
        "Kitchen fire at rental duplex",
        "Tenant left a pan of oil unattended; grease fire ignited on the stovetop and spread to "
        "the cabinets before the extinguisher knocked it down. Heavy smoke throughout the unit, "
        "scorched cabinetry, one occupant with a minor burn to the forearm. Fire department "
        "responded and confirmed the fire was out.",
    ),
    (
        "Water leak in server room",
        "Burst pipe discovered overnight; approximately two inches of standing water around racks "
        "A3-A5. One technician slipped and twisted an ankle. UPS units at risk and one rack lost "
        "power.",
    ),
    (
        "Slip-and-fall injury in store entrance",
        "A customer slipped on a wet floor near the entrance during heavy rain; no caution sign "
        "was posted. The customer fell, struck their head, and was dazed. First aid was given and "
        "an ambulance was called as a precaution.",
    ),
    (
        "Overnight break-in at small-business warehouse",
        "Forced entry through a rear loading door overnight. Padlock cut, two laptops and assorted "
        "tools stolen. Security camera footage captured the intrusion. No one was on site at the "
        "time; police report filed.",
    ),
    (
        "Power outage at branch office",
        "A storm took down a utility line and the branch lost power for roughly three hours. The "
        "backup generator failed to start, so the building was dark and the alarm panel went "
        "offline. No injuries; perishable inventory in the break room fridge may be affected.",
    ),
]


def _analyze_offline(inc: dict) -> dict:
    """Same pipeline as the app's _analyze, but guaranteed offline (rule layer + fallbacks)."""
    inc["severity"], inc["severity_rationale"] = risk.score(inc)
    inc["summary"] = ai.summarize(inc)
    inc["recommended_actions"] = ai.recommend(inc)
    inc["ai_generated"] = True
    inc["status"] = "reviewing"
    return inc


def seed(reset: bool = False) -> list:
    """Insert the demo incidents (offline-analyzed) into the configured store. Returns them."""
    if reset:
        existing = store.list_all()
        if existing:
            import sqlite3
            from sentinel import config
            with sqlite3.connect(config.DB_PATH) as c:
                c.execute("DELETE FROM incidents")

    created = []
    for title, desc in DEMO_INCIDENTS:
        inc = new_incident(title, desc)
        _analyze_offline(inc)
        store.save(inc)
        created.append(inc)
    return created


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Seed Sentinel with demo incidents (offline).")
    ap.add_argument("--reset", action="store_true",
                    help="delete existing incidents before seeding")
    args = ap.parse_args(argv)

    from sentinel import config
    created = seed(reset=args.reset)

    print(f"Seeded {len(created)} demo incidents into {config.DB_PATH} (offline rule layer).")
    for inc in created:
        print(f"  [{inc['severity'].upper():>8}]  {inc['title']}")
    print("\nStart the dashboard to view them:")
    print("  SENTINEL_ASSUME_CONSENT=1 uvicorn sentinel.app:app --app-dir src")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
