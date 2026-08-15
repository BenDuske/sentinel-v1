# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Grounded risk scoring — deterministic rule layer reconciled with the LLM.

Severity is NOT just "the AI said so." A transparent rule layer sets a defensible FLOOR from a
categorized keyword taxonomy (the categories an insurer / facilities / security reviewer expects:
injury/medical, fire/smoke, water/flood, electrical/power, gas/chemical, structural, security
breach/intrusion, theft, outage, weather). The LLM adds judgment on top; the final score is the
HIGHER of the two ("floor logic"), with a rationale that shows BOTH the rule hits and the LLM's
call. That auditability is what insurers and technical judges want.
"""
import re

from . import ai

_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_INV = {v: k for k, v in _RANK.items()}

# Categorized taxonomy: each category maps a severity FLOOR -> keyword/phrase signals.
# Matching is case-insensitive substring matching against "title. description". The floor is the
# highest severity whose signals appear; the rationale names the category + the matched terms so a
# human can audit exactly why the floor was set. Keep this explainable and conservative: signals
# imply a *minimum* severity, never a maximum (the LLM or a human can raise it).
TAXONOMY = {
    "injury/medical": {
        # "cardiac arrest" sits at critical, but its universal LAY synonym "heart attack" — the
        # phrasing a non-clinical reporter actually writes — matched nothing and dropped to LOW: the
        # SAME life-threatening event scored CRITICAL or LOW purely on word choice (the exact word-
        # choice asymmetry class as the gas smell/odor fix below). Anaphylaxis (a rapidly fatal
        # allergic reaction) was likewise absent. Offline the rule layer is the only floor, so add
        # these unambiguous acute-emergency terms. Kept conservative: whole-word matchers mean
        # "heart attack" won't fire from "heartfelt"/"heart of the matter", and polysemous bare
        # tokens ("stroke" = keystroke/brush stroke/stroke of luck, "seizure" = asset seizure) are
        # deliberately NOT added — that would over-fire, not fix a miss.
        "critical": ["fatality", "fatalities", "death", "died", "deceased", "casualty",
                     "casualties", "unconscious", "cardiac arrest", "heart attack", "anaphylaxis",
                     "anaphylactic", "not breathing", "severe bleeding", "amputation",
                     "life-threatening", "multiple injured"],
        "high":     ["injury", "injured", "hospitalized", "ambulance", "broken bone",
                     "fracture", "concussion", "burn", "burned", "electrocuted", "overdose",
                     "collapsed", "bleeding", "head injury"],
        "medium":   ["first aid", "minor injury", "slip", "trip", "fall", "fell", "sprain",
                     "bruise", "cut", "laceration", "dizzy", "nausea"],
    },
    "fire/smoke": {
        "critical": ["fire", "ablaze", "blaze", "explosion", "explosive", "engulfed",
                     "structure fire", "wildfire", "conflagration"],
        "high":     ["smoke", "smoldering", "scorch", "charred", "burning smell",
                     "fire alarm", "sparks"],
        "medium":   ["overheating", "hot to the touch", "burnt smell"],
    },
    "water/flood": {
        "critical": ["flood", "flooding", "flooded", "submerged", "sewage backup",
                     "burst main", "dam failure"],
        "high":     ["water damage", "burst pipe", "pipe burst", "leak", "leaking",
                     "standing water", "ceiling collapse from water", "overflow"],
        "medium":   ["drip", "dripping", "damp", "moisture", "condensation", "minor leak"],
    },
    "electrical/power": {
        "critical": ["live wire", "arc flash", "electrocution", "electrical fire"],
        "high":     ["electrical", "exposed wiring", "short circuit", "shorted", "sparking",
                     "power surge", "breaker tripped repeatedly", "burning wire"],
        "medium":   ["flickering lights", "tripped breaker", "loose outlet", "brownout"],
    },
    "gas/chemical": {
        "critical": ["gas leak", "carbon monoxide", "toxic", "chemical spill", "hazmat",
                     "hazardous material", "fumes", "asphyxiation", "ammonia leak",
                     "chlorine leak", "explosive gas"],
        # A gas ODOR is a leak indicator and the taxonomy intends it as a HIGH floor. Prior fixes
        # covered the NOUN-order phrasings ("smell of gas"/"odor of gas" + the "gas smell"/"gas odor"
        # noun-compounds and their "natural gas ..." variants), but the equally-common VERB-order
        # phrasing a person actually writes — "I smell gas", "we smell natural gas", "it smells like
        # gas" — still matched nothing and dropped to LOW: the SAME hazard scored HIGH or LOW purely
        # on word order. Offline the rule layer is the only floor, so add the verb-order forms to
        # finish closing the asymmetry. Conservative: these are whole-word adjacent phrases, so a
        # benign "gas" mention with no odor report (a filled gas tank, a gas station errand, gas
        # prices) still does NOT fire. ("gas leak" already sits at critical and subsumes "natural
        # gas leak" via the \bgas\s+leak\b matcher.)
        "high":     ["chemical", "spill", "odor of gas", "smell of gas", "gas smell", "gas odor",
                     "natural gas smell", "natural gas odor", "smell of natural gas",
                     "odor of natural gas", "smell gas", "smell natural gas", "smells like gas",
                     "smells of gas", "propane leak", "fuel leak", "corrosive"],
        "medium":   ["odor", "strong smell", "mild fumes"],
    },
    "structural": {
        "critical": ["collapse", "collapsed", "building collapse", "structural failure",
                     "imminent collapse", "foundation failure", "roof collapse"],
        "high":     ["crack in wall", "structural crack", "sagging", "buckling", "subsidence",
                     "load-bearing", "compromised", "leaning"],
        "medium":   ["hairline crack", "settling", "cosmetic crack", "loose railing"],
    },
    "security/intrusion": {
        # Firearm/shooting VIOLENT-EVENT terms (an actual discharge or active threat) belong at
        # the critical floor alongside "active shooter"/"armed": offline, the rule layer is the
        # ONLY floor, and a report like "shots fired; shooter fled" previously matched nothing and
        # scored low. Deliberately excludes the bare noun "firearm" — its mere presence (e.g. "the
        # firearm display case") is not an incident and is guarded against as a false positive in
        # tests; only discharge/active-threat terms are added here. Word boundaries (\b, below) keep
        # these from firing inside benign words — "shooter" not in "troubleshooter"/"sharpshooter",
        # "shooting" not in "troubleshooting".
        "critical": ["active shooter", "armed", "weapon", "hostage", "bomb threat",
                     "intruder armed", "kidnapping", "gunshot", "gunshots", "gunfire",
                     "shots fired", "active shooting", "shooter", "shooting"],
        "high":     ["break-in", "broke in", "broken into", "intrusion", "intruder",
                     "unauthorized access", "forced entry", "trespass", "assault",
                     "data breach", "breach", "ransomware", "malware", "compromised account"],
        "medium":   ["suspicious person", "suspicious activity", "tailgating", "prowler",
                     "loitering", "phishing", "failed login", "unauthorized attempt"],
    },
    "theft": {
        "critical": ["armed robbery", "robbery at gunpoint"],
        "high":     ["theft", "stolen", "robbery", "burglary", "looting", "embezzlement",
                     "missing equipment", "missing inventory"],
        "medium":   ["shoplifting", "petty theft", "missing item", "misplaced"],
    },
    "outage": {
        "critical": ["total outage", "complete outage", "datacenter down", "site-wide outage",
                     "all systems down"],
        "high":     ["outage", "power outage", "offline", "system down", "server down",
                     "service down", "network down", "downtime", "blackout"],
        "medium":   ["degraded", "slow response", "intermittent", "partial outage",
                     "latency", "timeout"],
    },
    "weather": {
        "critical": ["tornado", "hurricane", "earthquake", "flash flood", "wildfire",
                     "tsunami", "severe storm warning"],
        "high":     ["storm", "lightning strike", "hail", "high winds", "fallen tree",
                     "downed line", "ice storm", "blizzard"],
        "medium":   ["heavy rain", "wind damage", "snow", "frost", "heat advisory"],
    },
}


# Precompiled whole-word matchers for every taxonomy signal. Word-boundary matching prevents
# embedded-substring false positives that would otherwise fire the floor on benign text — e.g.
# "armed" inside "unarmed" or "fire" inside "firearm" wrongly scoring CRITICAL. \b sits on either
# side of each (possibly multi-word) phrase.
#
# Internal whitespace in a multi-word phrase is matched as \s+ (any run of whitespace), NOT a single
# literal space. Incident text is free-form — PDF-extracted descriptions, pasted form fields, and
# multi-line reports routinely put a newline, tab, or double space between words. With a lone
# escaped space, a critical signal like "shots fired" silently missed "shots\nfired" / "shots  fired"
# and scored LOW exactly when (offline) the rule layer is the only floor. Splitting on whitespace and
# rejoining with \s+ closes that gap while staying conservative: \s+ matches ONLY whitespace, so
# punctuation between the words (e.g. "shots. fired") still won't over-fire the phrase.
def _phrase_pattern(kw: str) -> str:
    return r"\b" + r"\s+".join(re.escape(tok) for tok in kw.split()) + r"\b"


_MATCHERS = {
    kw: re.compile(_phrase_pattern(kw))
    for levels in TAXONOMY.values()
    for kws in levels.values()
    for kw in kws
}


def rule_layer(text: str):
    """Return (severity, reasons). reasons is a list of human-readable rule hits.

    Scans the categorized taxonomy; the floor is the highest severity matched across all
    categories. Signals match on WHOLE words/phrases (word boundaries), so a keyword never fires
    from inside a larger word. Each reason names the category, the floor it implies, and the
    matched terms so the score is fully auditable.
    """
    t = (text or "").lower()
    best, reasons = -1, []
    for category, levels in TAXONOMY.items():
        for sev, kws in levels.items():
            hits = [k for k in kws if _MATCHERS[k].search(t)]
            if hits:
                reasons.append(f"{category} → {sev} (matched: {', '.join(hits)})")
                if _RANK[sev] > best:
                    best = _RANK[sev]
    if best < 0:
        return "low", ["no risk taxonomy signals matched (default floor: low)"]
    # Order reasons most-severe first for readability.
    reasons.sort(key=lambda r: -_RANK[r.split(" → ")[1].split(" ")[0]])
    return _INV[best], reasons


def score(incident: dict):
    """Return (severity, rationale). Combines rule floor + LLM judgment, takes the HIGHER.

    rationale shows BOTH sides explicitly (rule-layer hits and the LLM's call) plus the final
    reconciliation, so the score is defensible.
    """
    text = f"{incident.get('title', '')}. {incident.get('description', '')}"
    rule_sev, rule_reasons = rule_layer(text)

    llm_raw = ai.llm_severity(text)
    if llm_raw:
        llm_part = f"AI judgment → {llm_raw}."
        llm_rank = _RANK.get(llm_raw, 0)
    else:
        llm_part = "AI judgment → unavailable (offline); rule layer governs."
        llm_rank = -1  # offline: don't let a missing LLM lower the floor

    final = _INV[max(_RANK[rule_sev], llm_rank)]
    rule_part = "Rule layer → " + rule_sev + " [" + "; ".join(rule_reasons) + "]."
    rationale = (f"{rule_part} {llm_part} "
                 f"Final = higher of the two → {final}.")
    return final, rationale
