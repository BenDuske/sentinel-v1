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
        # "unconscious" sits at critical, but its plain-English synonyms — "lost consciousness" /
        # "loss of consciousness", the exact phrasing a non-clinical reporter writes for the SAME
        # event — matched nothing and dropped to LOW: the SAME word-choice asymmetry class as the
        # heart-attack / not-breathing fixes above. Offline the rule layer is the only floor, so add
        # these unambiguous phrases. Kept conservative — both are multi-word adjacency phrases with
        # no benign meaning, and the polysemous single-word synonyms a reporter might use are
        # deliberately NOT added: "passed out" ("passed out the agenda/flyers" = distribute) and
        # "unresponsive" ("the server/app is unresponsive" = IT) would over-fire, not fix a miss.
        # "CPR" is the single most unambiguous lay marker of a life-threatening arrest: a reporter
        # who writes "we're doing CPR" / "started CPR" is describing an active cardiac/respiratory
        # arrest (the critical event the "cardiac arrest"/"not breathing" floors already cover), yet
        # the bare word matched nothing and dropped to LOW — the SAME word-choice asymmetry class as
        # the heart-attack / lost-consciousness fixes. Offline the rule layer is the only floor, so
        # add it. The acronym has no benign English meaning, so \bcpr\b is safe from false positives.
        "critical": ["fatality", "fatalities", "death", "died", "deceased", "casualty",
                     "casualties", "unconscious", "lost consciousness", "loss of consciousness",
                     "cardiac arrest", "heart attack", "cpr", "anaphylaxis",
                     "anaphylactic", "not breathing", "severe bleeding", "amputation",
                     "life-threatening", "multiple injured"],
        # Respiratory distress is a serious medical emergency, but only apnea ("not breathing")
        # sat at the critical floor — the far more common lay reports of someone STILL breathing
        # but in distress ("trouble breathing", "can't breathe", "shortness of breath") matched
        # nothing and dropped to LOW: the SAME word-choice asymmetry class as the flames/gas-odor/
        # heart-attack fixes. Offline the rule layer is the only floor, so add these at HIGH (a
        # conservative floor — the LLM or a human can raise a specific case to critical; apnea stays
        # critical above). Every added signal is a multi-word adjacency phrase, so a benign lone
        # word ("take a breath", "trouble with the printer", "short of staff") does NOT fire.
        # "fainted" (a faint = a transient loss of consciousness) is added here at the HIGH floor —
        # ongoing "lost consciousness" sits at critical above, a brief faint is a conservative HIGH.
        # The whole-word matcher means the adjective "faint" ("a faint smell", "faint wifi signal")
        # does NOT fire — only the medical verb "fainted".
        # A convulsion (a person "convulsing" / having "convulsions") is a serious acute medical
        # event that previously matched nothing and dropped to LOW. The bare word "seizure" — its
        # clinical synonym — is DELIBERATELY excluded above because it is polysemous ("asset
        # seizure"), but the "convuls-" forms carry no such benign collision, so they safely fill
        # that gap at the conservative HIGH floor (the LLM or a human can raise a specific case;
        # ongoing arrest terms stay critical above). Each is a whole word with no benign meaning.
        "high":     ["injury", "injured", "hospitalized", "ambulance", "broken bone",
                     "fracture", "concussion", "burn", "burned", "electrocuted", "overdose",
                     "collapsed", "bleeding", "head injury", "trouble breathing",
                     "difficulty breathing", "can't breathe", "cannot breathe",
                     "struggling to breathe", "shortness of breath", "short of breath",
                     "gasping for air", "respiratory distress", "fainted",
                     "convulsing", "convulsion", "convulsions"],
        "medium":   ["first aid", "minor injury", "slip", "trip", "fall", "fell", "sprain",
                     "bruise", "cut", "laceration", "dizzy", "nausea"],
    },
    "fire/smoke": {
        # "flames" is the most common lay word a reporter writes for an active fire, yet it was
        # absent while "fire"/"ablaze"/"blaze"/"engulfed" were present — so "the building is in
        # flames" / "visible flames on the roof" (no literal "fire" token) scored LOW, the SAME
        # word-choice asymmetry class as the gas-odor and heart-attack fixes. Offline the rule layer
        # is the only floor, so add it. Whole-word matching keeps it conservative: \bflames\b does
        # not fire from "inflames" or the singular "flame" in "flame-retardant"/"flame war"; only the
        # plural incident-word "flames" (nearly always literal fire) is added, not bare "flame".
        "critical": ["fire", "flames", "ablaze", "blaze", "explosion", "explosive", "engulfed",
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
        # A person receiving an "electric shock" is an injury/hazard the taxonomy already floors at
        # HIGH via "electrical" — but ONLY for the adjective "electrical". The far more common lay
        # phrasing "electric shock" (adjective "electric", no -al) matched nothing and dropped to
        # LOW: the SAME hazard scored HIGH or LOW purely on the electric/electrical word choice (the
        # same word-choice asymmetry class as the flames / gas-odor / heart-attack floor fixes).
        # Offline the rule layer is the only floor, so add the "electric shock" forms. Kept
        # conservative — these are multi-word adjacency phrases, so the bare polysemous noun "shock"
        # ("culture shock", "shock absorber", "the news was a shock") is NOT added and does not fire;
        # and the plural "electric shocks" needs its own entry because \bshock\b won't match "shocks".
        "high":     ["electrical", "electric shock", "electric shocks", "exposed wiring",
                     "short circuit", "shorted", "sparking", "power surge",
                     "breaker tripped repeatedly", "burning wire"],
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
