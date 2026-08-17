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
        # "no pulse" / "no heartbeat" / "pulseless" are the EMS/lay phrasings a reporter writes for
        # the SAME life-threatening cardiac arrest the "cardiac arrest"/"heart attack"/"cpr"/"not
        # breathing" floors already cover ("he has no pulse", "she's pulseless", "no heartbeat"), yet
        # they matched nothing and dropped to LOW — the SAME word-choice asymmetry class as those
        # fixes. Offline the rule layer is the only floor, so add them. Kept conservative: "no pulse"
        # and "no heartbeat" are multi-word adjacency phrases and "pulseless" is a single clinical
        # word (pulseless electrical activity), none with a benign meaning; the bare polysemous noun
        # "pulse" ("pulse oximeter", "the pulse of the organization", an electrical pulse) is
        # DELIBERATELY excluded — it would over-fire, not fix a miss.
        # Apnea (a person not breathing) sits at critical, but ONLY the present-tense phrasing "not
        # breathing" was listed — and by substring it only covers "is/are/was not breathing". The
        # equally-common lay phrasings for the SAME arrest — the past tense "stopped breathing", the
        # "no longer breathing" form, and the contractions "isn't breathing" / "wasn't breathing"
        # (which do NOT contain the substring "not breathing") — all matched nothing and dropped to
        # LOW: the SAME word-choice/tense asymmetry class as the heart-attack / no-pulse fixes above.
        # Offline the rule layer is the only floor, so add them. Kept conservative: each is a
        # multi-word adjacency phrase describing apnea with no benign meaning, so a lone "breathing"
        # / "breath" mention ("took a breather", "breathing room in the budget") does NOT fire.
        # A ruptured "aneurysm" and a pulmonary "embolism" are acute, frequently-fatal vascular
        # emergencies a reporter names directly ("he had a brain aneurysm", "suspected pulmonary
        # embolism"), yet both matched nothing and dropped to LOW — the SAME word-choice asymmetry
        # class as the heart-attack / no-pulse fixes above. Offline the rule layer is the only floor,
        # so add them (plus "aneurism", the common lay misspelling). These are DELIBERATELY safer
        # additions than the polysemous neighbors still excluded above: "stroke" (keystroke / brush
        # stroke / stroke of luck) and "seizure" (asset seizure) carry benign collisions and would
        # over-fire, but "aneurysm"/"aneurism"/"embolism" are whole clinical words with NO benign
        # English meaning, so they close the miss without any false-positive risk.
        # The NOUN "amputation" floors at critical here, but the participle/verb form "amputated" —
        # the way an acute report is actually written ("his arm was amputated", "amputated finger",
        # "traumatically amputated") — matched nothing and dropped to LOW: the SAME verb-vs-noun
        # word-form asymmetry already fixed for lightning-struck / smell-gas / electric-shocks, the
        # same hazard scored critical-or-LOW purely on grammatical form. "amputated" is a whole word
        # with NO benign English meaning, so it closes the miss at the noun's floor with zero
        # false-positive risk (the chronic descriptor "amputee" is deliberately NOT added — "amputee
        # support group" / "amputee parking" is not an acute emergency).
        # A traumatic "decapitation" — and its participle/verb form "decapitated" ("the worker was
        # decapitated by the machine", "decapitated at the roller") — is an unambiguous, virtually
        # always-fatal trauma a reporter names directly, yet BOTH the noun and the participle matched
        # nothing and dropped to LOW: the SAME whole-clinical-word miss class as aneurysm / embolism /
        # impaled / amputated above — an entire severe-trauma word simply absent from the floor.
        # Offline the rule layer is the only floor, so add both at critical. Neither has any benign
        # English meaning (unlike the polysemous neighbors "crushed"/"pinned" deliberately excluded),
        # so they close the miss with zero false-positive risk; the participle is a separate entry
        # because \bdecapitation\b does not match "decapitated".
        "critical": ["fatality", "fatalities", "death", "died", "deceased", "casualty",
                     "casualties", "unconscious", "lost consciousness", "loss of consciousness",
                     "cardiac arrest", "heart attack", "cpr", "no pulse", "no heartbeat",
                     "pulseless", "anaphylaxis", "aneurysm", "aneurism", "embolism",
                     "anaphylactic", "not breathing", "stopped breathing", "no longer breathing",
                     "isn't breathing", "wasn't breathing", "severe bleeding",
                     "amputation", "amputated", "decapitation", "decapitated",
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
        # "burn"/"burned" already sit at HIGH, but the PLURAL noun "burns" — the single most common
        # way a burn injury is actually reported ("severe burns", "third-degree burns", "burns to
        # the hands") — matched nothing and dropped to LOW, because \bburn\b does not match "burns":
        # the SAME tokenization miss already fixed for the electrical "electric shocks" plural below.
        # Offline the rule layer is the only floor, so add it. "impaled" is added alongside as an
        # unambiguous severe-trauma term the probe surfaced at LOW — a whole word with no benign
        # English meaning, so it carries no false-positive risk (unlike polysemous neighbors such as
        # "crushed"/"choking" that were deliberately left out).
        # "hypothermia" (and the adjective "hypothermic", e.g. "found hypothermic in the walk-in
        # freezer") is an acute exposure emergency a reporter names directly, yet it matched nothing
        # and dropped to LOW — the SAME word-choice miss class as aneurysm/embolism/impaled above. It
        # is a whole clinical word with NO benign English meaning, so it closes the miss with zero
        # false-positive risk; added at the conservative HIGH floor (the LLM or a human can raise a
        # specific severe case). Surfaced and flagged safe in the 2026-08-16 rule-probe backlog.
        # The HOT counterpart to hypothermia was the mirror-image gap: "heat stroke"/"heatstroke"
        # (and its clinical synonym "hyperthermia"/"hyperthermic", plus the milder "heat exhaustion")
        # is a life-threatening exposure emergency a reporter names directly — "collapsed with heat
        # stroke on the roof", "found hyperthermic in the boiler room" — yet all matched nothing and
        # dropped to LOW while hypothermia now floors at HIGH: the SAME exposure-emergency class, one
        # commit apart, split only by which extreme the reporter's word points at. Offline the rule
        # layer is the only floor, so add them at the same conservative HIGH floor as hypothermia.
        # Kept conservative: "heat stroke"/"heat exhaustion" are multi-word adjacency phrases and
        # "heatstroke"/"hyperthermia"/"hyperthermic" are whole clinical words with NO benign meaning,
        # so a bare "heat" mention (a heat wave, a heat exchanger, "turn up the heat", the "heat
        # advisory" that already sits at weather/medium) does NOT fire the injury/medical floor.
        # "bleeding" sits at HIGH, but its direct clinical synonym "hemorrhage"/"hemorrhaging" —
        # the word a report actually uses for profuse blood loss ("worker is hemorrhaging", "massive
        # hemorrhage") — matched nothing and dropped to LOW: the SAME word-choice asymmetry class as
        # heart-attack/flames/electric-shock/amputated, the same injury scored HIGH-or-LOW purely on
        # which synonym the reporter chose. Added at the same HIGH floor as "bleeding" (whose money
        # metaphor "bleeding cash" already fires there, so this introduces no new over-fire class).
        # Both the noun and the participle are needed (\bhemorrhage\b does not match "hemorrhaging"),
        # and whole-word matching keeps the benign prefix-sharer "hemorrhoid"/"hemorrhoids" — not an
        # acute emergency — from firing.
        # The NOUN "concussion" sits at HIGH, but the participle "concussed" — how an acute report is
        # actually written ("worker was concussed", "concussed and disoriented") — matched nothing
        # (\bconcussion\b does not match "concussed") and dropped to LOW: the SAME verb-vs-noun form
        # gap as amputation/amputated and hemorrhage/hemorrhaging, the same head injury scored
        # HIGH-or-LOW purely on grammatical form. Added at the noun's HIGH floor. Unlike the
        # deliberately-excluded polysemous "stroke"/"seizure", "concussed" is a whole clinical word
        # with NO benign English meaning, so it closes the miss with zero false-positive risk.
        # "broken bone" sits at HIGH, but the PLURAL "broken bones" — the single most common way the
        # injury is actually reported ("multiple broken bones", "several broken bones") — matched
        # nothing and dropped to LOW, because \bbroken bone\b does not match "broken bones": the SAME
        # plural-tokenization miss already fixed for the singular->plural "burn"->"burns" and the
        # electrical "electric shock"->"electric shocks" below. Offline the rule layer is the only
        # floor, so add it. Kept conservative: "broken bones" is a multi-word adjacency phrase with no
        # benign meaning (unlike bare "broken" = broken machine), so it fires only on the injury
        # phrasing. The idiom "no broken bones" (= unharmed) over-fires HIGH, but that is NOT a new
        # over-fire class — the already-accepted singular "broken bone" fires the same way on "no
        # broken bone", and the rule layer is a conservative floor the LLM/human can lower.
        # The NOUN "overdose" sits at HIGH, but the participle/verb form "overdosed" — how an acute
        # report is actually written ("worker overdosed in the restroom", "he overdosed on the
        # loading dock") — matched nothing (\boverdose\b does not match "overdosed") and dropped to
        # LOW: the SAME verb-vs-noun form gap already fixed for amputation/amputated,
        # concussion/concussed, and hemorrhage/hemorrhaging, the same medical emergency scored
        # HIGH-or-LOW purely on grammatical form. Added at the noun's HIGH floor. "overdosed" is a
        # whole clinical word with NO benign English meaning, so it closes the miss with zero
        # false-positive risk (unlike the deliberately-excluded polysemous "stroke"/"seizure").
        # The NOUN "fracture" sits at HIGH, but the participle/verb "fractured" ("fractured his leg",
        # "fractured skull") and the PLURAL "fractures" ("multiple fractures", "several stress
        # fractures") — the way a break is actually reported — matched nothing (\bfracture\b does not
        # match "fractured"/"fractures") and dropped to LOW: the SAME verb-vs-noun + singular->plural
        # word-form gap already fixed for concussion/concussed, overdose/overdosed, amputation/
        # amputated, and burn/burns, the same injury scored HIGH-or-LOW purely on grammatical form.
        # Added at the noun's HIGH floor. This introduces NO new over-fire class the accepted bare
        # "fracture" doesn't already carry: the money/politics metaphor ("a fractured coalition")
        # fires the same way the existing "fracture" does on "a fracture in the coalition" — and the
        # rule layer is a conservative floor the LLM/human can lower.
        # The singular "injury" and the participle "injured" both sit at HIGH, but the PLURAL noun
        # "injuries" — the single most common way a multi-casualty report is actually written
        # ("multiple injuries", "several injuries reported", "workers sustained injuries") — matched
        # nothing (\binjury\b does not match "injuries") and dropped to LOW: the SAME singular->plural
        # tokenization gap already fixed for burn/burns, broken bone/broken bones, and fracture/
        # fractures, here on the most fundamental injury word of all. Added at the same HIGH floor as
        # "injury"/"injured". "injuries" is a whole clinical/lay word with NO benign collision (unlike
        # bare "burn" needing care), so this adds no new over-fire class; and because it whole-word
        # matches inside "head injuries", it also closes that plural of the existing "head injury".
        # "hospitalized" and "hemorrhage"/"hemorrhaging" sit at HIGH, but their British/international
        # spellings — "hospitalised" and "haemorrhage"/"haemorrhaging" — matched nothing and dropped
        # to LOW: the SAME word-form asymmetry class already fixed for the "aneurism" lay misspelling
        # above, here on en-GB orthography ("worker hospitalised", "patient is haemorrhaging"). Non-US
        # incident reports are routine (contractors, international tenants, imported PDF templates), so
        # the same injury scoring HIGH-or-LOW purely on which spelling the reporter learned is exactly
        # the miss the taxonomy exists to close. Added at the same HIGH floor as their US twins, which
        # introduces ZERO new over-fire class — the US spelling already fires identically. Whole-word
        # matching keeps the benign en-GB prefix-sharer "haemorrhoid"/"haemorrhoids" from firing, the
        # exact parallel to the existing "hemorrhoid" guard.
        "high":     ["injury", "injured", "injuries", "hospitalized", "hospitalised", "ambulance",
                     "broken bone", "broken bones",
                     "fracture", "fractured", "fractures",
                     "concussion", "concussed", "burn", "burned", "burns", "impaled",
                     "hypothermia", "hypothermic",
                     "heat stroke", "heatstroke", "hyperthermia", "hyperthermic",
                     "heat exhaustion",
                     "electrocuted", "overdose", "overdosed",
                     "collapsed", "bleeding", "hemorrhage", "hemorrhaging",
                     "haemorrhage", "haemorrhaging",
                     "head injury", "trouble breathing",
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
        # The NOUN "explosion" floors at critical, but its verb/participle form "exploded" — how an
        # acute report is actually written ("the transformer exploded", "a boiler exploded", "the
        # tank exploded on the roof") — matched nothing (\bexplosion\b does not match "exploded") and
        # dropped to LOW: the SAME verb-vs-noun word-form gap already fixed for amputation/amputated,
        # concussion/concussed, fracture/fractured, and hemorrhage/hemorrhaging, the same catastrophic
        # event scored critical-or-LOW purely on grammatical form. Added at the noun's critical floor.
        # This introduces NO new over-fire class the accepted noun "explosion" doesn't already carry:
        # the growth metaphor ("sales exploded") fires the same way the existing "explosion" does on
        # "explosion of growth"/"population explosion" — and the rule layer is a conservative floor
        # the LLM/human can lower.
        # The past-tense "exploded" now floors, but the PLURAL noun "explosions" ("secondary
        # explosions rocked the plant", "multiple explosions reported") and the present participle
        # "exploding" ("a transformer exploding on the roof", "batteries exploding in the bay") — the
        # equally-common ways an unfolding or multi-blast event is actually written — still matched
        # nothing (\bexplosion\b matches neither "explosions" nor "exploding", and \bexploded\b matches
        # neither) and dropped to LOW: the SAME word-form gap already fixed for the past tense
        # explosion/exploded, plus the singular->plural tokenization discipline applied to burn/burns
        # and fracture/fractures. Added at the noun's critical floor. These introduce NO new over-fire
        # class the accepted "explosion"/"exploded" don't already carry — the growth metaphor
        # ("costs exploding", "population explosions") fires the same way — and the rule layer is a
        # conservative floor the LLM/human can lower.
        "critical": ["fire", "flames", "ablaze", "blaze", "explosion", "exploded", "explosions",
                     "exploding", "explosive",
                     "engulfed", "structure fire", "wildfire", "conflagration"],
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
        # The NOUN "asphyxiation" floors at critical, but its verb/participle form "asphyxiated" —
        # how an acute report is actually written ("the worker was asphyxiated in the tank",
        # "asphyxiated in the confined space") — matched nothing (\basphyxiation\b does not match
        # "asphyxiated") and dropped to LOW: the SAME verb-vs-noun word-form gap already fixed for
        # explosion/exploded, amputation/amputated, concussion/concussed, and hemorrhage/hemorrhaging,
        # the same life-threatening event scored critical-or-LOW purely on grammatical form. Added at
        # the noun's critical floor. "asphyxiated" is a whole clinical word with NO benign English
        # meaning, so it closes the miss with zero false-positive risk and introduces no new over-fire
        # class the accepted noun "asphyxiation" doesn't already carry.
        # The past-tense "asphyxiated" now floors, but the present participle "asphyxiating" — how an
        # UNFOLDING exposure emergency is actually written ("workers asphyxiating in the tank",
        # "crews asphyxiating in the confined space") — still matched nothing (\basphyxiation\b does
        # not match "asphyxiating", and \basphyxiated\b does not either) and dropped to LOW: the SAME
        # present-participle word-form gap already fixed for exploded->exploding (fire/smoke) and
        # collapsed->collapsing (structural), the same live emergency scored critical-or-LOW purely on
        # grammatical form. Added at the noun's critical floor. "asphyxiating" is a whole clinical word
        # with NO benign English meaning, so it closes the miss with zero false-positive risk and
        # introduces no new over-fire class the accepted "asphyxiation"/"asphyxiated" don't already carry.
        "critical": ["gas leak", "carbon monoxide", "toxic", "chemical spill", "hazmat",
                     "hazardous material", "fumes", "asphyxiation", "asphyxiated", "asphyxiating",
                     "ammonia leak", "chlorine leak", "explosive gas"],
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
        # The noun/past-tense "collapse"/"collapsed" floor at critical, but the present participle
        # "collapsing" — how an UNFOLDING structural emergency is actually written ("the roof is
        # collapsing right now", "walls collapsing in the east wing", "the floor is actively
        # collapsing") — matched nothing (\bcollapse\b matches neither "collapsing" nor the "-ing"
        # suffix, and \bcollapsed\b does not either) and dropped to LOW: the SAME verb-form word gap
        # already fixed for exploded->exploding in fire/smoke, the same live emergency scored
        # critical-or-LOW purely on grammatical form. Added at the noun's critical floor. It
        # introduces NO new over-fire class the accepted "collapse"/"collapsed" don't already carry
        # — the metaphor ("the deal is collapsing", collapsing a menu/table) fires exactly the way
        # "the deal collapsed"/"collapse the menu" already can — and the rule layer is a conservative
        # floor the LLM/human can lower.
        "critical": ["collapse", "collapsed", "collapsing", "building collapse", "structural failure",
                     "imminent collapse", "foundation failure", "roof collapse"],
        # A "sinkhole" is an acute ground-failure emergency a reporter names directly ("a sinkhole
        # opened under the parking lot", "sinkhole swallowed the sidewalk"), yet it matched nothing
        # and dropped to LOW while its slower cousin "subsidence" already floors at HIGH here: the
        # SAME structural ground-failure class scored HIGH-or-LOW purely on which word the reporter
        # reached for. Offline the rule layer is the only floor, so add it at the same conservative
        # HIGH floor as "subsidence"/"sagging"/"buckling" (the LLM or a human can raise a specific
        # case where a structure is actively involved). "sinkhole" is a whole word with NO benign
        # English meaning, so it closes the miss with zero false-positive risk — the common word
        # "sink" (a kitchen sink, to sink a budget) does NOT fire, since \bsinkhole\b does not match
        # "sink". The plural "sinkholes" needs its own entry (\bsinkhole\b won't match "sinkholes"),
        # the same singular->plural tokenization discipline already applied to burn/burns and
        # injury/injuries in injury/medical above.
        "high":     ["crack in wall", "structural crack", "sagging", "buckling", "subsidence",
                     "sinkhole", "sinkholes",
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
        "high":     ["storm", "lightning strike", "lightning struck", "struck by lightning",
                     "hail", "high winds", "fallen tree",
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
