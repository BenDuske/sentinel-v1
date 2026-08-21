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
        # "dismemberment" — and its participle "dismembered" ("the worker was dismembered by the
        # machine", "traumatic dismemberment at the press") — is a catastrophic trauma on par with
        # decapitation / amputation (both already critical here), yet BOTH the noun and the participle
        # matched nothing and dropped to LOW: the SAME whole-clinical-word absent-term miss class.
        # Offline the rule layer is the only floor, so add both at critical beside decapitation. In an
        # incident report "dismembered"/"dismemberment" denotes only the physical trauma — the rare
        # literary figurative ("dismembered his argument") does not appear in operational incident text,
        # the same tolerance already accepted for "decapitated"; the participle is a separate entry
        # because \bdismemberment\b does not match "dismembered".
        # "cardiac arrest" floors at critical, but its clinical twin "respiratory arrest" — and
        # "cardiopulmonary arrest", the combined-arrest phrasing a responder writes — matched
        # nothing and dropped to LOW: the SAME word-choice asymmetry class as the cardiac-arrest /
        # heart-attack fixes, one arrest phrasing critical and its equally-severe sibling LOW purely
        # on which clinical term the reporter chose. The lay apnea phrasings ("stopped breathing",
        # "not breathing") already floor, but the clinical phrase "respiratory arrest" itself is not
        # a substring of any of them. Offline the rule layer is the only floor, so add both at the
        # noun's critical floor. Both are multi-word clinical phrases with NO benign English meaning
        # (unlike the polysemous "stroke"/"seizure" deliberately excluded), so they close the miss
        # with zero false-positive risk.
        # "severe bleeding" floors at critical and its clinical synonym "hemorrhage" sits at HIGH, but
        # the clinical term for fatal blood loss itself — "exsanguination" / the participle
        # "exsanguinated" ("the patient exsanguinated before EMS arrived", "cause of injury:
        # exsanguination") — matched nothing and dropped to LOW/MEDIUM: the SAME word-choice asymmetry
        # class as the heart-attack / hemorrhage fixes, the fatal endpoint of bleeding scored LOW purely
        # because the reporter used the clinical word. The lay phrasings a report actually writes ("bled
        # to death", "bleeding to death") already floor via the "death" token, but the bare clinical
        # word does not. Offline the rule layer is the only floor, so add both at the noun's critical
        # floor beside "severe bleeding". "exsanguination"/"exsanguinated" are whole clinical words with
        # NO benign English meaning (unlike the polysemous "bleed out" — "bleed out the brake line" /
        # "bleed the radiator" — deliberately excluded), so they close the miss with zero false-positive
        # risk; the participle is a separate entry because \bexsanguination\b does not match "exsanguinated".
        # "evisceration" — and its participle "eviscerated" ("the worker was eviscerated by the machine",
        # "traumatic abdominal evisceration at the press") — is a catastrophic trauma on par with
        # decapitation / dismemberment / amputation / impalement (all already critical here), yet BOTH the
        # noun and the participle matched nothing and dropped to LOW: the SAME whole-clinical-word absent-
        # term miss class as those trauma words. Offline the rule layer is the only floor, so add both at
        # critical beside dismemberment. In operational incident text "eviscerated"/"evisceration" denotes
        # only the physical trauma — the rare literary figurative ("eviscerated his argument") does not
        # appear in incident reports, the same tolerance already accepted for "decapitated"/"dismembered";
        # the participle is a separate entry because \bevisceration\b does not match "eviscerated".
        # "strangulation" — mechanical neck compression cutting off the airway/carotid flow, an
        # immediately life-threatening trauma a report names directly ("manual strangulation of the
        # worker caught in the machine", "death by strangulation confirmed by the coroner") — matched
        # nothing and dropped to LOW unless a coincident token (death/assault) happened to fire: the
        # SAME whole-clinical-word absent-term miss class as asphyxiation/suffocation (both already
        # critical in the gas category) and decapitation/dismemberment/evisceration here, the same
        # fatal airway-occlusion event scored critical-or-LOW purely on whether another word coincided.
        # Offline the rule layer is the only floor, so add the noun at critical. DELIBERATELY only the
        # noun: exactly the tolerance boundary already drawn for "suffocation" (added) vs "suffocated"/
        # "suffocating" (excluded — heavy figurative usage) — the participle "strangled" and gerund
        # "strangling" carry benign figurative meaning ("a strangled cry", "the merger strangled
        # competition", "strangling the budget"), all live-verified LOW, so they would over-fire, not
        # fix a miss. \bstrangulation\b matches only the literal noun, so it closes the miss with zero
        # false-positive risk.
        # "cardiac tamponade" floors at critical, and "tension pneumothorax" is its direct respiratory
        # sibling — a one-way-valve lung collapse building pressure that compresses the heart and great
        # vessels into obstructive shock, an immediately lethal emergency needing emergency needle/chest
        # decompression, the term an EMS/ED report names directly ("tension pneumothorax on scene,
        # needle decompression performed", "developed a tension pneumothorax after the chest trauma") —
        # yet it matched nothing and dropped to LOW: the SAME word-choice asymmetry class as cardiac-
        # tamponade / ventricular-fibrillation; an immediately-fatal event scored critical-or-LOW purely
        # on which clinical phrase the reporter chose. It is a whole multi-word clinical phrase with ZERO
        # benign English meaning, so it closes the miss at the tamponade floor with no new over-fire
        # class. DELIBERATELY NOT the bare word "pneumothorax": a small spontaneous/simple pneumothorax
        # can be stable and merely monitored — a genuine severity judgment, not a clean miss — so the
        # bare token would over-fire on a routine chest note; it stays Ben-review. \btension pneumothorax\b
        # matches only the lethal qualified form, so it closes the miss with zero false-positive risk.
        # "cardiac arrest" floors at critical, and "asystole" is the flatline rhythm that IS a pulseless
        # cardiac arrest — zero cardiac electrical activity, the immediately-fatal non-shockable end of
        # the same event "ventricular fibrillation" (already critical) begins, the term an AED/monitor/
        # EMS report writes directly ("monitor showed asystole", "patient in asystole; CPR continued",
        # "asystole confirmed on the rhythm strip") — yet it matched nothing and dropped to LOW: the
        # SAME word-choice asymmetry class as heart-attack/myocardial-infarction and ventricular-
        # fibrillation; the same immediately-fatal cardiac event scored critical-or-LOW purely on which
        # clinical term the reporter chose. It is a whole clinical word with ZERO benign English meaning,
        # so it closes the miss at the cardiac-arrest floor with no new over-fire class. \basystole\b is
        # a distinct token from the routine cardiac-cycle words "systole"/"diastole"/"systolic" (which
        # appear in normal blood-pressure notes — "systole 120, diastole 80" — and must NOT fire); the
        # whole-word matcher means adding asystole does not touch those, so it closes the miss with zero
        # false-positive risk.
        # "septic shock" floors at critical, but its immediately-lethal shock SIBLINGS — "cardiogenic
        # shock" (the heart can no longer pump enough blood, the pump-failure shock that follows the
        # already-critical MI / cardiac arrest, ~50% mortality) and "hypovolemic shock" (circulatory
        # collapse from massive blood/fluid loss, the shock state of the already-critical exsanguination
        # / severe-bleeding pathway) — matched nothing and dropped to LOW: the SAME word-choice
        # asymmetry class as heart-attack/myocardial-infarction and septic shock itself, the same
        # immediately-life-threatening shock state scored critical-or-LOW purely on which qualifier the
        # reporter chose ("EMS reports the patient is in cardiogenic shock", "hypovolemic shock on
        # arrival; rapid transfusion begun"). Both are whole two-word clinical phrases with ZERO benign
        # English meaning, so they close the miss at the septic-shock floor with no new over-fire class.
        # DELIBERATELY the QUALIFIED two-word phrases only — never the bare word "shock", which is
        # heavily polysemous (emotional "in shock", "electric shock" which already floors at electrical
        # HIGH, "shock absorber", "shock of the near miss"); the phrase matcher means \bcardiogenic
        # shock\b / \bhypovolemic shock\b cannot fire from any of those, so they close the miss with
        # zero false-positive risk.
        # "hypovolemic shock" floors at critical, and "hemorrhagic shock" is its direct clinical
        # synonym for the blood-loss case — circulatory collapse specifically from massive hemorrhage,
        # the shock endpoint of the already-critical exsanguination / severe-bleeding pathway, the term
        # an EMS/trauma report writes directly ("in hemorrhagic shock from the leg wound", "class IV
        # hemorrhagic shock, massive transfusion protocol activated"). Today the phrase fires only HIGH
        # via the bare "hemorrhagic" bleeding adjective below — the SAME under-scored-synonym class as
        # myocardial-infarction (the clinical twin of the already-critical heart attack) and
        # cardiorespiratory arrest: the same immediately-life-threatening shock state scored HIGH-not-
        # critical purely on which qualifier the reporter chose, when its physiologic twin "hypovolemic
        # shock" is critical. It is not a substring of "hypovolemic shock" (different first word), so
        # the escalation is a clean miss. A whole two-word clinical phrase with ZERO benign English
        # meaning, so it closes the miss at the shock-sibling floor with no new over-fire class.
        # DELIBERATELY the QUALIFIED phrase only — never the bare "shock" (polysemous, excluded above)
        # nor the bare "hemorrhagic" (which stays at its conservative HIGH bleeding floor below);
        # \bhemorrhagic\s+shock\b cannot fire from either, so it closes the miss with zero false-
        # positive risk.
        # "status epilepticus" — a continuous or back-to-back seizure that does not stop on its own
        # (>5 min / no recovery of consciousness between fits), a true neurological emergency that
        # causes hypoxic brain injury and death if not aborted, the term an EMS/ED report names
        # directly ("patient in status epilepticus, benzodiazepines given", "convulsive status
        # epilepticus on arrival") — yet it matched nothing and dropped to LOW: the SAME word-choice
        # asymmetry class as the cardiogenic-shock / asystole fixes, and specifically the critical
        # escalation of the already-HIGH convulsion floor below ("convulsing"/"convulsions" sit at
        # HIGH; their non-stopping, life-threatening form belongs at critical). It is a whole two-word
        # clinical phrase with ZERO benign English meaning. DELIBERATELY the QUALIFIED phrase only —
        # never the bare word "status" (massively polysemous: "status update", "status report",
        # "status meeting", "on-call status") nor the bare "seizure" (already excluded above for its
        # "asset seizure" polysemy); the phrase matcher means \bstatus epilepticus\b cannot fire from
        # any routine "status" note, so it closes the miss at zero false-positive risk.
        # "aortic dissection" — a tear in the aortic wall letting blood split the layers apart, an
        # immediately life-threatening arterial catastrophe (Type A ~1-2% mortality PER HOUR
        # untreated) the term an EMS/ED/CT report names directly ("acute aortic dissection on CT",
        # "Type A aortic dissection, to the OR emergently") — yet it matched nothing and dropped to
        # LOW: the SAME word-choice asymmetry class as the tension-pneumothorax / cardiac-tamponade
        # fixes, an immediately-fatal vascular event scored critical-or-LOW purely on which clinical
        # phrase the reporter chose. It is a whole two-word clinical phrase with ZERO benign English
        # meaning, and it is a DISTINCT pathology from the already-critical "aneurysm" (a bulge, not a
        # tear — "aortic aneurysm" fires today, "aortic dissection" does not), so it closes the miss
        # at the aneurysm/tamponade floor with no new over-fire class. DELIBERATELY the QUALIFIED
        # phrase only — never the bare word "dissection", which is heavily polysemous (the routine
        # surgical/anatomical sense: "careful surgical dissection of the tissue plane", "the frog
        # dissection in the lab", and the figurative "a dissection of the argument"), all live-verified
        # LOW; the phrase matcher means \baortic dissection\b cannot fire from any of those, so it
        # closes the miss with zero false-positive risk.
        # "cardiopulmonary arrest" floors at critical, but its British/international synonym
        # "cardiorespiratory arrest" — the exact same immediately-fatal event (heart and breathing
        # both stopped), just the term a UK/Commonwealth EMS/ED report writes ("patient in
        # cardiorespiratory arrest", "cardiorespiratory arrest on arrival, CPR commenced") — matched
        # nothing and dropped to LOW: the SAME orthographic/word-choice asymmetry class already fixed
        # for the en-GB hospitalised/haemorrhage spellings, here on the combined-arrest phrasing. It
        # is not a substring of "cardiopulmonary arrest" (different middle word) nor of the bare
        # "arrest" (deliberately excluded for its police-"arrested" polysemy), so the same fatal
        # arrest scored critical-or-LOW purely on which clinical tradition the reporter learned. It
        # is a whole two-word clinical phrase with ZERO benign English meaning, so it closes the miss
        # at the cardiopulmonary-arrest floor with no new over-fire class. DELIBERATELY the qualified
        # two-word phrase only — never the bare adjective "cardiorespiratory" (a routine monitoring
        # word: "cardiorespiratory monitor", "cardiorespiratory fitness test", "cardiorespiratory
        # exam normal", all benign); the phrase matcher means \bcardiorespiratory arrest\b cannot
        # fire from any of those, so it closes the miss with zero false-positive risk.
        # "aneurysm" floors at critical, but the clinical name for the event a RUPTURED cerebral
        # aneurysm actually IS — "subarachnoid hemorrhage" (~85% of spontaneous SAH is a berry-
        # aneurysm rupture, ~50% mortality, the hyperacute catastrophe a CT/ED report names directly:
        # "acute subarachnoid hemorrhage on CT, to the OR emergently", "Hunt-Hess IV subarachnoid
        # haemorrhage") — fires only HIGH today, via the bare "hemorrhage"/"haemorrhage" bleeding term
        # below: the SAME under-scored-synonym class as hemorrhagic shock and myocardial infarction,
        # the same immediately-fatal ruptured-aneurysm event scored HIGH-not-critical purely on whether
        # the reporter wrote "aneurysm" or its clinical result "subarachnoid hemorrhage". It is not a
        # substring of "aneurysm" (different words entirely), so the escalation is a clean miss. Both
        # US and British spellings are needed (as with hemorrhage/haemorrhage), and each is a whole
        # two-word clinical phrase with ZERO benign English meaning. DELIBERATELY the QUALIFIED
        # "subarachnoid" phrase only — never the bare "hemorrhage"/"haemorrhage" (which stay at their
        # conservative HIGH bleeding floor below), and NOT the umbrella "intracranial hemorrhage"
        # (which can name a slow chronic subdural, not always a hyperacute emergency); \bsubarachnoid
        # h(a)?emorrhage\b cannot fire from either, so it closes the miss at the aneurysm floor with no
        # new over-fire class. Surfaced in the 2026-08-21 rule-probe backlog (vascular-catastrophe
        # sibling; next after aortic dissection).
        "critical": ["fatality", "fatalities", "death", "died", "deceased", "casualty",
                     "casualties", "unconscious", "lost consciousness", "loss of consciousness",
                     "cardiac arrest", "heart attack", "myocardial infarction",
                     "ventricular fibrillation", "asystole",
                     "cardiac tamponade", "pericardial tamponade",
                     "tension pneumothorax",
                     "cpr", "no pulse", "no heartbeat",
                     "pulseless", "anaphylaxis", "aneurysm", "aneurism", "embolism",
                     "anaphylactic", "not breathing", "stopped breathing", "no longer breathing",
                     "isn't breathing", "wasn't breathing", "severe bleeding",
                     "exsanguination", "exsanguinated",
                     "respiratory arrest", "cardiopulmonary arrest",
                     "cardiorespiratory arrest", "septic shock",
                     "cardiogenic shock", "hypovolemic shock", "hemorrhagic shock",
                     "status epilepticus", "aortic dissection",
                     "subarachnoid hemorrhage", "subarachnoid haemorrhage",
                     "amputation", "amputated", "decapitation", "decapitated",
                     "dismemberment", "dismembered",
                     "evisceration", "eviscerated", "strangulation",
                     "life-threatening", "multiple injured"],
        # "heart attack" floors at critical, but its clinical twin "myocardial infarction" — the
        # term an EMS/medical report actually uses ("suspected myocardial infarction", "acute
        # myocardial infarction confirmed") — matched nothing and dropped to LOW: the SAME
        # word-choice asymmetry class already fixed for heart-attack/cardiac-arrest, hemorrhage,
        # exsanguination and scald; the same life-threatening cardiac event scored critical-or-LOW
        # purely on which synonym the reporter chose. It is a whole multi-word clinical phrase with
        # ZERO benign English meaning, so it closes the miss at the heart-attack floor with no new
        # over-fire class. DELIBERATELY NOT the bare acronym "MI" (massively polysemous — Michigan,
        # "mi", mile) nor the polysemous "flatline"/"flatlined" ("sales flatlined", "the economy
        # flatlined" — both live-verified LOW, would over-fire); those stay Ben-review.
        # "cardiac arrest" floors at critical, and "ventricular fibrillation" is the lethal shockable
        # rhythm that IS a pulseless cardiac arrest — the term an AED/EMS/monitor report actually
        # writes ("patient in ventricular fibrillation", "confirmed ventricular fibrillation on the
        # monitor") — yet it matched nothing and dropped to LOW: the SAME word-choice asymmetry class
        # as heart-attack/myocardial-infarction; the same immediately-fatal cardiac event scored
        # critical-or-LOW purely on which term the reporter chose. It is a whole multi-word clinical
        # phrase with ZERO benign English meaning, so it closes the miss at the cardiac-arrest floor
        # with no new over-fire class. DELIBERATELY NOT the bare acronym "v-fib" (needs the spelled-
        # out form; acronym polysemy) nor "ventricular tachycardia" (VT can be stable/pulsed — a
        # genuine severity judgment, not a clean miss); both stay Ben-review.
        # "cardiac arrest" floors at critical, and "cardiac tamponade" / "pericardial tamponade" —
        # compression of the heart by pericardial fluid/blood under pressure, an immediately
        # life-threatening emergency an EMS/ED/echo report names directly ("developed cardiac
        # tamponade after the chest trauma", "confirmed pericardial tamponade on the echo") — matched
        # nothing and dropped to LOW: the SAME word-choice asymmetry class as heart-attack/myocardial-
        # infarction/ventricular-fibrillation; a critical cardiac event scored critical-or-LOW purely
        # on which clinical phrase the reporter chose. Both are whole multi-word clinical phrases with
        # ZERO benign English meaning, so they close the miss at the cardiac floor with no new
        # over-fire class ("pericardial tamponade" is a separate entry — \bcardiac tamponade\b does
        # not match it). DELIBERATELY NOT the bare word "tamponade": inside medicine it is polysemous
        # — a therapeutic maneuver to stop bleeding ("balloon tamponade of the varix", "uterine
        # tamponade", "nasal tamponade") — so the bare token would over-fire on a treatment note; it
        # stays Ben-review.
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
        # "frostbite"/"frostbitten" is the direct cold-exposure sibling of hypothermia (already HIGH):
        # an acute injury a reporter names outright ("severe frostbite on both hands", "the technician
        # was frostbitten in the freezer"), yet both forms matched nothing and dropped to LOW — the
        # SAME whole-clinical-word absent-term miss class as hypothermia/aneurysm above. Both the noun
        # and the participle are needed (\bfrostbite\b does not match "frostbitten"), each is a whole
        # word with NO benign English meaning, so they close the miss at zero false-positive risk;
        # added at the same conservative HIGH floor as hypothermia. Surfaced in the 2026-08-17 probe.
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
        # "degloved"/"degloving" is a severe avulsion trauma — the skin and soft tissue torn from a
        # limb, the way a machinery report actually names it ("his hand was degloved in the roller",
        # "degloving injury to the forearm") — yet BOTH forms matched nothing and dropped to LOW: the
        # SAME whole-clinical-word absent-term miss class as impaled/frostbite/hypothermia above (the
        # generic word "injury" floored "degloving injury" at HIGH, but "was degloved" alone scored
        # LOW). "deglove"/"degloved"/"degloving" has NO benign English meaning — it is exclusively
        # this avulsion injury — so it closes the miss with zero false-positive risk. Added at the
        # conservative HIGH floor (a survivable-but-serious trauma; the LLM or a human can raise a
        # specific case to critical), the participle a separate entry because \bdegloved\b does not
        # match "degloving". Surfaced in the 2026-08-17 rule-probe backlog.
        # "scald"/"scalded" is the thermal-burn sibling of "burn"/"burns" (already HIGH): a burn from
        # hot liquid or steam, an acute injury a reporter names outright ("scalded by steam", "the
        # worker was scalded", "a scald from the hot water line"), yet both forms matched nothing and
        # dropped to LOW while "burn"/"burns" score HIGH — the SAME word-choice asymmetry as
        # bleeding/hemorrhage, the same thermal injury scored HIGH-or-LOW purely on which word the
        # reporter chose. Both the noun/verb "scald" and the participle "scalded" are needed
        # (\bscald\b does not match "scalded"), each a whole word for a thermal injury. Added at the
        # same HIGH floor as "burn". DELIBERATELY EXCLUDES the participle-adjective "scalding" —
        # unlike "scald"/"scalded" it is polysemous (the harsh-criticism metaphor "scalding review",
        # "scalding rebuke", and the intensifier "scalding hot"), so adding it would over-fire on
        # non-incident text; whole-word matching keeps \bscald\b/\bscalded\b from touching "scalding".
        # Surfaced in the 2026-08-17 rule-probe backlog.
        # "bleeding" sits at HIGH, but its direct clinical synonym "hemorrhage"/"hemorrhaging" —
        # the word a report actually uses for profuse blood loss ("worker is hemorrhaging", "massive
        # hemorrhage") — matched nothing and dropped to LOW: the SAME word-choice asymmetry class as
        # heart-attack/flames/electric-shock/amputated, the same injury scored HIGH-or-LOW purely on
        # which synonym the reporter chose. Added at the same HIGH floor as "bleeding" (whose money
        # metaphor "bleeding cash" already fires there, so this introduces no new over-fire class).
        # Both the noun and the participle are needed (\bhemorrhage\b does not match "hemorrhaging"),
        # and whole-word matching keeps the benign prefix-sharer "hemorrhoid"/"hemorrhoids" — not an
        # acute emergency — from firing.
        # Surfaced in the 2026-08-17 rule-probe backlog (next-pick after stab wound).
        # "hemorrhage"/"hemorrhaging" sit at HIGH, but their ADJECTIVE form "hemorrhagic" — the word a
        # trauma/clinical report actually writes ("hemorrhagic stroke suspected", "hemorrhagic fever
        # outbreak") — matched nothing and dropped to LOW: the SAME tokenization miss already fixed for
        # burns/electric-shocks/hemorrhaging (\bhemorrhage\b does not match "hemorrhagic"), the same
        # profuse-bleeding emergency scored HIGH-or-LOW purely on the grammatical form. "hemorrhagic"
        # has ZERO benign English meaning — it exclusively denotes bleeding — so it closes the miss at
        # the same HIGH floor with no new over-fire class; every phrase it appears in (hemorrhagic
        # stroke/fever) is itself a bleeding emergency. Both the US and British spellings are needed
        # (as with hemorrhage/haemorrhage), and whole-word matching still keeps the prefix-sharer
        # "hemorrhoid" from firing. Surfaced in the 2026-08-20 rule-probe backlog (adjectival sibling;
        # next after the hemorrhage/hemorrhaging pair). NOTE: the one phrase where this adjective names
        # an immediately-lethal SHOCK state — "hemorrhagic shock" — is escalated to CRITICAL up in the
        # shock-sibling block above (beside hypovolemic/cardiogenic shock); the bare adjective and its
        # non-shock phrases stay at this HIGH floor.
        # "bleeding"/"hemorrhage" sit at HIGH, but the plain-English phrase a report actually uses for
        # the same emergency — "blood loss" ("severe blood loss", "massive blood loss", "the worker
        # suffered significant blood loss") — matched nothing and dropped to LOW: the SAME word-choice
        # asymmetry class as bleeding/hemorrhage, the same injury scored HIGH-or-LOW purely on which
        # synonym the reporter chose. Added the whole two-word clinical phrase at the same HIGH floor;
        # it has near-zero benign meaning and, matched as a phrase, cannot fire from the bare token
        # "blood" (blood drive/blood pressure/bloodline) — no new over-fire class.
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
        # A "punctured lung" (traumatic pneumothorax/hemothorax) is a serious acute chest trauma a
        # reporter names directly ("the worker suffered a punctured lung", "punctured lung from a
        # broken rib"), yet an ISOLATED report of it matched nothing and dropped to LOW — the phrasings
        # that scored above only fired off an incidental "fall"/"fracture", not the injury itself.
        # Added as the MULTI-WORD phrase "punctured lung" at the same conservative HIGH floor as
        # impaled/blood loss (the LLM or a human can raise a specific case; serious-but-survivable so
        # not critical). Deliberately NOT bare "punctured" — that is polysemous ("punctured tire",
        # "punctured the drywall") and would over-fire; the two-word phrase cannot fire from it (both
        # live-verified LOW). Surfaced in the 2026-08-18 rule-probe.
        # Sepsis - the body's life-threatening response to infection - is an acute medical emergency a
        # reporter names directly ("the patient is in sepsis", "worker went septic after the wound
        # infected"), yet "sepsis" matched nothing and dropped to LOW, and its terminal form "septic
        # shock" (circulatory collapse, ~30-40% mortality) also matched nothing: the SAME
        # whole-clinical-word absent-term miss class as aneurysm/embolism/anaphylaxis above. Added
        # "sepsis" at the conservative HIGH floor (serious but develops over hours and is treatable -
        # the LLM or a human can raise a specific case) and "septic shock" at CRITICAL alongside
        # "anaphylactic"/"respiratory arrest" (a named shock state = imminent circulatory collapse, the
        # same critical tier as the other shock/arrest phrases). DELIBERATELY EXCLUDES the bare
        # adjective "septic" - it is polysemous ("septic tank", "septic system", "the septic line
        # backed up") and would over-fire on routine plumbing/facilities text; whole-word matching
        # keeps \bsepsis\b and the two-word \bseptic shock\b from ever firing off bare "septic".
        # Surfaced in the 2026-08-18 rule-probe.
        # A near-drowning (a nonfatal submersion event) is an acute emergency a reporter names
        # directly ("child pulled from the pool in a near-drowning", "near drowning at the facility
        # pool"), yet both the spaced and hyphenated forms matched nothing and dropped to LOW. The
        # water/flood floor already covers "submerged" at critical, but a rescue reported by the
        # clinical phrase alone (victim already out of the water) fired nothing. DELIBERATELY EXCLUDES
        # bare "drowning"/"drowned" - "drowning" is polysemous ("drowning in debt/paperwork") and
        # "drowned" collides with the idiom "drowned out"; the two-word/hyphenated "near drowning"/
        # "near-drowning" has NO benign meaning, so it closes the miss with zero false-positive risk
        # (both decoys live-verified LOW). Added at the conservative HIGH floor (serious but the
        # "near" implies survived/rescued - the LLM or a human can raise a specific case to critical,
        # and an active submersion still floors critical via water/flood). Surfaced in the 2026-08-18
        # rule-probe backlog as the safest of the water-emergency candidates.
        # A "puncture wound" is a penetrating trauma a reporter names directly ("the victim has a
        # puncture wound to the chest", "multiple puncture wounds from the nail gun", "deep puncture
        # wound to the abdomen"), yet an ISOLATED report of it matched nothing and dropped to LOW,
        # while its weapon-implying siblings "stab wound"/"gunshot wound" already floor CRITICAL. It
        # is the penetrating-trauma twin of "punctured lung" above and the medical-descriptor cousin
        # of stab/gunshot wound. Added as the MULTI-WORD phrase "puncture wound"/"puncture wounds" at
        # the same conservative HIGH floor (NOT critical) as punctured lung/impaled — unlike a
        # stab/gunshot wound it carries no weapon connotation and is routinely minor (nail, needle,
        # animal bite), so HIGH is the defensible floor; the LLM or a human can raise a specific case.
        # Deliberately NOT bare "puncture" — that is polysemous ("punctured tire", "puncture in the
        # fuel line", "puncture-resistant gloves") and would over-fire; the two-word phrase cannot
        # fire from it (all decoys live-verified LOW). Surfaced in the 2026-08-18 rule-probe backlog.
        # The participle "impaled" sits at HIGH, but the NOUN form "impalement" ("suffered an
        # impalement", "an impalement injury on the rebar") matched nothing (\bimpaled\b does not
        # match "impalement") and dropped to LOW: the SAME participle/verb-vs-noun word-form gap
        # already fixed for decapitation/decapitated, amputation/amputated, and concussion/concussed —
        # here in reverse, the participle present and the noun absent — the same penetrating trauma
        # scored HIGH-or-LOW purely on grammatical form. Added at the participle's existing HIGH floor
        # (NOT a new severity call). "impalement" is a whole word with NO benign English meaning, so it
        # closes the miss with zero false-positive risk. Surfaced in the 2026-08-18 rule-probe backlog.
        "high":     ["injury", "injured", "injuries", "hospitalized", "hospitalised", "ambulance",
                     "punctured lung", "puncture wound", "puncture wounds",
                     "sepsis", "near drowning", "near-drowning",
                     "broken bone", "broken bones",
                     "fracture", "fractured", "fractures",
                     "concussion", "concussed", "burn", "burned", "burns", "impaled", "impalement",
                     "scald", "scalded",
                     "degloved", "degloving",
                     "hypothermia", "hypothermic", "frostbite", "frostbitten",
                     "heat stroke", "heatstroke", "hyperthermia", "hyperthermic",
                     "heat exhaustion",
                     "overdose", "overdosed",
                     "collapsed", "bleeding", "hemorrhage", "hemorrhaging",
                     "haemorrhage", "haemorrhaging", "blood loss",
                     "hemorrhagic", "haemorrhagic",
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
        # The whole explosion word-family floors at critical, but its direct synonym "detonation" and
        # its verb forms "detonate"/"detonated"/"detonating" — how a blast/bomb report is actually
        # written ("the device detonated near the entrance", "a car bomb detonated in the structure",
        # "a detonation was heard on the third floor") — matched nothing (\bexplosion\b/\bexploded\b
        # match none of the "deton-" forms) and dropped to LOW: the SAME synonym + verb-vs-noun
        # word-form gap already fixed for explosion/exploded/exploding, the same catastrophic blast
        # scored critical-or-LOW purely on which word the reporter chose. Added at the noun's critical
        # floor. Every "deton-" form denotes EXCLUSIVELY an explosion — there is NO benign English
        # meaning (unlike the growth-metaphor "exploded" already accepted), so this closes the miss
        # with zero false-positive risk. Each form is a separate entry because the whole-word matcher
        # won't cross the suffix boundary (\bdetonate\b does not match "detonated"/"detonation"/
        # "detonating"); the noun "detonation" also subsumes "car bomb"/"pipe bomb ... detonation".
        # DELIBERATELY EXCLUDES the device noun "detonator" — mere presence of a detonator (bomb squad
        # inventory, a demolition-supplies audit) is not itself a blast, the same discipline that added
        # discharge/active-threat terms but excluded the bare "firearm"; whole-word matching keeps the
        # "deton-" verb/noun forms from ever firing off "detonator". Surfaced in the 2026-08-19
        # rule-probe (twin of the accepted explosion/exploded family).
        # "arson"/"arsonist" — an intentionally-set fire named by its crime — was absent from the
        # whole fire word-family, so a report that names the act rather than the flame ("suspected
        # arson at the vacant warehouse overnight", "arsonist set the dumpster alight") matched
        # nothing and dropped to LOW while the plain word "fire" floors at critical: the SAME
        # word-choice miss class as molotov/detonation, an unambiguous fire event scored
        # critical-or-LOW purely on whether the reporter wrote the flame or the crime. Placed at
        # fire/smoke critical (not security) because the physical hazard and the offline next-step
        # actions — evacuate, confirm the fire is out, notify the fire department, preserve the scene
        # for cause investigation — are the fire/smoke ones, the same home as "structure fire"/
        # "wildfire". Both "arson" (the act) and "arsonist" (the actor) are added: neither is a
        # substring of the other under whole-word matching (\barson\b does not match "arsonist"), the
        # same discipline applied to decapitation/decapitated. Each is a whole word denoting ONLY
        # deliberate fire-setting — NO benign English meaning (unlike the polysemous neighbors left
        # out elsewhere) — so this closes the miss with zero false-positive risk; whole-word matching
        # keeps \barson\b from firing inside the benign "parson" (the exact armed/unarmed guard).
        # Surfaced in the 2026-08-19 rule-probe (sibling of the molotov incendiary-attack fix).
        "critical": ["fire", "flames", "ablaze", "blaze", "explosion", "exploded", "explosions",
                     "exploding", "explosive",
                     "detonation", "detonate", "detonated", "detonating",
                     "arson", "arsonist",
                     "engulfed", "structure fire", "wildfire", "conflagration"],
        "high":     ["smoke", "smoldering", "scorch", "charred", "burning smell",
                     "fire alarm", "sparks"],
        "medium":   ["overheating", "hot to the touch", "burnt smell"],
    },
    "water/flood": {
        # "flood"/"flooding"/"flooded" all floor at critical, but the noun "floodwater"/"floodwaters"
        # — the way an active inundation is actually reported ("floodwaters rose to the second floor",
        # "rising floodwaters trapped staff") — is a distinct whole-word token that \bflood\b does not
        # match (no boundary before "water"), so it dropped to LOW: the SAME singular/compound-word
        # tokenization gap already fixed for burns/injuries/fractures and the weather plurals. Both
        # forms are whole words with NO benign English meaning, so zero over-fire risk; the plural
        # "floodwaters" needs its own entry because \bfloodwater\b won't match it.
        "critical": ["flood", "flooding", "flooded", "floodwater", "floodwaters",
                     "submerged", "sewage backup", "burst main", "dam failure"],
        "high":     ["water damage", "burst pipe", "pipe burst", "leak", "leaking",
                     "standing water", "ceiling collapse from water", "overflow"],
        "medium":   ["drip", "dripping", "damp", "moisture", "condensation", "minor leak"],
    },
    "electrical/power": {
        # The NOUN "electrocution" floors at critical, but its verb/participle forms — how an acute
        # report is actually written ("a lineman was electrocuted", "the arc will electrocute anyone
        # who touches it", "workers electrocuting themselves on the exposed bus") — scored LOWER: the
        # participle "electrocuted" caught only the injury/medical HIGH floor (a burn-family neighbor),
        # and "electrocute"/"electrocutes"/"electrocuting" matched nothing and dropped to LOW. The SAME
        # lethal event — death or grave injury by electric current — was scored critical-or-lower purely
        # on grammatical form, the identical verb-vs-noun word-form gap already fixed for
        # explosion/exploded, asphyxiation/asphyxiated, and collapse/collapsing. Added the whole verb
        # family at the noun's critical floor beside "electrocution". By definition "electrocute" means
        # to kill/severely injure by electric shock — the forms carry essentially NO benign or
        # figurative meaning (unlike the deliberately-HIGH "electric shock", which can be minor or the
        # idiom "the news was an electric shock"), so this closes the miss with zero over-fire risk and
        # introduces no new class the accepted noun "electrocution" doesn't already carry. Each derived
        # form needs its own entry — \belectrocution\b matches none of them. Surfaced in the 2026-08-19
        # rule-probe (verb-form twin of the electrocution critical floor).
        "critical": ["live wire", "arc flash", "electrocution",
                     "electrocuted", "electrocute", "electrocutes", "electrocuting",
                     "electrical fire"],
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
        # The clinical noun "asphyxiation" floors at critical, but its universal LAY synonym for the SAME
        # event — "suffocation" ("suffocation reported at the plant", "death by suffocation") — matched
        # nothing and dropped to LOW: the SAME word-choice asymmetry class as asphyxiation/heart-attack,
        # the same life-threatening oxygen-deprivation event scored critical-or-LOW purely on whether the
        # reporter reached for the clinical or the plain word. Added the NOUN at the critical floor beside
        # asphyxiation. DELIBERATELY only the noun: unlike "asphyxiated"/"asphyxiating" (zero benign
        # meaning), the verb/adjective forms "suffocated"/"suffocating" carry heavy figurative usage
        # ("suffocated by the workload", "suffocating heat", "the suffocating bureaucracy") and would
        # over-fire — the same polysemy discipline that added "scald"/"scalded" but excluded the
        # metaphor-heavy "scalding". \bsuffocation\b matches only the literal noun, so this closes the
        # miss with zero false-positive risk.
        # The derived forms "asphyxiation"/"asphyxiated"/"asphyxiating" all floor at critical, but the
        # bare clinical ROOT noun "asphyxia" — the exact word a medical/coroner report uses ("traumatic
        # asphyxia", "positional asphyxia", "asphyxia due to the confined space") — matched nothing and
        # dropped to LOW, because \basphyxiation\b does NOT match "asphyxia" (no word boundary before the
        # "-tion" suffix) and neither do the other forms: the SAME whole-clinical-word absent-term miss
        # class as decapitation/aneurysm. "asphyxia" is a whole clinical word with essentially zero
        # benign English meaning — even less figurative than the accepted "asphyxiating" — so it closes
        # the miss with no new over-fire class the accepted "asphyxiation" doesn't already carry.
        # Surfaced in the 2026-08-18 rule-probe.
        # "hydrogen sulfide" is a lethal confined-space/sour-gas toxicant (H2S) — the classic
        # "rotten-egg gas" that kills wastewater, oil-and-gas and vault workers. It had ZERO
        # gas/chemical coverage: no bare-name entry AND no "hydrogen sulfide leak" phrase (only
        # "ammonia leak"/"chlorine leak" existed), so "hydrogen sulfide detected in the sump" and
        # even "a hydrogen sulfide leak" dropped to LOW/HIGH-off-"leak" while "carbon monoxide" (the
        # sibling always-a-hazard multi-word gas name) floors CRITICAL. Added the whole two-word
        # chemical name at critical beside "carbon monoxide"; like CO it has ZERO benign English
        # meaning → no new over-fire class. DELIBERATELY NOT the bare single words "chlorine"/
        # "ammonia" (routine pool-treatment / cleaning-product uses → polysemous, Ben-review) nor the
        # acronym "h2s" (short/ambiguous, same exclusion class as v-fib/MI). Surfaced 2026-08-20.
        "critical": ["gas leak", "carbon monoxide", "hydrogen sulfide", "toxic", "chemical spill",
                     "hazmat",
                     "hazardous material", "fumes", "asphyxia", "asphyxiation", "asphyxiated",
                     "asphyxiating",
                     "suffocation", "ammonia leak", "chlorine leak", "explosive gas"],
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
        # "stab wound"/"stab wounds" is the bladed-weapon sibling of "gunshot wound" — a directly-named
        # violent penetrating trauma. "gunshot wound" already reaches this critical floor via "gunshot",
        # but the knife-assault equivalent ("victim has a stab wound", "multiple stab wounds", "knife
        # attack, stab wounds") matched nothing and dropped to LOW: the SAME weapon-vs-weapon word-choice
        # asymmetry as flames/electric-shock/hemorrhage. Deliberately a MULTI-WORD adjacency phrase, NOT
        # the bare polysemous "stab"/"stabbed"/"stabbing" — those carry benign collisions ("stabbing
        # pain", "stabbed at the food", "took a stab at it") and are guarded LOW; only the whole phrase
        # "stab wound(s)" (zero benign meaning) fires here.
        # A "molotov" (Molotov cocktail) is a thrown incendiary weapon — an unambiguous violent
        # attack a reporter names directly ("a molotov cocktail was thrown through the window",
        # "protesters hurled molotovs at the guard shack"), yet it matched nothing and dropped to
        # LOW while its sibling violent-attack terms (bomb threat / gunshot / stab wound) already
        # floor at critical: the SAME weapon-word miss class as the stab-wound fix above. Bare
        # "molotov" is a whole word that also covers "molotov cocktail(s)" (\bmolotov\b matches
        # inside the phrase); the plural "molotovs" needs its own entry (\bmolotov\b does not match
        # it), the same singular->plural tokenization discipline applied to burns/injuries/typhoons.
        # The ONLY benign collision is the historical surname "Molotov" (the Soviet foreign
        # minister) — a proper noun essentially absent from operational facility/security incident
        # text, so this is the SAME negligible tolerance already accepted for "decapitated"/
        # "dismembered" (which likewise fire on their rare figurative use); a critical floor a
        # human/LLM can lower on the astronomically rare historical mention is far safer than a
        # missed incendiary attack.
        # The NOUN "kidnapping" floors at critical, but the participle/verb "kidnapped" — how an
        # abduction is actually reported ("a worker was kidnapped from the loading dock", "employee
        # kidnapped at the north gate") — and the PLURAL "kidnappings" ("two kidnappings reported in
        # the visitor lot") matched nothing (\bkidnapping\b matches neither "kidnapped" nor
        # "kidnappings") and dropped to LOW: the SAME verb-vs-noun + singular->plural word-form gap
        # already fixed for carjacking/carjacked/carjackings and molotov/molotovs, the same violent
        # crime scored critical-or-LOW purely on grammatical form. Added at the noun's critical floor.
        # Both "kidnapped" and "kidnappings" are whole words denoting EXCLUSIVELY the crime — NO benign
        # English meaning — so they close the miss with zero false-positive risk. DELIBERATELY EXCLUDES
        # the legal synonyms "abduction"/"abducted": unlike "kidnapped" they are polysemous — in an
        # injury/rehab context "abduction" is the anatomical range-of-motion term ("limited shoulder
        # abduction", "hip abduction exercises") a PT/ergonomics note routinely uses — so adding them
        # would over-fire CRITICAL on benign medical text, the SAME polysemy discipline that added the
        # unambiguous "typhoon"/"arson" but excluded the polysemous "cyclone"/"detonator"/"septic".
        # Surfaced in the 2026-08-19 rule-probe (word-form sibling of the carjacking fix).
        # A "pistol-whipping" (beating a victim with a firearm used as a bludgeon) is a directly-named
        # ARMED violent assault a reporter writes outright ("the guard was pistol-whipped during the
        # robbery", "suspect pistol whipped the cashier"), yet it matched nothing and dropped to LOW
        # while its sibling armed-violence terms (armed / weapon / gunshot / stab wound / molotov)
        # already floor at critical: the SAME weapon-word miss class as the stab-wound and molotov
        # fixes. It belongs at critical, not merely at the "assault" HIGH floor, precisely because an
        # armed offender is present — the same reason "armed"/"weapon" are already critical here (a
        # human/LLM can lower a specific case). The compound has ZERO benign English meaning, so it
        # closes the miss with no false-positive risk — bare "whipped" (whipped cream, "whipped the
        # team") is deliberately NOT added; only the two-word/hyphenated compound fires. Reports write
        # it both hyphenated and spaced, and as the past participle and the gerund/noun, so each needs
        # its own entry (\bpistol-whipped\b matches none of the others): "pistol-whipped"/"pistol
        # whipped" (the assault happened) + "pistol-whipping"/"pistol whipping" (the event as a noun),
        # the same multi-form tokenization discipline as molotov/molotovs and the hyphenated
        # "load-bearing". Surfaced in the 2026-08-19 structural/security-violence rule-probe. Left the
        # polysemous assault-family verbs (assaulted/brawl/fistfight — real figurative use: "political
        # brawl", "fistfight of ideas") for Ben-review, per the polysemous-token discipline.
        # The knife-assault ACT is only half-covered: "stab wound(s)" (the injury) floors critical,
        # but the way a reporter actually names the EVENT — "a stabbing attack in the cafeteria",
        # "a mass stabbing at the mall", "stabbing spree on campus", "knife attack in the lobby,
        # assailant fled", "held at knifepoint" — matched nothing and dropped to LOW, while the
        # firearm EVENT family (gunshot / shots fired / active shooting / shooter) is fully covered:
        # the SAME weapon-event miss class as the stab-wound / molotov / pistol-whipping fixes. These
        # belong at critical (an armed offender is present), the same reason "armed"/"weapon"/"active
        # shooter" already do. Each is a MULTI-WORD adjacency phrase or the unambiguous single word
        # "knifepoint" (zero benign meaning — the bladed sibling of "gunpoint") — DELIBERATELY still
        # excluding the bare polysemous "stab"/"stabbed"/"stabbing"/"stabbings" guarded LOW above
        # ("stabbing pain", "took a stab at it", "back-stabbing"); only whole violent-act phrases fire.
        # Surfaced in the 2026-08-21 weapon-event rule-probe.
        "critical": ["active shooter", "armed", "weapon", "hostage", "bomb threat",
                     "intruder armed", "kidnapping", "kidnapped", "kidnappings",
                     "gunshot", "gunshots", "gunfire",
                     "shots fired", "active shooting", "shooter", "shooting",
                     "stab wound", "stab wounds", "molotov", "molotovs",
                     "knife attack", "stabbing attack", "stabbing spree",
                     "mass stabbing", "knifepoint",
                     "pistol-whipped", "pistol whipped",
                     "pistol-whipping", "pistol whipping"],
        "high":     ["break-in", "broke in", "broken into", "intrusion", "intruder",
                     "unauthorized access", "forced entry", "trespass", "assault",
                     "data breach", "breach", "ransomware", "malware", "compromised account"],
        "medium":   ["suspicious person", "suspicious activity", "tailgating", "prowler",
                     "loitering", "phishing", "failed login", "unauthorized attempt"],
    },
    "theft": {
        # A "carjacking" (taking a vehicle from an occupant by force or threat) is a directly-named
        # violent robbery a reporter writes outright ("employee carjacked at gunpoint in the parking
        # garage", "a carjacking at the north entrance overnight"), yet it matched nothing and dropped
        # to LOW while its siblings "armed robbery"/"robbery at gunpoint" already floor at critical
        # here: the SAME weapon/force-theft class scored critical-or-LOW purely on which word the
        # reporter reached for, the identical miss class as the molotov / stab-wound violent-attack
        # fixes. Added at the theft critical floor beside "armed robbery" (a carjacking is armed
        # robbery of a vehicle; the LLM or a human can lower a specific case). "carjacking"/"carjacked"
        # denote EXCLUSIVELY the crime — NO benign English meaning (the bare token "car" is deliberately
        # NOT added: a car park / company car is routine), so this closes the miss with zero false-
        # positive risk. The verb "carjacked" and the plural "carjackings" each need their own entry —
        # \bcarjacking\b matches neither — the same verb-vs-noun + singular->plural tokenization
        # discipline already applied to molotov/molotovs and burn/burns. Surfaced in the 2026-08-19
        # rule-probe (violent-theft sibling of the armed-robbery floor).
        "critical": ["armed robbery", "robbery at gunpoint",
                     "carjacking", "carjacked", "carjackings"],
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
        # "hurricane" floors at critical, but its exact meteorological twin "typhoon" — the SAME event
        # (a tropical cyclone), just the regional name used in the Northwest Pacific — matched nothing
        # and dropped to LOW: the SAME event scored critical-or-LOW purely on which regional word the
        # reporter reached for, the identical class as the en-GB spelling fix (hospitalised/haemorrhage)
        # and the lay-synonym fixes (heart attack/cardiac arrest). Non-US/international incident reports
        # are routine (overseas facilities, imported PDF templates, contractors), so a Pacific-region
        # report writing "typhoon" instead of "hurricane" is exactly the miss the taxonomy exists to
        # close. Added at the same critical floor as "hurricane"; the plural "typhoons" needs its own
        # entry (\btyphoon\b won't match "typhoons"), the same singular→plural tokenization discipline
        # already applied to hurricanes/tornadoes/earthquakes. "typhoon" is a whole word with NO benign
        # English meaning, so it closes the miss with zero false-positive risk. DELIBERATELY EXCLUDES the
        # other tropical-cyclone synonym "cyclone" — it is polysemous ("cyclone fence" = chain-link
        # fence, "cyclone separator" = industrial dust collector) and would over-fire on routine
        # facilities text; only the unambiguous "typhoon"/"typhoons" are added. Surfaced in the
        # 2026-08-18 rule-probe.
        # The plural spellings of the remaining weather catastrophes were the last gap in this list's
        # singular→plural coverage: "wildfire"/"flash flood"/"tsunami" floor at critical, but the
        # plurals a reporter actually writes — "wildfires" ("three wildfires threatening the north
        # perimeter"), "flash floods" ("flash floods reported across the county"), "tsunamis"
        # ("tsunamis following the offshore quake") — are distinct tokens that \bwildfire\b /
        # \bflash\s+flood\b / \btsunami\b do NOT match, so they dropped to LOW: the SAME singular→plural
        # tokenization miss already closed for tornado(es)/hurricane(s)/typhoon(s)/earthquake(s) right
        # above, the same catastrophe scored critical-or-LOW purely on singular-vs-plural form. Added
        # each plural beside its singular. They are whole words with no benign polysemy, so this finishes
        # the list's plural coverage with zero false-positive risk. Surfaced in the 2026-08-19
        # weather-plural rule-probe.
        "critical": ["tornado", "tornadoes", "tornados", "hurricane", "hurricanes",
                     "typhoon", "typhoons",
                     "earthquake", "earthquakes",
                     "flash flood", "flash floods", "wildfire", "wildfires",
                     "tsunami", "tsunamis", "severe storm warning"],
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
