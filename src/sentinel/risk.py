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
        # The PRESENT participle "exsanguinating" — the way an active trauma report is actually written
        # ("patient is actively exsanguinating", "exsanguinating hemorrhage from the femoral wound",
        # "massive exsanguinating injury") — matched neither "exsanguination" nor "exsanguinated" (\b on
        # either side means the "-ating" form is a distinct token) and dropped to LOW/HIGH: the SAME
        # verb-form asymmetry already fixed for amputation→amputated / decapitation→decapitated, the same
        # immediately-fatal blood-loss event scored critical-or-LOW purely on grammatical tense. It is a
        # whole clinical word with NO benign English meaning, so it closes the miss at the noun's critical
        # floor with zero false-positive risk (the polysemous "bleed out" stays deliberately excluded).
        # "evisceration" — and its participle "eviscerated" ("the worker was eviscerated by the machine",
        # "traumatic abdominal evisceration at the press") — is a catastrophic trauma on par with
        # decapitation / dismemberment / amputation / impalement (all already critical here), yet BOTH the
        # noun and the participle matched nothing and dropped to LOW: the SAME whole-clinical-word absent-
        # term miss class as those trauma words. Offline the rule layer is the only floor, so add both at
        # critical beside dismemberment. In operational incident text "eviscerated"/"evisceration" denotes
        # only the physical trauma — the rare literary figurative ("eviscerated his argument") does not
        # appear in incident reports, the same tolerance already accepted for "decapitated"/"dismembered";
        # the participle is a separate entry because \bevisceration\b does not match "eviscerated".
        # The PRESENT participles "amputating" / "decapitating" / "dismembering" — the way an ACTIVE
        # machine-trauma report is written ("the press was amputating fingers on every cycle", "the
        # rotating blade was decapitating the worker who reached in", "the auger was dismembering the
        # worker who fell into it") — matched neither the noun nor the past participle (\b on the
        # "-ing" form is a distinct token) and dropped to LOW/MEDIUM: the SAME verb-form asymmetry
        # already closed for exsanguination→exsanguinating, the same catastrophic trauma scored
        # critical-or-LOW purely on grammatical tense. Each noun AND its past participle are ALREADY
        # critical here (amputation/amputated, decapitation/decapitated, dismemberment/dismembered), so
        # the present participle introduces NO new over-fire class the accepted forms don't already
        # carry — the rare literary figurative ("dismembering his argument") does not appear in
        # operational incident text, the same tolerance already accepted for "decapitated"/"dismembered".
        # "eviscerating" and "strangling" are DELIBERATELY excluded: both carry live figurative meaning
        # ("an eviscerating critique", "strangling the budget") the taxonomy already keeps out at the
        # noun/participle level. Each present participle needs its own entry — \bamputation\b/\bamputated\b
        # match none of them. Surfaced in the 2026-08-24 present-participle rule-probe (sibling of the
        # exsanguinating verb-form floor).
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
        # "tension pneumothorax" floors at critical, and hemothorax is its direct BLOOD twin — blood
        # instead of air filling the pleural space and collapsing the lung. A "massive hemothorax" (the
        # ATLS immediately-life-threatening chest injury: classically ≥1500 mL of blood or ongoing brisk
        # loss, driving hypovolemic shock and needing emergent tube thoracostomy / thoracotomy) and a
        # "tension hemothorax" (the pressurized form directly analogous to tension pneumothorax, shifting
        # the mediastinum into obstructive shock) are the terms an EMS/trauma/ED report writes directly
        # ("massive hemothorax on the left, chest tube drained 1.8 L", "developed a tension hemothorax
        # after the penetrating chest trauma") — yet they matched nothing and dropped to LOW: the SAME
        # word-choice asymmetry class as tension-pneumothorax / cardiac-tamponade; the same immediately-
        # fatal chest catastrophe scored critical-or-LOW purely on which clinical phrase the reporter
        # chose. Each is a whole two-word clinical phrase (British "haemothorax" spelling paired) with
        # ZERO benign English meaning, and none is a substring of any floored term (\bmassive hemothorax\b
        # cannot fire from "massive hemorrhage" — different second word — nor from "tension pneumothorax"),
        # so they close the miss at the tension-pneumothorax floor with no new over-fire class.
        # DELIBERATELY the QUALIFIED forms only — NEVER the bare "hemothorax"/"haemothorax": a small or
        # minimal hemothorax is routinely observed or managed with a single chest tube — a genuine
        # severity judgment, not a clean miss — exactly the tolerance already drawn for bare "pneumothorax"
        # (excluded) vs "tension pneumothorax" (floored) and bare "hemorrhage" (HIGH) vs "massive
        # hemorrhage" (critical); the bare token stays Ben-review. Surfaced in the 2026-08-26 chest-
        # catastrophe rule-probe (blood-twin sibling of the tension-pneumothorax / tamponade cluster).
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
        # "aortic rupture" / "ruptured aorta" is the frank blow-out of the aortic wall — the terminal,
        # even-more-immediately-fatal endpoint of the already-critical "aortic dissection" (a dissection
        # that tears fully through) an EMS/CT/trauma report names directly ("confirmed aortic rupture on
        # CT, massive hemothorax", "the patient suffered a ruptured aorta before EMS arrived") — yet BOTH
        # word orders matched nothing and dropped to LOW: the SAME word-choice asymmetry class as the
        # aortic-dissection / subarachnoid-hemorrhage vascular fixes, one aortic catastrophe critical and
        # its terminal twin LOW purely on which phrase the reporter chose. Both are whole multi-word
        # phrases with ZERO benign English meaning ("aorta"/"aortic" appears in no benign context and is
        # a substring of no other floored term — \baortic dissection\b cannot match either), so they close
        # the miss at the dissection floor with no new over-fire class. Next after subarachnoid hemorrhage
        # in the 2026-08-21 vascular-catastrophe backlog.
        # "subarachnoid hemorrhage" and the vascular catastrophes above floor at critical, but brain
        # HERNIATION — the terminal endpoint of raised intracranial pressure, where brain tissue is
        # forced across a dural fold and crushes the brainstem — matched nothing and dropped to LOW: the
        # SAME word-choice asymmetry class, the same immediately-fatal neuro-emergency scored
        # critical-or-LOW purely on which term the reporter chose. The qualified phrases a neuro/ICU/CT
        # report actually writes ("uncal herniation on CT", "signs of brain herniation", "transtentorial
        # herniation", "tonsillar herniation / coning") are whole clinical phrases with ZERO benign
        # English meaning, so they close the miss at the vascular-catastrophe floor with no new over-fire
        # class. DELIBERATELY the QUALIFIED brain phrases only — NEVER the bare "herniation" or "hernia",
        # which the whole-word matcher keeps benign ("disc herniation" in L5, an "inguinal hernia" repair
        # are routine, NOT critical — both live-verified LOW and STAY low). Surfaced in the 2026-08-24
        # intracranial-catastrophe rule-probe (sibling of subarachnoid hemorrhage / aortic rupture).
        # "respiratory arrest" / "not breathing" floor at critical, but "agonal" — the terminal
        # gasping respiration of a dying or actively-arresting patient ("agonal breathing", "agonal
        # respirations", "the patient is agonal"), a near-universal EMS/ED marker of imminent death
        # that IS effective apnea (ineffective, non-perfusing gasps) — matched nothing and dropped to
        # LOW: the SAME word-choice asymmetry class as respiratory-arrest / not-breathing; the same
        # immediately-fatal event scored critical-or-LOW purely on which term the reporter chose. The
        # bare word "agonal" has ZERO benign English meaning (it means "pertaining to the death
        # agony"), so it closes the miss at the respiratory-arrest floor with no new over-fire class —
        # and the whole-word matcher means it CANNOT fire from the words that merely contain the
        # substring "agonal" ("diagonal", "hexagonal", "octagonal", "pentagonal"; all live-verified
        # non-firing). Bare "agonal" covers the phrase forms ("agonal breathing/respirations/gasps")
        # in one term. Surfaced in the 2026-08-23 respiratory-terminal rule-probe (sibling of the
        # respiratory-arrest / apnea cluster).
        # "severe bleeding" floors at critical, but its clinical QUALIFIED synonyms — "massive
        # hemorrhage" / "catastrophic hemorrhage" / "uncontrolled hemorrhage" (and the British
        # "haemorrhage" spellings), the exact terms a trauma/EMS report writes for the SAME
        # immediately-fatal exsanguinating bleed ("massive hemorrhage, activated the massive
        # transfusion protocol", "catastrophic haemorrhage from the femoral wound", "uncontrolled
        # hemorrhage, could not stop it") — fired only HIGH, via the bare "hemorrhage"/"haemorrhage"
        # bleeding term below: the SAME word-choice asymmetry class as subarachnoid-hemorrhage and
        # hemorrhagic shock, the same life-threatening bleed scored critical-or-HIGH purely on whether
        # the reporter wrote the lay "severe bleeding" or its clinical equal. Each is a whole
        # two-word clinical phrase with ZERO benign English meaning that names the massive-transfusion /
        # <C>-catastrophic-hemorrhage life-threat, so they close the miss at the "severe bleeding" floor
        # with no new over-fire class. DELIBERATELY the QUALIFIED phrases only — never the bare
        # "hemorrhage"/"haemorrhage" (which stay at their conservative HIGH bleeding floor below), and
        # NOT the lay "massive/uncontrolled/catastrophic bleeding" (the "bleeding" token carries the
        # business metaphor "bleeding cash/talent" the taxonomy already tolerates only at HIGH — the
        # -rrhage matcher cannot fire from any of those). Live-verified: the metaphor "hemorrhaged our
        # budget" stays LOW (\bhemorrhage\b won't fire from "hemorrhaged"). Surfaced in the 2026-08-24
        # exsanguination-cluster rule-probe (sibling of severe-bleeding / exsanguination).
        "critical": ["fatality", "fatalities", "death", "died", "deceased", "casualty",
                     "casualties", "unconscious", "lost consciousness", "loss of consciousness",
                     "cardiac arrest", "heart attack", "myocardial infarction",
                     "ventricular fibrillation", "asystole", "asystolic",
                     "cardiac standstill",
                     "cardiac tamponade", "pericardial tamponade",
                     "cardiac rupture", "myocardial rupture", "ventricular rupture",
                     "tension pneumothorax",
                     "massive hemothorax", "massive haemothorax",
                     "tension hemothorax", "tension haemothorax",
                     "cpr", "no pulse", "no heartbeat",
                     "pulseless", "anaphylaxis", "aneurysm", "aneurism", "embolism",
                     "anaphylactic", "not breathing", "stopped breathing", "no longer breathing",
                     "isn't breathing", "wasn't breathing",
                     "airway obstruction", "obstructed airway",
                     "severe bleeding",
                     "massive hemorrhage", "massive haemorrhage",
                     "catastrophic hemorrhage", "catastrophic haemorrhage",
                     "uncontrolled hemorrhage", "uncontrolled haemorrhage",
                     "exsanguination", "exsanguinated", "exsanguinating",
                     "respiratory arrest", "cardiopulmonary arrest",
                     "agonal",
                     "cardiorespiratory arrest", "septic shock",
                     "cardiogenic shock", "hypovolemic shock", "hemorrhagic shock",
                     "status epilepticus", "aortic dissection",
                     "aortic rupture", "ruptured aorta",
                     "subarachnoid hemorrhage", "subarachnoid haemorrhage",
                     "brain herniation", "cerebral herniation", "uncal herniation",
                     "transtentorial herniation", "tonsillar herniation",
                     "amputation", "amputated", "amputating",
                     "decapitation", "decapitated", "decapitating",
                     "dismemberment", "dismembered", "dismembering",
                     "evisceration", "eviscerated", "strangulation",
                     "ischemic stroke", "ischaemic stroke",
                     "hemorrhagic stroke", "haemorrhagic stroke",
                     "acute stroke", "suspected stroke", "stroke victim",
                     "cerebrovascular accident", "cerebral infarction",
                     "life-threatening", "multiple injured"],
        # "heart attack" floors at critical, and a STROKE ("brain attack") is the same tier of acute,
        # time-critical emergency — every minute of delay loses brain tissue, so an active stroke is a
        # 911/thrombolysis event, not a routine report. Yet the lay/EMS phrasings a reporter actually
        # writes matched nothing and dropped to LOW: the SAME word-choice asymmetry class already fixed
        # for heart-attack/myocardial-infarction/ventricular-fibrillation — the same life-threatening
        # neurovascular event scored critical-or-LOW purely on which synonym the reporter chose. Added
        # only the QUALIFIED multi-word forms with ZERO benign English meaning: "ischemic/ischaemic
        # stroke" (the commonest type), "hemorrhagic/haemorrhagic stroke", "acute stroke", "suspected
        # stroke", "stroke victim", and the clinical umbrella "cerebrovascular accident" (CVA). This
        # mirrors the qualified-escalates / bare-and-umbrella-stays discipline: DELIBERATELY NOT the
        # bare word "stroke" (massively polysemous — brush/swim/key stroke, "stroke of luck/genius",
        # back/breaststroke, two-stroke engine; all live-verified LOW and left LOW), NOT the idiom-
        # substring forms "had a/having a/suffered a stroke" (\bhaving a stroke\b fires inside "having a
        # stroke of genius"; "suffered a stroke of bad luck" — both live-verified would over-fire), and
        # NOT the TRANSIENT/minor forms "TIA"/"transient ischemic attack"/"mini-stroke" (a TIA resolves
        # and is a genuine severity judgment, not a clean miss — same conservative reasoning that holds
        # the "intracranial hemorrhage" umbrella at HIGH). Those stay Ben-review / LLM-raise territory.
        # The qualified-stroke forms above floor at critical, but the direct CLINICAL name a
        # radiology/EMS report actually writes for an ischemic stroke — "cerebral infarction" ("CT
        # confirmed an acute cerebral infarction", "large cerebral infarction on imaging") — matched
        # nothing and dropped to LOW: the SAME word-choice asymmetry class as heart-attack/myocardial-
        # infarction, here the exact neuro TWIN of the already-floored "myocardial infarction" (dead
        # tissue from an occluded artery — cerebral instead of cardiac). Added at the same critical
        # floor. It is a whole two-word clinical phrase with ZERO benign English meaning, and NOT a
        # substring of any floored term (\bcerebral infarction\b cannot fire from "myocardial
        # infarction" — different first word — nor from any stroke/CVA entry), so it closes the miss
        # with no new over-fire class. This is exactly parallel to "myocardial infarction" itself,
        # which floors critical unconditionally even though "old myocardial infarction" can be a
        # chronic ECG finding — the rule layer is a conservative floor the LLM/human lowers on the rare
        # chronic case. DELIBERATELY NOT the bare short form "cerebral infarct" (an "old cerebral
        # infarct" is routinely an incidental chronic radiology finding, a genuine severity judgment
        # not a clean miss — the \bcerebral infarction\b phrase cannot fire from it, live-verified
        # LOW), and NOT the qualified-idiom risk of "massive stroke" (which fires inside "a massive
        # stroke of luck", live-verified LOW and left LOW). Surfaced in the 2026-08-26 stroke-cluster
        # rule-probe (clinical-twin sibling of the qualified-stroke / myocardial-infarction cluster).
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
        # "heart attack" / "myocardial infarction" floor at critical, but the terminal MECHANICAL
        # complication of an MI — rupture of the heart wall itself ("cardiac rupture" / "myocardial
        # rupture" / "ventricular rupture", i.e. free-wall or septal blow-out, an almost uniformly
        # fatal event an ED/cath-lab/autopsy report names directly: "arrested from cardiac rupture
        # post-MI", "autopsy confirmed left ventricular free-wall rupture") — matched nothing and
        # dropped to LOW: the SAME word-choice asymmetry class as heart-attack/myocardial-infarction/
        # ventricular-fibrillation/tamponade; the same immediately-fatal cardiac catastrophe scored
        # critical-or-LOW purely on which phrase the reporter chose. Each is a whole two-word clinical
        # phrase with ZERO benign English meaning, and NONE is a substring of any floored term
        # (\bmyocardial rupture\b cannot fire from "myocardial infarction"; \bcardiac rupture\b cannot
        # fire from "cardiac arrest"/"cardiac tamponade"/"cardiac rehab"; \bventricular rupture\b cannot
        # fire from "ventricular fibrillation"/"ventricular assist device") — all live-verified LOW
        # today, so they close the miss at the cardiac floor with no new over-fire class. DELIBERATELY
        # NOT the bare word "rupture" (heavily polysemous — "water main rupture", "disc rupture",
        # "spleen rupture" spanning many severities and categories); the phrase matcher means the
        # three adjacency phrases cannot fire from any of those. Surfaced in the 2026-08-23 cardiac
        # rule-probe (heart-wall-rupture sibling of the MI/tamponade cluster).
        # "cardiac arrest" / "asystole" floor at critical, and two of asystole's own clinical
        # phrasings matched nothing and dropped to LOW: the SAME word-choice asymmetry class as
        # heart-attack/myocardial-infarction/ventricular-fibrillation. (a) "asystolic" — the adjective
        # a monitor/EMS/code report actually writes for a patient in asystole ("the patient was
        # asystolic on arrival", "found asystolic and pulseless") — is NOT a substring match of the
        # base "asystole" (\basystole\b needs the trailing "e"; "asystolic" ends "olic"), so it slipped
        # through. Added at the same critical floor, exactly parallel to the single-word medical twins
        # already present alongside their bases ("exsanguinated"/"amputated"/"decapitated"): a purely
        # clinical word with ZERO benign English meaning. (b) "cardiac standstill" — the term a
        # point-of-care echo / bedside-ultrasound report writes for the same no-mechanical-activity
        # arrest ("POCUS showed cardiac standstill", "echo confirmed cardiac standstill") — is a whole
        # two-word clinical phrase with ZERO benign English meaning and NOT a substring of any floored
        # term, so it closes the miss at the cardiac floor with no new over-fire class. DELIBERATELY
        # NOT the bare word "standstill" (heavily polysemous — "traffic at a standstill", "talks at a
        # standstill", "production standstill"); the two-word adjacency phrase \bcardiac standstill\b
        # cannot fire from any of those (different first word), the identical discipline that keeps
        # "cardiac tamponade" safe from bare "tamponade". Both live-verified LOW before the add.
        # Surfaced in the 2026-08-26 asystole-cluster rule-probe (adjective/echo-synonym siblings of
        # the ventricular-fibrillation / cardiac-arrest cluster).
        # "not breathing" / "asphyxiation" / "suffocation" floor at critical, but the direct clinical
        # MECHANISM a report names when the same person cannot move air — "airway obstruction" /
        # "obstructed airway" (a blocked airway = the person is not breathing) — matched nothing and
        # dropped to LOW: the SAME word-choice asymmetry class as the not-breathing / choking cluster.
        # Both are whole two-word clinical phrases with NO benign everyday meaning, so they close the
        # miss at the same critical floor with zero new over-fire class. The single "airway obstruction"
        # signal also covers every qualified variant a clinician writes ("complete/upper/partial/foreign
        # body airway obstruction") because each CONTAINS it as a substring; "obstructed airway" is added
        # separately only because the reversed word order is not a substring match. DELIBERATELY NOT the
        # bare polysemous words "airway" (a flight corridor / ventilation duct) or "obstruction" ("bowel
        # obstruction" spans severities; "obstruction of justice / an obstruction on the track" are not
        # medical) — the two-word adjacency phrases cannot fire from any of those, the identical
        # discipline that keeps "cardiac standstill" safe from bare "standstill". Both live-verified LOW
        # before the add. Surfaced in the 2026-08-26 airway rule-probe (mechanism twin of the
        # not-breathing / asphyxiation / suffocation cluster).
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
        # The clinical synonyms "intracranial hemorrhage"/"cerebral hemorrhage"/"brain hemorrhage" and
        # the phrase "bleeding on the brain" all fire HIGH (via the bare "hemorrhage"/"bleeding" floor),
        # but the everyday LAY name for the identical event — a "brain bleed" ("CT confirmed a large
        # brain bleed", "she had a brain bleed after the fall", "the scan showed she'd bled on the
        # brain") — matched NOTHING and dropped to LOW: the SAME word-choice asymmetry class as
        # hemorrhage/hemorrhaging and hospitalised/haemorrhage, an intracranial hemorrhage scored
        # HIGH-or-LOW purely on whether the reporter reached for the clinical "hemorrhage"/"bleeding"
        # or the plain-English "bleed". Added the whole adjacency phrases at the SAME HIGH bleeding
        # floor as "intracranial hemorrhage" (deliberately NOT critical — the lay umbrella, like its
        # clinical umbrella, can name a slow chronic subdural, not always a hyperacute emergency; the
        # QUALIFIED "subarachnoid hemorrhage" is the only intracranial form that escalates). Both the
        # noun ("brain bleed"/"brain bleeds") and the "…on the brain" forms are needed because the bare
        # token is "bleed", NOT the already-floored "bleeding" (\bbrain bleed\b does not match "brain
        # bleeds"; neither shares a boundary with "bleeding"). DELIBERATELY NEVER the bare polysemous
        # "bleed" ("bleeding-edge tech", "bleed the brakes", "colors bleed", "bleed cash") — only the
        # zero-benign brain-adjacency phrases fire, so a benign sentence with "brain" and "bleed" as
        # non-adjacent words stays LOW. Surfaced in the 2026-08-23 cranial-bleed rule-probe.
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
        # "decompression sickness" (DCS, "the bends") is the diving/hyperbaric/caisson injury a report
        # names directly ("the diver developed decompression sickness after the rapid ascent",
        # "decompression sickness confirmed after the commercial dive", "caisson worker suffered
        # decompression sickness") — yet an isolated report matched nothing and dropped to LOW: the
        # SAME whole-hazard absent-term miss class as frostbite/hypothermia/punctured lung above. Added
        # as the MULTI-WORD phrase at the same conservative HIGH floor (NOT critical) as those exposure/
        # trauma siblings: DCS spans mild (joint pain, "niggles") to fatal (neurological/pulmonary), but
        # is routinely survivable and treated by hyperbaric recompression, so HIGH is the defensible
        # floor (the LLM or a human can raise a specific fatal case). Deliberately NOT bare
        # "decompression" — that is polysemous ("archive decompression", "needle/chest decompression"
        # for a tension pneumothorax, "decompression of the pressure vessel") and would over-fire; the
        # two-word phrase \bdecompression\s+sickness\b cannot fire from any of them (all decoys live-
        # verified LOW, and the existing needle/chest-decompression CASES still floor critical off
        # "tension pneumothorax", not this term). Surfaced in the 2026-08-22 rule-probe backlog.
        # "crush syndrome" (traumatic rhabdomyolysis) — the systemic, potentially-fatal complication of
        # prolonged crushing/entrapment: on release the damaged muscle floods the circulation with
        # myoglobin + potassium, risking hyperkalemic cardiac arrest and acute kidney injury. A responder/
        # EMS report names it directly ("crush syndrome suspected after the worker was freed from under
        # the machinery", "treated for crush syndrome following the prolonged entrapment"), yet an isolated
        # report matched nothing and dropped to LOW: the SAME whole-hazard absent-term miss class as
        # sepsis / near-drowning / decompression sickness above. Added as the MULTI-WORD phrase at the same
        # conservative HIGH floor (NOT critical) as those siblings — like sepsis it is serious but evolves
        # over hours after extrication and is treatable (aggressive fluids, dialysis), so HIGH is the
        # defensible floor (the LLM or a human can raise a specific arrest to critical, and a coincident
        # cardiac-arrest term still floors critical on its own). DELIBERATELY the two-word phrase, NEVER
        # the bare polysemous "crush"/"crushed" — heavily figurative ("crushed the sprint goal", "crushed
        # gravel", "the merger crushed competition") and deliberately excluded elsewhere in this file, so
        # bare "crush" would over-fire; \bcrush\s+syndrome\b cannot fire from any of them, closing the miss
        # with zero false-positive risk. Surfaced in the 2026-08-23 rule-probe backlog.
        "high":     ["injury", "injured", "injuries", "hospitalized", "hospitalised", "ambulance",
                     "punctured lung", "puncture wound", "puncture wounds",
                     "decompression sickness", "crush syndrome",
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
                     "brain bleed", "brain bleeds",
                     "bleed on the brain", "bled on the brain",
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
        # "thermal runaway" — the self-sustaining exothermic chain reaction inside a lithium-ion cell
        # (battery/EV/BESS/ESS) that vents flammable + toxic gas and drives fire/explosion, cell-to-cell,
        # essentially unstoppable once started — was absent from the whole fire word-family, so a report
        # that names the runaway rather than the resulting flame ("the ESS went into thermal runaway",
        # "thermal runaway detected in the lithium-ion cells", "thermal runaway of the EV battery in the
        # parking garage") matched nothing and dropped to LOW, while the same event WITH a coincident
        # "fire" token floored critical off "fire" — the SAME whole-hazard absent-term miss class as
        # "arc blast" beside "arc flash" and "hydrogen sulfide" beside "carbon monoxide": a directly-named
        # battery catastrophe scored critical-or-LOW purely on whether the reporter wrote the runaway or
        # the flame. Placed at fire/smoke critical (the physical hazard + offline next-steps — evacuate,
        # keep clear of the vent/off-gas cloud, let the fire service manage re-ignition — are the
        # fire/smoke ones), a sibling of "structure fire"/"wildfire". The two-word phrase denotes ONLY the
        # uncontrolled exothermic runaway (battery, chemical reactor, semiconductor — every sense is a
        # hazardous condition); it has ZERO benign English meaning, so it carries the SAME conservative-
        # floor tolerance already accepted for "arc flash"/"carbon monoxide" (a spec/PM mention like
        # "the BMS is designed to prevent thermal runaway" floors the same way those already do, and the
        # rule layer is a floor the LLM/human can lower) — no NEW over-fire class. Surfaced in the
        # 2026-08-20 rule-probe, queued as the next clean candidate by the 2026-08-21 arc-blast cycle.
        # "flashover" and "backdraft" — the two lethal fire-BEHAVIOR events a compartment fire produces —
        # were absent from the whole fire word-family, so a report that names the fire behavior rather than
        # the flame ("flashover in the second-floor compartment", "conditions deteriorating toward
        # flashover", "a backdraft threw the nozzleman off the landing when the door was opened") matched
        # nothing and dropped to LOW, while the same event WITH a coincident "fire"/"smoke" token floored
        # off that other word — the SAME whole-hazard absent-term miss class as "thermal runaway" beside
        # "fire", "arc blast" beside "arc flash", and "hydrogen sulfide" beside "carbon monoxide": a
        # directly-named fire catastrophe scored critical-or-LOW purely on whether the reporter wrote the
        # behavior or the flame. A flashover (near-simultaneous auto-ignition of every exposed combustible
        # in a space) and a backdraft (the smoke/deflagration explosion when air suddenly reaches an
        # oxygen-starved fire) are the classic firefighter-killing events; they belong at fire/smoke
        # critical (the physical hazard + offline next-steps — evacuate, keep clear, let the fire service
        # manage the space — are the fire/smoke ones), siblings of "structure fire"/"thermal runaway".
        # Each is a whole single word denoting ONLY a hazardous condition — EVERY real sense is a hazard
        # ("flashover" also = the electrical disruptive-discharge arc across an insulator; "backdraft" also
        # = the reverse-flow flue/chimney condition that spills combustion gas/CO indoors) — so like
        # "thermal runaway" they carry ZERO benign English meaning and the SAME conservative-floor
        # tolerance (a training/spec mention floors the same way, and the rule layer is a floor the
        # LLM/human can lower) with NO new over-fire class. DELIBERATELY singular-only: the plurals
        # "flashovers"/"backdrafts" are vanishingly rare in incident text (the event is written singular
        # or as "flashover conditions"), the same scope discipline as the singular-dominant terms; add a
        # plural entry only if a real plural miss surfaces. Surfaced in the 2026-08-21 fire-behavior
        # rule-probe (sibling of the thermal-runaway whole-hazard fix).
        "critical": ["fire", "flames", "ablaze", "blaze", "explosion", "exploded", "explosions",
                     "exploding", "explosive",
                     "detonation", "detonate", "detonated", "detonating",
                     "arson", "arsonist", "thermal runaway", "flashover", "backdraft",
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
        # "arc blast" is the pressure-wave/concussive component of an arc-flash event — the explosive
        # blast of superheated air and vaporized metal that hurls workers, ruptures eardrums and drives
        # shrapnel, a distinct hazard an electrical/OSHA incident report names in its own right ("an arc
        # blast threw the electrician across the room", "arc blast during the breaker replacement"). Its
        # sibling "arc flash" (the thermal/radiant component) already floors at critical, but the two-word
        # phrase "arc blast" matched nothing (\barc flash\b does not match "arc blast") and dropped to LOW
        # unless a coincident token happened to fire: the SAME whole-hazard absent-term miss class as
        # "hydrogen sulfide" beside "carbon monoxide". It is an unambiguous electrical catastrophe with
        # ZERO benign English meaning — the same tolerance as "arc flash" itself — so it closes the miss
        # with no new over-fire class. Surfaced in the 2026-08-20 rule-probe (sibling of arc flash).
        # A "downed power line" is the canonical fallen live-conductor hazard a utility/storm/first-
        # responder incident names directly ("a downed power line across the road", "downed power lines
        # near the substation", "crews found a downed powerline at the perimeter"). Standard practice is
        # to treat EVERY downed line as energized and lethal — it is the contact-electrocution twin of
        # "live wire" (already critical here), plus arc and ignition risk. Yet a directly-named downed
        # power line matched nothing and dropped to LOW: the ONLY existing token was the generic "downed
        # line" over in the WEATHER category at HIGH, and \bdowned\s+line\b does not match "downed POWER
        # line" (the word "power" breaks the adjacency), so the specific electrical hazard fell through —
        # the SAME whole-hazard absent-term miss class as arc blast beside arc flash. Added the phrase at
        # electrical CRITICAL beside "live wire" (not the weather HIGH floor: a generic "downed line" may
        # be a phone/data/guy line and correctly stays a conservative HIGH, but a "downed power line" is
        # unambiguously an energized-conductor emergency). "power line"/"powerline" denotes EXCLUSIVELY
        # electrical distribution and "downed" makes it a hazard → ZERO benign English meaning, so it
        # carries no new over-fire class (benign "the production/phone/assembly line went down", "he
        # downed a glass" all live-verified LOW — \bdowned power line\b never touches them). Reports write
        # it two-word AND one-word ("powerline"), singular AND plural, so each is a lexically-distinct
        # token needing its own entry (\bdowned power line\b matches none of the other three) — the same
        # multi-form discipline as tornado/tornadoes and molotov/molotovs. Surfaced in the 2026-08-21
        # 23:3x rule-probe (sibling of arc blast / live wire).
        "critical": ["live wire", "arc flash", "arc blast", "electrocution",
                     "electrocuted", "electrocute", "electrocutes", "electrocuting",
                     "downed power line", "downed power lines", "downed powerline",
                     "downed powerlines",
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
        # "phosgene" (COCl2) and "cyanide" (HCN and its salts) are the two remaining canonical lethal
        # industrial/confined-space toxicants absent from the whole gas/chemical family, so a report that
        # names the agent directly ("phosgene detected in the tank farm", "hydrogen cyanide in the
        # confined space", "cyanide gas release in the plating shop", "collapsed from cyanide poisoning")
        # matched nothing and dropped to LOW while their sibling always-a-hazard named gases "carbon
        # monoxide" and "hydrogen sulfide" (the latter added 2026-08-20) floor CRITICAL — the SAME
        # named-gas whole-hazard absent-term miss class as hydrogen-sulfide beside carbon-monoxide. Both
        # are canonical mass-casualty toxic gases (phosgene = the WWI choking agent still used in
        # isocyanate/plastics manufacture; cyanide = plating, fumigation, gold leaching, combustion of
        # plastics), so they belong at the same critical floor beside carbon monoxide / hydrogen sulfide.
        # Bare "phosgene" is a whole word denoting EXCLUSIVELY the gas (ZERO benign English meaning), and
        # bare "cyanide" likewise denotes only the toxicant — and being bare it subsumes every phrasing a
        # reporter writes ("hydrogen cyanide", "cyanide gas", "cyanide poisoning", "sodium/potassium
        # cyanide") via \bcyanide\b, so no separate multi-word entry is needed. Neither has any figurative
        # use, so they carry only the SAME conservative-floor tolerance already accepted for carbon
        # monoxide / hydrogen sulfide (a spec/inventory mention — "the lab stores sodium cyanide for the
        # assay" — floors the same way "the CO detector is tested monthly" already does, and the rule
        # layer is a floor the LLM/human can lower) with NO new over-fire class. Whole-word matching keeps
        # \bcyanide\b / \bphosgene\b from firing inside the benign prefix-sharers "cyan" (cyan ink) and
        # "phosphorescent"/"phosphate". Surfaced in the 2026-08-21 toxic-gas rule-probe (named-gas
        # siblings of the hydrogen-sulfide fix).
        # "hydrofluoric acid" (aqueous HF) and "hydrogen fluoride" (the anhydrous gas — the SAME compound
        # in two forms a report names distinctly) are the other canonical lethal industrial toxicant
        # missing from the family. HF is a mass-casualty acid/gas in semiconductor etching, oil-refinery
        # alkylation, and rust/scale removal — beyond the skin burn it drives fatal systemic hypocalcemia
        # from a small splash — yet a directly-named exposure ("splashed with hydrofluoric acid on the
        # line", "exposure to hydrogen fluoride gas") dropped to LOW, and even "hydrofluoric acid burn"
        # only reached HIGH off the generic word "burn" — the SAME whole-hazard / word-choice absent-term
        # miss class as hydrogen-sulfide, phosgene, and cyanide beside carbon monoxide. Both are whole
        # two-word chemical names denoting EXCLUSIVELY the toxicant (ZERO benign English meaning), so they
        # belong at the same critical floor beside carbon monoxide / hydrogen sulfide / phosgene / cyanide.
        # They are lexically distinct tokens (\bhydrofluoric acid\b does NOT match "hydrogen fluoride"), so
        # each gets its own entry — the same two-form discipline as "cardiac tamponade"/"pericardial
        # tamponade". DELIBERATELY NOT bare "fluoride" (benign in "fluoride toothpaste"/"water
        # fluoridation" → would over-fire) nor bare "hydrofluoric" (rare, but the full "hydrofluoric acid"
        # is what reports write). Carries only the SAME conservative-floor tolerance already accepted for
        # carbon monoxide / cyanide (an inventory mention floors the same way, and the rule layer is a
        # floor the LLM/human can lower). Surfaced in the 2026-08-21 22:0x toxic-gas rule-probe.
        # "oxygen deficient atmosphere" / "oxygen-deficient atmosphere" is the canonical OSHA confined-
        # space killer — an atmosphere below ~19.5% O2 that renders a tank/vault/silo/pit immediately
        # dangerous to life (O2 displaced by inert gas purge, rust/oxidation, decomposition, or an
        # inert-gas leak). It is the leading cause of confined-space fatalities, yet a directly-named
        # report ("atmospheric test showed an oxygen-deficient atmosphere before entry", "entrant
        # overcome in an oxygen deficient atmosphere") matched nothing and dropped to LOW: the SAME
        # whole-hazard absent-term miss class as hydrogen sulfide beside carbon monoxide — its own
        # asphyxiation sibling "nitrogen asphyxiation" already floors CRITICAL off \basphyxiation\b, but
        # the O2-deficiency phrasing had no coverage. Live-probed the miss first (all four isolated
        # phrasings LOW). Added at gas/chemical CRITICAL beside asphyxiation. Two lexically-distinct
        # forms a report writes — spaced and hyphenated (\boxygen\s+deficient\s+atmosphere\b does NOT
        # match "oxygen-deficient atmosphere", the hyphen breaks the \s+ adjacency) — so each gets its
        # own entry, the same spaced/hyphenated discipline as "downed power line"/"downed powerline".
        # Each is a whole three-word/two-word phrase denoting EXCLUSIVELY the confined-space hazard with
        # ZERO benign English meaning → no new over-fire class; a bare benign "oxygen" mention (a refilled
        # oxygen tank, supplemental oxygen administered, a liquid-oxygen delivery) is NOT adjacent to
        # "deficient atmosphere" and correctly stays LOW. Surfaced in the 2026-08-22 confined-space
        # O2-displacement rule-probe.
        # "phosgene" — a WWI chemical-warfare agent that is also an industrial gas — already floors
        # CRITICAL here, but its named chemical-weapon siblings a report would write outright ("nerve
        # agent released in the mailroom", "responders treating mustard gas exposure at the loading
        # dock", "multiple nerve agents suspected in the ventilation") matched nothing and dropped to
        # LOW: the SAME whole-hazard absent-term miss class as typhoon-beside-hurricane and hydrogen-
        # sulfide-beside-carbon-monoxide, the same lethal agent scored critical-or-LOW purely on which
        # named toxin the reporter reached for. Live-probed the miss first (all phrasings LOW). Added
        # at gas/chemical CRITICAL beside phosgene. Each is a whole word / multi-word phrase denoting
        # EXCLUSIVELY a chemical-weapon exposure with ZERO benign English meaning AND no routine-
        # presence collision (unlike "anhydrous ammonia", a named refrigerant whose mere presence is
        # routine and is therefore DELIBERATELY EXCLUDED — the same non-incident-presence discipline
        # that excluded bare "volcano"/"firearm"/"bomb"). The plural "nerve agents" needs its own entry
        # (\bnerve\s+agent\b does not match the trailing "s"), the same singular->plural discipline as
        # hostage/hostages and pipe bomb/pipe bombs. The bare polysemous name "sarin" is DELIBERATELY
        # EXCLUDED (a surname collision — "reported by J. Sarin" — the same bare-token caution that
        # excluded "stab"/"whipped"); the unambiguous "mustard gas" phrase fires while the condiment
        # "mustard" alone does not. Surfaced in the 2026-08-22 chemical-weapon rule-probe.
        "critical": ["gas leak", "carbon monoxide", "hydrogen sulfide", "phosgene", "cyanide",
                     "nerve agent", "nerve agents", "mustard gas",
                     "hydrofluoric acid", "hydrogen fluoride",
                     "oxygen deficient atmosphere", "oxygen-deficient atmosphere",
                     "grain engulfment",
                     "toxic", "chemical spill",
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
        # A trench/excavation/mine "cave-in" — soil or rock giving way into an open excavation, a
        # LEADING cause of construction fatality (OSHA) a reporter names directly ("the trench
        # caved in on the crew", "a cave-in buried the excavation", "cave-in at the north dig") —
        # matched nothing and dropped to LOW: unlike "collapse"/"structural failure"/"foundation
        # failure" (critical) it contains NO floored substring, and unlike "sinkhole"/"subsidence"
        # (HIGH) the earth-movement sibling term itself was simply absent — the SAME whole-hazard
        # absent-term miss class as sinkhole above, one excavation over. Added at the SAME
        # conservative HIGH floor as "sinkhole"/"subsidence" (the earth/ground-failure precedent, not
        # the built-structure "collapse" critical floor): a cave-in ranges from an empty-trench slump
        # to a fatal burial, so HIGH is the defensible floor the LLM/human can raise — and a cave-in
        # that buries or injures a worker independently floors critical via the injury/medical terms.
        # DELIBERATELY the HYPHENATED noun "cave-in"/"cave-ins" ONLY, NEVER the spaced idiom "cave
        # in" (to "cave in to demands/pressure", the negotiation metaphor) — \bcave\-in\b requires a
        # literal hyphen so it cannot fire from "cave in to", closing the miss with zero
        # false-positive risk. The plural "cave-ins" needs its own entry (\bcave\-in\b won't match
        # "cave-ins"), the same singular->plural tokenization discipline as sinkhole/sinkholes.
        # Surfaced in the 2026-08-24 structural earth-failure rule-probe (excavation sibling of the
        # sinkhole/subsidence ground-failure cluster).
        # A "rockslide" — a mass of rock breaking loose and sliding down a slope onto a road, rail
        # line, quarry, or work area (burying/crushing whatever is below) — is the same acute
        # earth-movement emergency a reporter names directly ("a rockslide buried the highway",
        # "rockslide at the quarry trapped two workers"), yet it matched nothing and dropped to LOW
        # while its ground-failure siblings "sinkhole"/"cave-in"/"subsidence" already floor at HIGH:
        # the SAME whole-hazard absent-term miss class, one slope over. Added at the SAME conservative
        # HIGH earth-failure floor (a rockslide ranges from an empty-road slump to a fatal burial, so
        # HIGH is the defensible floor the LLM/human can raise — and one that buries or injures a
        # worker independently floors critical via the injury/medical terms). The plural "rockslides"
        # needs its own entry (\brockslide\b does not match "rockslides"), the same singular->plural
        # tokenization discipline as sinkhole/sinkholes and cave-in/cave-ins. DELIBERATELY the
        # ONE-WORD "rockslide"/"rockslides" ONLY — NEVER the polysemous earth-disaster siblings
        # "landslide" ("a landslide victory"), "mudslide" (the cocktail/ice-cream), or "avalanche"
        # ("an avalanche of tickets/complaints"), all live-verified LOW and left Ben-review; and NOT
        # the spaced "rock slide" (a playground/water-park attraction). \brockslide\b as one word has
        # zero benign English meaning, so it closes the miss with zero false-positive risk (bare
        # "rock" and "slide" do not fire). Surfaced in the 2026-08-25 earth-movement rule-probe.
        "high":     ["crack in wall", "structural crack", "sagging", "buckling", "subsidence",
                     "sinkhole", "sinkholes", "cave-in", "cave-ins", "rockslide", "rockslides",
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
        # The explosive-device family is only HALF-covered: "bomb threat" floors critical, but by
        # substring \bbomb\s+threat\b matches ONLY the THREAT — an actual found or detonated DEVICE a
        # reporter names outright ("a pipe bomb was found in the lobby", "car bomb detonated in the
        # lot", "two pipe bombs left at the north gate") matched nothing and dropped to LOW: the SAME
        # threat-vs-device miss class as the stab-wound (injury) vs stabbing-attack (event) split just
        # fixed above, the same hazard scored critical-or-LOW purely on which half of the event the
        # words name. These belong at critical beside "bomb threat"/"molotov" (an actual explosive is
        # present). Each is a MULTI-WORD adjacency phrase denoting EXCLUSIVELY an explosive device
        # (zero benign meaning), and the plurals need their own entries (\bpipe\s+bomb\b won't match
        # "pipe bombs"), the same singular->plural tokenization discipline as molotov/molotovs and
        # kidnapping/kidnappings. DELIBERATELY EXCLUDES the bare polysemous "bomb" — "bath bomb",
        # "photobomb", "the movie bombed", "bombed the interview" all carry benign collisions — the
        # SAME whole-phrase-only discipline that added "stab wound"/"pistol-whipped" but excluded the
        # bare "stab"/"whipped". ("car bomb detonated" already fires fire/smoke via \bdetonated\b; this
        # closes the FOUND/undetonated-device case that carries no explosion token.) Surfaced in the
        # 2026-08-22 explosive-device rule-probe (threat-vs-device sibling of the stabbing-event fix).
        # The singular "hostage" floors CRITICAL, but its plural "hostages" — how an active abduction
        # crisis is actually reported ("the gunman took hostages in the control room", "multiple
        # hostages held in the vault") — is a distinct token that \bhostage\b does NOT match (no
        # boundary before the trailing "s"), so it dropped to LOW: the SAME singular->plural
        # tokenization miss already closed for molotov/molotovs, pipe bomb/pipe bombs,
        # kidnapping/kidnappings and the weather plurals. "hostages" is a whole word with ZERO benign
        # English meaning, so its own entry closes the miss with no new over-fire class. Surfaced in
        # the 2026-08-22 security/violence rule-probe.
        # "gunman"/"gunmen" name the armed offender in an active-shooter/armed-intrusion event exactly
        # as a report writes it ("a lone gunman barricaded on the third floor", "two gunmen entered the
        # facility") — the sibling of the already-critical "shooter"/"active shooter", yet both matched
        # nothing and dropped to LOW (a "gunman opened fire" line only floored off the coincidental
        # "fire" token, not the offender). The irregular plural means \bgunman\b does NOT match "gunmen"
        # (the a->e vowel change breaks it, no trailing-s form), so each needs its own entry — the SAME
        # singular->plural miss class as hostage/hostages, molotov/molotovs. Both are whole words with
        # ZERO benign English meaning; the word boundary keeps them from firing inside "gunmetal". Added
        # in the 2026-08-22 security/violence rule-probe.
        # "gunpoint" (the firearm-coercion event a report writes as "held at gunpoint", "robbed at
        # gunpoint", "employees ordered around at gunpoint") is the firearm sibling of the already-
        # critical "knifepoint" — and the comment on knifepoint above already NAMES gunpoint as its
        # sibling — yet the bare word lived ONLY inside theft's "robbery at gunpoint", so a gunpoint
        # incident WITHOUT the word "robbery" ("staff held at gunpoint", "suspect fled at gunpoint")
        # matched nothing here and dropped to LOW. \bgunpoint\b is a whole word with ZERO benign
        # English meaning, so its own entry closes the miss with no over-fire (theft's "robbery at
        # gunpoint" still fires too; both are critical, security wins the tie by category order).
        # "grenade"/"grenades" (a thrown explosive weapon) is the sibling of the already-critical
        # "pipe bomb"/"molotov"/"car bomb": a directly-named explosive-device attack a report writes
        # plainly ("a grenade was thrown into the lobby", "grenades recovered from the vehicle"), yet
        # both matched nothing and dropped to LOW. Whole words with no benign OPERATIONAL polysemy in
        # incident text; the plural is a distinct token needing its own entry (\bgrenade\b lacks a
        # boundary before the trailing "s"), same singular->plural class as pipe bomb/pipe bombs.
        # "ied"/"ieds" is the universal military/EMS acronym for the already-critical "improvised
        # explosive device" (gas/chemical + here): a first responder or guard writes "IED discovered
        # in the lobby" far more often than the spelled-out phrase, yet the acronym matched nothing
        # and dropped to LOW while the full phrase floored critical — the SAME acronym-vs-phrase miss
        # class as CPR (== cardiac arrest). \bied\b / \bieds\b are word-boundary-guarded, so they do
        # NOT fire inside "studied"/"tied"/"applied" (no boundary before the internal "ied"); no
        # English word IS "ied", so zero false-positive risk. Plural is a distinct token.
        # "sniper"/"snipers" (a concealed shooter deliberately targeting people) is the active-shooter
        # sibling of the already-critical "shooter"/"active shooter"/"gunman": a report names it
        # directly ("sniper on the parking-garage roof", "reports of a sniper near the north gate"),
        # yet both matched nothing and dropped to LOW. Whole words; in operational incident text a
        # "sniper" is an active lethal threat with no benign meaning (the metaphor is the GERUND
        # "sniping", deliberately NOT added). Plural is a distinct token. Added in the 2026-08-22
        # security/violence rule-probe (gunpoint / grenade / IED / sniper).
        # The passive/fatal report forms of a stabbing — "was stabbed" ("a worker was stabbed in the
        # parking lot", "the guard was stabbed"), "stabbed to death", "fatally stabbed" — are the way
        # an assault is actually written up, yet they matched nothing and dropped to LOW while the NOUN
        # "stab wound" and the event phrase "stabbing attack" floor critical: the SAME word-form
        # asymmetry class as amputation->amputated / decapitation->decapitated. Offline the rule layer
        # is the only floor, so add these at critical beside "stab wound". DELIBERATELY qualified
        # multi-word phrases, NOT the bare participle "stabbed" — the existing FP guard proves "stabbed"
        # is polysemous ("stabbed at his lunch" = poked, "took a stab at it"), so bare "stabbed" would
        # over-fire; "was stabbed" / "stabbed to death" / "fatally stabbed" carry no benign sense (the
        # same qualified-phrase discipline used for "cardiogenic shock" over bare "shock"). Added in
        # the 2026-08-22 security/violence rule-probe.
        # The fatal-SHOOTING victim-outcome phrasings are the exact twins of the stabbing forms just
        # above, yet only the stabbing side was covered: "fatally stabbed"/"stabbed to death" floor
        # critical, but their firearm siblings — "fatally shot", "shot to death", "shot and killed",
        # the way a fatal shooting is actually written up ("the victim was fatally shot in the lot",
        # "a man was shot to death outside", "the guard was shot and killed by an intruder") — matched
        # NOTHING and dropped to LOW. The firearm EVENT/OFFENDER family (gunshot / gunfire / shots fired
        # / shooter / gunman) is fully covered, but NONE of its tokens appear in "the victim was fatally
        # shot", so that report currently scores LOW — the SAME word-form asymmetry class as
        # fatally-stabbed vs stab-wound, one weapon's fatal-outcome phrasing critical and the other's
        # LOW purely on word choice. "opened fire" is the twin of the already-critical "shots fired":
        # by substring it only ever floored off the COINCIDENTAL "fire" token (fire/smoke), giving the
        # RIGHT severity for the WRONG reason (a shooting mis-attributed to fire) — its own entry fixes
        # the rationale, not just the score. All four are qualified MULTI-WORD adjacency phrases with no
        # benign sense (the same discipline as "was stabbed"/"stabbed to death"): DELIBERATELY NOT the
        # bare polysemous "shot" ("photo shot", "flu shot", "gave it a shot", "shot down the proposal")
        # nor "shot dead" (real benign "shot dead center" in archery/aim text) nor "gunned down" (the
        # driving sense "gunned down the highway") — only the whole fatal-shooting phrases fire. Added
        # in the 2026-08-24 security/violence rule-probe (firearm twin of the fatally-stabbed fix).
        # An "acid attack" — a corrosive-substance assault, a recognized violent-crime category a
        # reporter names directly ("victim of an acid attack at the gate", "two acid attacks reported
        # this month") — matched NOTHING and dropped to LOW, while its direct siblings "knife attack"/
        # "stabbing attack"/"mass stabbing" already floor at security/intrusion critical: the SAME
        # weapon/violent-assault miss class scored critical-or-LOW purely on which weapon the reporter
        # named. Added the qualified two-word phrase (+ plural "acid attacks", a distinct token
        # \bacid attack\b does NOT match — the same singular->plural discipline as stab wound/stab
        # wounds and molotov/molotovs) at the assault-family critical floor beside "knife attack".
        # DELIBERATELY NOT the bare polysemous "acid": "acid wash", "acid rain", "lactic acid", the
        # figurative "acid test" all carry benign senses (live-verified LOW) — only the whole
        # "acid attack" adjacency fires, so this closes the miss with zero false-positive risk.
        # Added in the 2026-08-25 security/violence rule-probe (corrosive-assault sibling of knife attack).
        # A "firebomb" — an incendiary weapon/attack a reporter names outright ("the office was
        # firebombed overnight", "a firebomb was thrown through the window", "firebombs hurled at the
        # gate") — matched NOTHING and dropped to LOW, even though its direct synonym "molotov" already
        # floors CRITICAL right here and "arson"/"ablaze" floor CRITICAL in fire/smoke: the SAME
        # incendiary-attack event scored critical-or-LOW purely on which word the reporter reached for.
        # A firebomb IS a molotov/incendiary device, so it belongs at the assault-family critical floor
        # beside "molotov". This is also a rationale twin of the "opened fire" fix — one might expect it
        # to at least floor off the "fire" token, but whole-word \bfire\b does NOT match inside
        # "firebomb"/"firebombed", so the report scored a true LOW, not merely a mis-attributed hit.
        # Each surface form needs its own entry — \bfirebomb\b matches neither "firebombs" (plural) nor
        # "firebombed" (verb) nor "firebombing" (gerund), the same verb/plural tokenization discipline as
        # molotov/molotovs and carjacking/carjacked/carjackings. "firebomb-" denotes EXCLUSIVELY the
        # incendiary weapon/act — NO benign English meaning (unlike the polysemous bare "fire" = fire
        # someone / open fire / fire drill, deliberately not leaned on) — so this closes the miss with
        # zero false-positive risk. Added in the 2026-08-25 security/violence rule-probe (incendiary-
        # weapon sibling of molotov).
        # A "suicide bomber" / "suicide bombing" / "suicide bomb" — a person-borne explosive attack, the
        # most iconic mass-casualty device — is the direct sibling of the already-critical explosive-device
        # cluster "pipe bomb"/"car bomb"/"bomb threat"/"grenade"/"ied", yet EVERY form matched NOTHING and
        # dropped to LOW: a report writing "a suicide bomber approached the north gate", "a suicide bombing
        # at the market entrance", or "a suicide bomb left under the bench" carries no other floored token
        # (bare "bomb" is DELIBERATELY excluded for its benign collisions — "bath bomb", "photobomb", "the
        # movie bombed" — so \bbomb\b is not in the taxonomy), so the whole attack scored LOW — the SAME
        # threat-vs-device / half-covered-family miss class as pipe bomb / car bomb beside bomb threat. Each
        # form is a MULTI-WORD adjacency phrase denoting EXCLUSIVELY a person-borne explosive attack with
        # ZERO benign English meaning, so they belong at critical beside "pipe bomb"/"car bomb"/"molotov".
        # Every surface form is a lexically-distinct token needing its own entry — \bsuicide\s+bomber\b
        # matches neither "suicide bombers" (plural), "suicide bombing" (gerund) nor "suicide bomb" (device;
        # the trailing "er"/"ers"/"ing"/"ings"/"s" breaks the word boundary) — the SAME verb/plural
        # tokenization discipline as firebomb/firebombs/firebombed/firebombing and pipe bomb/pipe bombs.
        # DELIBERATELY the QUALIFIED "suicide …" phrases only, NEVER the bare polysemous "suicide" —
        # "suicide prevention", a "suicide clause" in an insurance policy, a hockey "suicide pass" all carry
        # benign senses (live-verified LOW) — only the whole "suicide bomb*" adjacency fires, so this closes
        # the miss with zero false-positive risk. Surfaced in the 2026-08-26 security/violence rule-probe
        # (person-borne explosive-device sibling of pipe bomb / car bomb).
        # A "car bombing" / "car bombings" — the GERUND/event name of the already-critical device "car bomb"
        # / "car bombs", the direct sibling of the "suicide bombing"/"suicide bombings" gerund shipped the
        # same day. A report writing "a car bombing killed six outside the gate" or "a series of car bombings
        # struck the district" carries no other floored token (bare "bomb" is DELIBERATELY excluded for its
        # benign collisions — "bath bomb", "photobomb", "the movie bombed"), so the whole vehicle-borne
        # explosive attack scored LOW while the noun "car bomb" floors critical — the SAME word-choice
        # (device-noun-vs-event-gerund) asymmetry as suicide bomb/bombing and firebomb/firebombing. The
        # gerund is a lexically-distinct token needing its own entry — \bcar\s+bomb\b matches neither
        # "car bombing" (the trailing "ing" breaks the word boundary) nor, once added, does \bcar\s+bombing\b
        # match "car bombings" (trailing "s") — the SAME verb/gerund/plural tokenization discipline as
        # suicide bomber/bombing/bombings and firebomb/firebombed/firebombing. "car bombing" denotes
        # EXCLUSIVELY a vehicle-borne explosive attack with ZERO benign English meaning (the cocktail is the
        # noun "Irish Car Bomb", never "car bombing"), so it belongs at critical beside "car bomb"/"pipe
        # bomb"/"suicide bombing". Surfaced in the 2026-08-26 (evening) security/violence rule-probe
        # (vehicle-borne explosive-device gerund sibling of car bomb / suicide bombing).
        "critical": ["active shooter", "armed", "weapon", "hostage", "hostages", "bomb threat",
                     "suicide bomber", "suicide bombers", "suicide bombing", "suicide bombings",
                     "suicide bomb", "suicide bombs",
                     "pipe bomb", "pipe bombs", "car bomb", "car bombs",
                     "car bombing", "car bombings",
                     "intruder armed", "kidnapping", "kidnapped", "kidnappings",
                     "gunshot", "gunshots", "gunfire", "gunman", "gunmen",
                     "shots fired", "active shooting", "shooter", "shooting",
                     "fatally shot", "shot to death", "shot and killed", "opened fire",
                     "was stabbed", "stabbed to death", "fatally stabbed",
                     "stab wound", "stab wounds", "molotov", "molotovs",
                     "firebomb", "firebombs", "firebombed", "firebombing",
                     "knife attack", "stabbing attack", "stabbing spree",
                     "mass stabbing", "knifepoint", "gunpoint",
                     "grenade", "grenades", "ied", "ieds",
                     "sniper", "snipers",
                     "pistol-whipped", "pistol whipped",
                     "pistol-whipping", "pistol whipping",
                     "acid attack", "acid attacks"],
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
        # The natural-catastrophe family is missing its volcanic members: "earthquake"/"tsunami"/
        # "hurricane"/"typhoon"/"wildfire" all floor at critical, but a "volcanic eruption" — a named
        # natural disaster on exactly the same footing, with "pyroclastic flow" as its lethal mechanism
        # (the superheated gas/ash surge that is the actual killer, the volcanic analogue of the tsunami
        # that follows a quake) and "lava flow" as the advancing molten hazard — all matched nothing and
        # dropped to LOW: the SAME whole-catastrophe absent-term miss class as typhoon beside hurricane,
        # the same event scored critical-or-LOW purely on which natural disaster it names. Added the
        # three MULTI-WORD phrases at the same critical floor as earthquake/tsunami. Each denotes
        # EXCLUSIVELY the volcanic hazard — ZERO benign English meaning — so this closes the miss with
        # no false-positive risk. DELIBERATELY EXCLUDES the bare polysemous roots "volcano" (a dormant
        # volcano near a site is geography, not an incident — same non-incident-presence discipline that
        # excluded bare "firearm"/"bomb") and "erupted"/"eruption" (heavy figurative use — "the crowd
        # erupted", "violence erupted", "an eruption of applause", "a rash erupted" — all guarded LOW);
        # only the whole zero-benign phrases fire. Surfaced in the 2026-08-22 natural-disaster rule-probe.
        # "storm surge" is the LETHAL MECHANISM of a hurricane/typhoon — the wind-driven wall of seawater
        # that is historically the LEADING cause of hurricane deaths (the Katrina killer), exactly the
        # tsunami-after-the-quake / pyroclastic-flow-of-the-volcano class of "the named catastrophe's
        # actual killer." Yet it scored only HIGH, not critical: the bare word "storm" sits at the weather
        # HIGH floor, so "storm surge" matched "storm" and floored one level LOW of the hurricane it
        # accompanies — WORSE than a pure absent-term miss, an active UNDER-floor of the deadliest coastal
        # hazard. Added the whole phrase at the hurricane/tsunami critical floor. "storm surge" has ZERO
        # benign English meaning (unlike the polysemous half "surge" — a power surge, a demand/traffic
        # surge, a surge of adrenaline — which is DELIBERATELY EXCLUDED and left to the existing whole
        # phrase "power surge"; only the adjacent two-word storm-surge phrase fires). Surfaced in the
        # 2026-08-22 natural-disaster rule-probe alongside the volcanic members.
        # A "derecho" is a widespread, long-lived, fast-moving STRAIGHT-LINE WINDSTORM — hurricane-force
        # gusts (58+ mph, often 80-100+) along a damage swath 240+ miles long, a directly-named, NWS-warned
        # convective catastrophe (the 2012 mid-Atlantic and 2020 Iowa derechos each killed people, caused
        # billions in damage, and left millions without power). It is a named natural disaster on exactly
        # the same footing as its critical siblings tornado/hurricane/typhoon — NOT the generic "storm"/
        # "high winds" that sit at the weather HIGH floor — yet it matched nothing and dropped to LOW: the
        # SAME whole-catastrophe absent-term miss class as typhoon-beside-hurricane and volcanic-eruption,
        # the same event scored critical-or-LOW purely on which named storm it is. Added the whole word at
        # the tornado/hurricane critical floor; the plural "derechos" needs its own entry (\bderecho\b
        # won't match "derechos"), the same singular->plural tokenization discipline already applied to
        # tornadoes/hurricanes/typhoons. As an English loanword "derecho" denotes EXCLUSIVELY this
        # windstorm — its Spanish sense ("right"/"straight"/"law") appears only in Spanish-language text,
        # never in English facility/incident reporting, so this is the SAME negligible cross-language
        # tolerance already accepted for "typhoon"/"molotov" and closes the miss with no operational
        # false-positive risk (bare "storm"/"winds" continue to floor HIGH on their own). Surfaced in the
        # 2026-08-25 severe-weather rule-probe.
        # A "lahar" is a volcanic mudflow/debris flow — a fast-moving torrent of volcanic ash, rock, and
        # water that races down a volcano's flanks, burying everything in its path (the 1985 Nevado del
        # Ruiz lahar entombed the town of Armero and killed ~23,000 people, one of the deadliest volcanic
        # disasters in recorded history). It is a directly-named, USGS/NWS-warned volcanic catastrophe on
        # exactly the same footing as its critical siblings already floored here — "volcanic eruption",
        # "pyroclastic flow", "lava flow" — yet "lahar" matched nothing and dropped to LOW: the SAME
        # whole-catastrophe absent-term miss class as derecho-beside-hurricane and typhoon-beside-hurricane,
        # a named disaster scored critical-or-LOW purely on which volcanic hazard it is. Added the whole word
        # at the volcanic critical floor; the plural "lahars" needs its own entry (\blahar\b does not match
        # "lahars"), the same singular->plural tokenization discipline already applied to
        # tornadoes/hurricanes/typhoons/derechos. As a Javanese/Indonesian loanword adopted into English and
        # global geology, "lahar" denotes EXCLUSIVELY this volcanic mudflow — it has zero benign English
        # meaning, so this is the SAME negligible cross-language tolerance already accepted for
        # "typhoon"/"derecho"/"molotov" and closes the miss with no operational false-positive risk. Surfaced
        # in the 2026-08-25 volcanic-hazard rule-probe.
        # "temblor"/"temblors" is the English-adopted loanword synonym for "earthquake" (Merriam-Webster
        # defines it simply as "earthquake"; from Spanish temblor = trembling). US and international
        # journalism reach for it routinely as a same-meaning variant ("a strong temblor struck the
        # region"), so an incident report is as likely to write "temblor" as "earthquake" — yet its exact
        # critical sibling "earthquake"/"earthquakes" is right here at the critical floor while "temblor"
        # matched nothing and dropped to LOW: the SAME synonym-of-a-critical-term miss as typhoon-beside-
        # hurricane, the same event scored critical-or-LOW purely on which word the reporter used. Added
        # both forms at the same critical floor as "earthquake"; the plural "temblors" needs its own entry
        # (\btemblor\b does not match "temblors"), the same singular->plural tokenization discipline
        # already applied to earthquakes/hurricanes/typhoons/derechos/lahars. As a Spanish loanword adopted
        # into English, "temblor" denotes EXCLUSIVELY an earthquake — it has zero benign English meaning, so
        # this is the SAME negligible cross-language tolerance already accepted for
        # "typhoon"/"derecho"/"lahar"/"molotov" and closes the miss with no operational false-positive risk.
        # Surfaced in the 2026-08-25 seismic-synonym rule-probe.
        # "megaquake"/"megaquakes" is a great earthquake (moment magnitude ~8+): the Cascadia, Nankai,
        # and Alaska "megaquake" is a directly-named seismic catastrophe journalism and hazard agencies
        # use routinely for the largest, most destructive quakes. Its exact sibling "earthquake"/"temblor"
        # already floors CRITICAL here, yet "megaquake" matched nothing and dropped to LOW: the SAME
        # synonym-of-a-critical-term miss as temblor-beside-earthquake — and STRICTLY worse, since a
        # megaquake is by definition a catastrophic earthquake, so flooring it CRITICAL can only ever be
        # correct. Added both forms at the same critical floor as "earthquake"; the plural "megaquakes"
        # needs its own entry (\bmegaquake\b does not match "megaquakes"), the same singular->plural
        # tokenization discipline already applied to earthquakes/temblors/hurricanes/typhoons/derechos/lahars.
        # "megaquake" is a whole word denoting EXCLUSIVELY a very large earthquake — it has zero benign
        # English meaning (unlike the polysemous bare "quake", which can mean "quake with fear" and is
        # deliberately left out), so this closes the miss with no operational false-positive risk. Surfaced
        # in the 2026-08-25 seismic-magnitude rule-probe.
        # "tropical cyclone"/"tropical cyclones" is the FORMAL meteorological umbrella (WMO/NWS/JTWC/BoM)
        # for the exact same storm that is regionally named "hurricane" (Atlantic/E-Pacific) or "typhoon"
        # (NW-Pacific) — both of which already floor CRITICAL here. Cyclone Idai (Mozambique) and Cyclone
        # Nargis (Myanmar, ~138,000 dead) are directly-named tropical-cyclone catastrophes agencies and
        # international journalism report routinely, yet "tropical cyclone" matched nothing and dropped to
        # LOW: the SAME whole-catastrophe absent-term miss class as typhoon-beside-hurricane, the same event
        # scored critical-or-LOW purely on which regional/umbrella word the reporter reached for. The bare
        # root "cyclone" was DELIBERATELY excluded above as polysemous ("cyclone fence" = chain-link fence,
        # "cyclone separator" = industrial dust collector); the QUALIFIER "tropical" removes that ambiguity
        # entirely — \btropical\s+cyclone\b cannot match a cyclone fence/separator — so the qualified phrase
        # closes the synonym gap WITHOUT reopening the excluded polysemous root, the exact bare-vs-qualified
        # discipline already drawn for tamponade (excluded) vs "cardiac tamponade" and pneumothorax
        # (excluded) vs "tension pneumothorax". Added both at the hurricane/typhoon critical floor; the
        # plural "tropical cyclones" needs its own entry (\btropical\s+cyclone\b won't match the plural), the
        # same singular->plural tokenization discipline applied to hurricanes/typhoons/earthquakes/lahars.
        # Zero benign English meaning -> zero operational false-positive risk. Surfaced in the 2026-08-25
        # tropical-cyclone-umbrella rule-probe.
        # "megatsunami"/"megatsunamis" is a great tsunami (wave amplitude far beyond an ordinary tsunami,
        # typically from a landslide/volcanic-flank collapse/asteroid impact rather than a fault rupture):
        # the 1958 Lituya Bay megatsunami ran up ~524 m, and a future Cumbre Vieja (La Palma) flank collapse
        # is the textbook Atlantic megatsunami scenario. It is the EXACT tsunami analogue of "megaquake"
        # (added in the same 2026-08-25 seismic-magnitude probe): a mega- prefix on an already-critical root,
        # denoting a strictly-worse version of the catastrophe, so flooring it CRITICAL can only ever be
        # correct. Its sibling "tsunami"/"tsunamis" is right here at the critical floor, yet "megatsunami"
        # matched nothing and dropped to LOW — \btsunami\b does NOT match "megatsunami" (no word boundary
        # before "tsunami" inside the compound), the SAME compound-prefix tokenization miss already closed
        # for megaquake-beside-earthquake. Added both forms; the plural "megatsunamis" needs its own entry
        # (\bmegatsunami\b does not match "megatsunamis"), the same singular->plural discipline applied to
        # earthquakes/temblors/megaquakes/tsunamis/hurricanes/typhoons/derechos/lahars. "megatsunami" is a
        # whole word denoting EXCLUSIVELY a very large tsunami — zero benign English meaning (unlike the
        # polysemous bare "wave"/"surge", deliberately not floored on their own) — so this closes the miss
        # with no operational false-positive risk. Surfaced in the 2026-08-25 tsunami-magnitude rule-probe.
        # "superstorm"/"superstorms" is the media/agency term for an exceptionally large, destructive storm
        # system that has escaped the ordinary-storm scale — Superstorm Sandy (2012, ~230 dead, ~$70B) and the
        # 1993 "Storm of the Century" superstorm (~300 dead) are the textbook cases, directly-named storm
        # catastrophes on the same footing as the critical siblings hurricane/typhoon/derecho already here. It
        # is the storm analogue of megaquake/megatsunami: a super- prefix on the storm root that denotes a
        # strictly-worse version of the base hazard. "storm" alone floors only HIGH (a routine thunderstorm is
        # ordinary), but \bstorm\b does NOT match "superstorm" (no word boundary before "storm" inside the
        # compound), so the qualified catastrophe dropped to LOW — the SAME compound-prefix tokenization miss
        # already closed for megaquake-beside-earthquake and megatsunami-beside-tsunami. Added both forms at
        # the named-catastrophe critical floor beside derecho (NOT the generic "storm" HIGH tier — you do not
        # call a routine storm a "superstorm"; the word is reserved for genuine catastrophes); the plural
        # "superstorms" needs its own entry (\bsuperstorm\b does not match "superstorms"), the same
        # singular->plural discipline applied to hurricanes/typhoons/derechos/megaquakes. "superstorm" is a
        # whole word denoting EXCLUSIVELY a catastrophic storm — zero benign English meaning — so this closes
        # the miss with no operational false-positive risk. Surfaced in the 2026-08-26 storm-magnitude rule-probe.
        # "bushfire"/"bushfires" is the Australian/NZ and international-Commonwealth regional name for the EXACT
        # same event already floored critical here as "wildfire"/"wildfires" — an uncontrolled vegetation fire
        # (the 2019-20 Australian "Black Summer" bushfires killed 33+ people directly, ~450 more from smoke, and
        # burned ~24 million hectares, one of the worst natural disasters in Australian history). Yet it matched
        # nothing and dropped to LOW: the SAME regional-synonym-of-a-critical-term miss as typhoon-beside-
        # hurricane (NW-Pacific name) and temblor-beside-earthquake (Spanish loanword), the same catastrophe
        # scored critical-or-LOW purely on which regional word the reporter reached for. Overseas facilities,
        # imported PDF templates, and Commonwealth contractors write "bushfire" as routinely as a US report
        # writes "wildfire" — exactly the international-report miss the taxonomy exists to close (the stated
        # typhoon rationale). Added both forms at the same critical floor as "wildfire"; the plural "bushfires"
        # needs its own entry (\bbushfire\b does not match "bushfires"), the same singular->plural tokenization
        # discipline already applied to wildfires/hurricanes/typhoons/derechos/lahars. "bushfire" is a whole
        # compound word denoting EXCLUSIVELY an uncontrolled wildland fire — zero benign English meaning — so
        # this closes the miss with no operational false-positive risk. Surfaced in the 2026-08-27 wildfire-
        # synonym rule-probe.
        # "megafire"/"megafires" name a great wildfire (the >100,000-acre class NIFC/USFS and the press use for
        # the worst events — the 2020 August Complex "gigafire" topped a million acres) — a directly-named fire
        # catastrophe strictly worse than the already-critical "wildfire"/"bushfire", the wildfire analogue of
        # the already-critical "megaquake"/"megatsunami". Yet bare "megafire" matched nothing (a sample only
        # reached HIGH incidentally off "burned" in the acreage clause) and the plural "megafires" dropped to
        # LOW: \bwildfire\b does not fire inside the compound "megafire", and \bmegafire\b does not match the
        # plural "megafires", so each is a distinct entry — the same "mega-" compound + singular->plural
        # discipline already applied to megaquake/megaquakes and megatsunami/megatsunamis. "megafire" is a whole
        # compound word denoting EXCLUSIVELY a catastrophic wildfire — zero benign English meaning — so flooring
        # it critical can only ever be correct. Surfaced in the 2026-08-27 wildfire-magnitude rule-probe.
        # "gigafire"/"gigafires" name the >1,000,000-acre wildfire — the press/agency term one order above the
        # already-critical "megafire" (>100,000 acres), coined for the 2020 August Complex fire that topped a
        # million acres in California (the very event the megafire rationale above cites). It is the strictly-
        # worse fire catastrophe, exactly the "giga-" step past "mega-" and the same directly-named-worse-version
        # relationship megafire has to wildfire. Bare "gigafire" matches nothing already floored: \bwildfire\b and
        # \bmegafire\b do not fire inside the compound "gigafire", and \bgigafire\b does not match the plural
        # "gigafires", so each is a distinct entry — the same "-fire" compound + singular->plural discipline just
        # applied to megafire/megafires. "gigafire" is a whole compound word denoting EXCLUSIVELY a catastrophic
        # million-acre wildfire — zero benign English meaning — so flooring it critical can only ever be correct.
        # Surfaced in the 2026-08-27 wildfire-magnitude rule-probe (order-of-magnitude sibling above megafire).
        # "firenado"/"firenados"/"firenadoes" is the press/plain-language name for the fire-generated tornadic
        # vortex — a whirling column of flame and superheated gas thrown off a large wildfire, EF-scale in the
        # worst cases: the 2018 Carr Fire firenado was rated EF3-equivalent (~143 mph) and killed a firefighter.
        # It is a directly-named lethal wildfire hazard on the same footing as the already-critical bare
        # "tornado", yet it matched NOTHING already floored: it is a single compound word, so \btornado\b does NOT
        # match it (no word boundary before "tornado" in "fire"+"nado"), and — unlike its spaced siblings —
        # \bfire\b does NOT match it either (no boundary after "fire"), so it dropped to LOW. The plural/variant
        # forms are distinct tokens (\bfirenado\b does not match "firenados"/"firenadoes"), the same
        # singular->plural discipline applied to megafires/gigafires. Its two SPACED siblings are deliberately NOT
        # added because each is already floored by an existing token: "fire whirl" (the USGS technical name) and
        # "fire tornado" both carry a bare "fire" as a separate word (\bfire\b, already critical in the fire/smoke
        # category) and "fire tornado" additionally carries "tornado" — only the closed compound "firenado" is a
        # genuine miss. It denotes EXCLUSIVELY the wildfire vortex — zero benign English meaning — so flooring
        # critical can only ever be correct. Surfaced in the 2026-08-27 wildfire-magnitude rule-probe (closed-
        # compound tornadic sibling of the already-critical wildfire cluster; verified against \bfire\b/\btornado\b
        # by fault injection).
        # "pyroclastic surge"/"pyroclastic surges" is the USGS-distinguished DILUTE, turbulent, fast-moving
        # volcanic density current — the more diffuse and often MORE lethal sibling of the denser "pyroclastic
        # flow" already floored critical here. The pyroclastic surge is the historical killer: the 1902 Mont
        # Pelee surge annihilated St. Pierre (~28,000-30,000 dead), and the surge component is what entombed
        # Herculaneum/Pompeii. USGS hazard reporting names the two as distinct phenomena ("pyroclastic flows and
        # surges"), yet \bpyroclastic\s+flow\b does NOT match "pyroclastic surge" (different second word), so the
        # surge matched nothing and dropped to LOW — the SAME whole-hazard absent-term miss class as "arc blast"
        # beside "arc flash" and "hydrogen sulfide" beside "carbon monoxide", a directly-named volcanic
        # catastrophe scored critical-or-LOW purely on which of the two density-current terms the reporter used.
        # Added both forms at the volcanic critical floor beside "pyroclastic flow"; the plural "pyroclastic
        # surges" needs its own entry (\bpyroclastic\s+surge\b does not match the plural), the same
        # singular->plural discipline applied to lahars/megafires. The two-word phrase denotes EXCLUSIVELY the
        # volcanic hazard — zero benign English meaning (unlike the polysemous bare "surge" — a power surge, a
        # surge of applause/adrenaline — which stays deliberately unfloored, left to the whole phrases
        # "storm surge"/"power surge") — so this closes the miss with no operational false-positive risk.
        # Surfaced in the 2026-08-27 volcanic-hazard rule-probe (density-current sibling of pyroclastic flow).
        # "pyroclastic density current"/"pyroclastic density currents" is the USGS/volcanology STANDARD umbrella
        # term (abbreviated PDC) for exactly the two phenomena already floored critical here — the dense
        # "pyroclastic flow" and the dilute "pyroclastic surge" — which modern hazard science treats as end-members
        # of a single continuum. A formal USGS/observatory hazard assessment reaches for "pyroclastic density
        # current" as the PREFERRED technical name ("PDCs are the deadliest volcanic hazard"), yet it matched
        # nothing and dropped to LOW: \bpyroclastic\s+flow\b and \bpyroclastic\s+surge\b do NOT match "pyroclastic
        # density current" (different second word), the SAME whole-hazard absent-term / formal-umbrella miss class
        # as "tropical cyclone" beside hurricane/typhoon — the identical lethal volcanic hazard scored
        # critical-or-LOW purely on which of the field's three interchangeable names the reporter used. Added both
        # forms at the volcanic critical floor beside "pyroclastic flow"/"pyroclastic surge"; the plural
        # "pyroclastic density currents" needs its own entry (\bpyroclastic\s+density\s+current\b does not match the
        # plural), the same singular->plural discipline applied to pyroclastic surges/lahars/megafires. The
        # three-word phrase denotes EXCLUSIVELY the volcanic hazard — zero benign English meaning (the bare
        # oceanographic "density current" — a gravity-driven sediment/salinity flow along a seabed — stays
        # deliberately unfloored, so a benign "estuary density current" sentence remains LOW) — so this closes the
        # miss with no operational false-positive risk. Surfaced in the 2026-08-27 volcanic-hazard rule-probe
        # (formal umbrella term of the already-critical pyroclastic flow + surge).
        # "limnic eruption"/"limnic eruptions" (a.k.a. a lake overturn) is the sudden release of a huge
        # dissolved-CO2 (or CH4) charge from the deep water of a stratified volcanic crater lake — a directly-
        # named, mass-casualty volcanic hazard on the same footing as the already-critical "pyroclastic flow"/
        # "lahar". It is the deadliest kind of volcanic gas event on record: the 1986 Lake Nyos (Cameroon) limnic
        # eruption released ~1.6 million tonnes of CO2 that flowed downslope as a denser-than-air blanket and
        # asphyxiated ~1,746 people and ~3,500 livestock in the valley below within minutes; Lake Monoun killed
        # ~37 in 1984. Yet bare "limnic eruption" matched nothing and dropped to LOW: \bvolcanic\s+eruption\b does
        # NOT match "limnic eruption" (different first word), no other floored token appears in a plain report of
        # the event, and the plural "limnic eruptions" is a distinct token (\blimnic\s+eruption\b does not match
        # the plural) — the SAME whole-hazard absent-term / singular->plural miss class as "pyroclastic surge"
        # beside "pyroclastic flow", a directly-named lethal volcanic catastrophe scored critical-or-LOW purely on
        # which named phenomenon the reporter cited. Added both forms at the volcanic critical floor beside
        # "pyroclastic flow"; the two-word phrase denotes EXCLUSIVELY the crater-lake gas burst — zero benign
        # English meaning ("limnic" is a technical limnology adjective that pairs with nothing benign here) — so
        # this closes the miss with no operational false-positive risk. Surfaced in the 2026-08-27 volcanic-hazard
        # rule-probe (crater-lake gas sibling of the already-critical volcanic-eruption/pyroclastic cluster;
        # verified LOW->CRITICAL by fault injection).
        #
        # "supereruption"/"supereruptions" name a VEI-8 volcanic super-eruption — the caldera-forming class that
        # ejects >1,000 km3 of material (Toba ~74,000 yrs ago, Yellowstone's Lava Creek). It is to "volcanic
        # eruption" exactly what the already-critical "megaquake" is to "earthquake" and "megatsunami" is to
        # "tsunami": a directly-named, strictly-worse magnitude compound denoting a continent-scale, mass-casualty
        # catastrophe, so flooring it CRITICAL can only ever be correct. Yet bare "supereruption" matched nothing
        # and dropped to LOW: \bvolcanic\s+eruption\b does NOT fire inside the closed compound "supereruption"
        # (different token, no space), no other floored token appears in a plain report of the event, and the
        # plural "supereruptions" is a distinct token (\bsupereruption\b does not match the plural) — the SAME
        # magnitude-compound + singular->plural miss class already closed for megaquake/megaquakes,
        # megatsunami/megatsunamis, and megafire/gigafire. The compound word denotes EXCLUSIVELY the catastrophic
        # eruption event (the geographic feature is "supervolcano", deliberately NOT floored here — it appears in
        # benign geology/tourism text like "Yellowstone is a supervolcano" and would over-fire; only the named
        # EVENT is floored, mirroring "volcano" vs "volcanic eruption"), so this closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-27 volcanic-magnitude rule-probe (VEI-8 sibling
        # one step above the already-critical volcanic-eruption cluster; verified LOW->CRITICAL by fault injection).
        #
        # "medicane"/"medicanes" (a portmanteau of "Mediterranean" + "hurricane") is the media/agency name for a
        # tropical-LIKE cyclone over the Mediterranean — a warm-core storm with a hurricane-style eye and eyewall,
        # the exact same phenomenon as its already-critical siblings "hurricane"/"typhoon"/"tropical cyclone", just
        # the regional name for the Mediterranean basin. It is a directly-named, mass-casualty catastrophe: Medicane
        # Ianos (2020) killed four in Greece, and Medicane Daniel (2023) drove the catastrophic rainfall that
        # collapsed two dams above Derna, Libya and killed ~11,000+ people, one of the deadliest weather disasters in
        # African history. Yet "medicane" matched nothing and dropped to LOW: the SAME regional-synonym-of-a-critical-
        # term miss as typhoon-beside-hurricane (NW-Pacific name) and bushfire-beside-wildfire (Commonwealth name),
        # the same event scored critical-or-LOW purely on which regional word the reporter reached for. Overseas
        # facilities, imported PDF templates, and Mediterranean-basin contractors write "medicane" as routinely as an
        # Atlantic report writes "hurricane" — exactly the international-report miss the taxonomy exists to close (the
        # stated typhoon rationale). Added both forms at the same critical floor as "hurricane"; the plural "medicanes"
        # needs its own entry (\bmedicane\b does not match "medicanes"), the same singular->plural tokenization
        # discipline applied to hurricanes/typhoons/derechos/lahars. "medicane" is a whole portmanteau word denoting
        # EXCLUSIVELY the Mediterranean tropical-like cyclone — zero benign English meaning — so this closes the miss
        # with no operational false-positive risk. Surfaced in the 2026-08-28 tropical-cyclone-synonym rule-probe.
        #
        # "bomb cyclone"/"bomb cyclones" is the NWS/media name for a rapidly intensifying extratropical cyclone
        # (bombogenesis: central pressure dropping >=24 mb in 24 h) — a directly-named winter-storm catastrophe on the
        # same footing as its critical siblings derecho/superstorm/tropical cyclone: the December 2022 bomb cyclone
        # ("Storm Elliott") drove the Buffalo blizzard that killed ~40+ people, and bomb cyclones routinely produce
        # blizzard whiteouts, hurricane-force winds, and coastal flooding. Yet it matched nothing and dropped to LOW.
        # The bare root "cyclone" was DELIBERATELY excluded above as polysemous ("cyclone fence" = chain-link fence,
        # "cyclone separator" = industrial dust collector); the QUALIFIER "bomb" removes that ambiguity entirely —
        # \bbomb\s+cyclone\b cannot match a cyclone fence/separator — so the qualified phrase closes the synonym gap
        # WITHOUT reopening the excluded polysemous root, the exact bare-vs-qualified discipline already drawn for
        # "cyclone" (excluded) vs "tropical cyclone" and tamponade (excluded) vs "cardiac tamponade". Added both at the
        # derecho/superstorm critical floor; the plural "bomb cyclones" needs its own entry (\bbomb\s+cyclone\b won't
        # match the plural), the same singular->plural discipline applied to tropical cyclones/derechos. The two-word
        # phrase denotes EXCLUSIVELY the meteorological catastrophe — zero benign English meaning — so this closes the
        # miss with no operational false-positive risk. Surfaced in the 2026-08-28 tropical-cyclone-synonym rule-probe.
        "critical": ["tornado", "tornadoes", "tornados", "hurricane", "hurricanes",
                     "typhoon", "typhoons",
                     "tropical cyclone", "tropical cyclones",
                     "medicane", "medicanes",
                     "bomb cyclone", "bomb cyclones",
                     "derecho", "derechos",
                     "superstorm", "superstorms",
                     "earthquake", "earthquakes",
                     "temblor", "temblors",
                     "megaquake", "megaquakes",
                     "flash flood", "flash floods", "wildfire", "wildfires",
                     "bushfire", "bushfires",
                     "megafire", "megafires",
                     "gigafire", "gigafires",
                     "firenado", "firenados", "firenadoes",
                     "tsunami", "tsunamis", "megatsunami", "megatsunamis", "severe storm warning",
                     "volcanic eruption", "supereruption", "supereruptions",
                     "pyroclastic flow", "pyroclastic surge",
                     "pyroclastic surges",
                     "pyroclastic density current", "pyroclastic density currents",
                     "limnic eruption", "limnic eruptions",
                     "lava flow",
                     "lahar", "lahars",
                     "storm surge"],
        # "haboob"/"haboobs" (Arabic for "blasting/drifting") is the meteorological/NWS name for an intense
        # wall-of-dust storm driven by thunderstorm-outflow winds — the dramatic Phoenix/Sahel dust storms that
        # drop highway visibility to zero and cause deadly multi-vehicle pileups (the 2011 Phoenix haboob; chronic
        # I-10 dust-storm pileups in Arizona). It is a directly-named severe weather hazard, a kind of storm no less
        # severe than the generic "storm" already floored HIGH, yet it matched nothing and dropped to LOW: as a
        # single loanword it shares no substring with "storm", so \bstorm\b cannot fire inside it — the SAME
        # regional/loanword tokenization miss as derecho/typhoon/medicane, one severity tier down. Floored at HIGH,
        # NOT critical: a haboob is a visibility/traffic/respiratory hazard rather than the guaranteed mass-casualty
        # catastrophe that warrants the CRITICAL floor (hurricane/derecho/tsunami); HIGH is the defensible floor a
        # human/LLM can raise, and one that injures a worker independently floors critical via injury/medical. Added
        # both forms at the "storm" HIGH floor; the plural "haboobs" needs its own entry (\bhaboob\b does not match
        # "haboobs"), the same singular->plural discipline applied to hurricanes/typhoons/derechos/lahars. "haboob"
        # is a whole loanword denoting EXCLUSIVELY the dust storm — zero benign English meaning — so this closes the
        # miss with no operational false-positive risk. Surfaced in the 2026-08-28 severe-weather rule-probe.
        #
        # "sandstorm"/"sandstorms" is the generic name for the exact hazard family the just-added "haboob" is an
        # intense subtype of — a wind-driven wall of sand/dust that drops highway visibility to zero and causes deadly
        # multi-vehicle pileups (the recurring I-10/I-40 desert-Southwest and Middle-East pileups; a respiratory and
        # flight/ops hazard). The SPACED "dust storm" already floors HIGH because \bstorm\b fires on its own "storm"
        # token, but the CLOSED compound "sandstorm" has no word boundary before "storm" (…d|storm…), so \bstorm\b
        # cannot fire inside it and it previously dropped to LOW — the SAME closed-compound tokenization miss as
        # superstorm-beside-storm (critical) and the loanword haboob one line up. Floored at HIGH, NOT critical, on the
        # identical rationale as haboob: a visibility/traffic/respiratory hazard rather than the guaranteed mass-casualty
        # catastrophe that warrants the CRITICAL floor (hurricane/derecho/tsunami); HIGH is the defensible floor a
        # human/LLM can raise, and one that injures a worker independently floors critical via injury/medical. Added both
        # forms at the "storm" HIGH floor beside haboob; the plural "sandstorms" needs its own entry (\bsandstorm\b does
        # not match "sandstorms"), the same singular->plural discipline applied to hurricanes/typhoons/derechos/haboobs.
        # "sandstorm" is a whole word denoting EXCLUSIVELY the meteorological event — zero benign English meaning in
        # facility/incident reporting — so this closes the miss with no operational false-positive risk. Surfaced in the
        # 2026-08-28 severe-weather closed-compound rule-probe.
        #
        # "snowstorm"/"snowstorms" is the direct sibling of the already-floored "ice storm" and "blizzard" (both HIGH) —
        # a winter storm that dumps snow and drives whiteout visibility, deadly highway pileups, roof-collapse loads, and
        # loss of access/power. The SPACED "ice storm" already floors HIGH via \bstorm\b on its own "storm" token, but
        # the CLOSED compound "snowstorm" has no word boundary before "storm" (…w|storm…), so \bstorm\b cannot fire
        # inside it — and \bsnow\b (the MEDIUM tier) has no boundary AFTER "snow" in "snowstorm" either, so neither floor
        # reached it and it previously dropped to LOW. This is the SAME closed-compound tokenization miss as
        # sandstorm-beside-storm and superstorm-beside-storm one/several lines up. Floored at HIGH, NOT critical, on the
        # identical rationale as its ice-storm/blizzard siblings: a winter-weather access/visibility/structural hazard
        # rather than the guaranteed mass-casualty catastrophe that warrants the CRITICAL floor (hurricane/derecho/tsunami);
        # HIGH is the defensible floor a human/LLM can raise, and one that injures a worker independently floors critical
        # via injury/medical. Added both forms at the "storm" HIGH floor beside sandstorm; the plural "snowstorms" needs
        # its own entry (\bsnowstorm\b does not match "snowstorms"), the same singular->plural discipline applied to
        # hurricanes/typhoons/derechos/haboobs/sandstorms. "snowstorm" is a whole word denoting EXCLUSIVELY the winter
        # storm — zero benign English meaning in facility/incident reporting — so this closes the miss with no operational
        # false-positive risk. Surfaced in the 2026-08-28 winter-weather closed-compound rule-probe.
        #
        # "windstorm"/"windstorms" is the generic directly-named severe-wind event — the closed-compound sibling
        # of the just-added "sandstorm"/"snowstorm" and the whole-word twin of the already-floored phrase "high
        # winds" (both HIGH): a wind event strong enough to down trees/lines, tear roofing/cladding, and topple
        # equipment (a straight-line-wind hazard one tier below the named-catastrophe derecho at CRITICAL). Two
        # existing HIGH signals SHOULD have caught it and neither can: the generic "storm" floors HIGH via
        # \bstorm\b, but the CLOSED compound "windstorm" has no word boundary before "storm" (…d|storm…), so
        # \bstorm\b cannot fire inside it; and "high winds" is a literal two-word phrase that does not match the
        # single token "windstorm" — so it previously dropped to LOW. This is the SAME closed-compound
        # tokenization miss as sandstorm/snowstorm-beside-storm and superstorm-beside-storm just above. Floored
        # at HIGH, NOT critical, on the identical rationale as its haboob/sandstorm/snowstorm siblings: a generic
        # damaging-wind hazard rather than the guaranteed mass-casualty catastrophe that warrants CRITICAL
        # (hurricane/derecho/tsunami); HIGH is the defensible floor a human/LLM can raise, and one that injures a
        # worker independently floors critical via injury/medical. Added both forms at the "storm"/"high winds"
        # HIGH floor beside snowstorm; the plural "windstorms" needs its own entry (\bwindstorm\b does not match
        # "windstorms"), the same singular->plural discipline applied to hurricanes/typhoons/derechos/haboobs/
        # sandstorms/snowstorms. "windstorm" is a whole word denoting EXCLUSIVELY the meteorological event — zero
        # benign English meaning in facility/incident reporting — so this closes the miss with no operational
        # false-positive risk. Surfaced in the 2026-08-28 severe-wind closed-compound rule-probe.
        "high":     ["storm", "lightning strike", "lightning struck", "struck by lightning",
                     "hail", "high winds", "fallen tree",
                     "haboob", "haboobs",
                     "sandstorm", "sandstorms",
                     "snowstorm", "snowstorms",
                     "windstorm", "windstorms",
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
