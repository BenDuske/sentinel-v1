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
        #
        # "levee failure"/"levee breach" (+ plurals) is the exact engineering twin of the already-
        # critical "dam failure" — the catastrophic loss of a flood-control embankment that releases
        # an uncontrolled inundation (the 2005 Katrina levee failures flooded ~80% of New Orleans and
        # killed over a thousand; the 2019 Midwest levee breaches inundated whole counties). Yet
        # "levee failure" matched NO floored token and dropped to LOW: \bdam\s+failure\b cannot match
        # "levee failure" (different first word), and neither "levee" nor "failure" is floored on its
        # own. WORSE, the singular "levee breach" hit the security/intrusion "breach" token (a
        # fence/perimeter breach) and floored only HIGH under the WRONG category — an active
        # under-floor + mis-category, the same worse-than-absent class as "storm tide" hitting bare
        # "storm". Added the whole phrases at the "dam failure" critical floor; each plural needs its
        # own entry (\blevee\s+failure\b won't match "levee failures", \blevee\s+breach\b won't match
        # "levee breaches"), the singular->plural discipline applied throughout the taxonomy. Both
        # two-word phrases denote EXCLUSIVELY the flood-control-structure catastrophe — "levee" has
        # zero benign meaning in facility/incident text (the archaic reception sense never appears) —
        # so this closes the miss with no operational false-positive risk (and the water/flood
        # critical floor now correctly outranks the spurious security "breach" hit). The British/
        # Dutch synonym "dike"/"dyke failure" is DELIBERATELY left for a separate probe (polysemous
        # spelling). Surfaced in the 2026-08-30 flood-control-structure rule-probe (twin of dam failure).
        #
        # "dam break"/"dam breach"/"dam burst" (+ plurals) are the plain-English / witness phrasings of the
        # exact catastrophe the noun "dam failure" already floors critical — the uncontrolled release of a
        # reservoir when the structure gives way (the 1889 Johnstown dam break killed ~2,200; the 2020 Edenville
        # dam breach forced 10,000 evacuations; the 1975 Banqiao dam burst killed tens of thousands). Yet the
        # engineering noun "dam failure" is the ONLY form that fired: \bdam\s+failure\b cannot match "dam break"/
        # "dam burst" (different second word) so both dropped to LOW, and — the same worse-than-absent class just
        # fixed for "levee breach" — the singular "dam breach" hit ONLY the security/intrusion "breach" token
        # (HIGH, WRONG category), actively under-flooring a reservoir-release catastrophe one tier LOW and
        # mis-labeling it a perimeter breach. Added the whole two-word phrases at the "dam failure" critical
        # floor; each plural needs its own entry (\bdam\s+break\b won't match "dam breaks", nor \bdam\s+breach\b
        # "dam breaches"), the singular->plural discipline applied throughout the taxonomy. All three denote
        # EXCLUSIVELY the dam catastrophe — "dam break"/"dam breach"/"dam burst" have zero benign meaning in
        # facility/incident text (the polysemous bare "dam" — beaver/log/cofferdam, "Amsterdam", "damn" — is
        # never floored on its own, only these adjacency phrases fire) — so this closes the miss with no
        # operational false-positive risk (and the water/flood critical floor now correctly outranks the spurious
        # security "breach" hit for "dam breach", exactly as it does for "levee breach"). The reordered/verb forms
        # "dam broke"/"burst dam" are DELIBERATELY left out (a "beaver dam broke" is a genuine but minor event the
        # CRITICAL floor would over-fire). Surfaced in the 2026-08-30 dam-catastrophe rule-probe (twin of dam
        # failure / levee breach).
        #
        # "levee break"/"levee breaks" is the plain-English / witness twin of the already-critical
        # "levee failure"/"levee breach" — the SAME flood-control-embankment give-way, in the words a
        # non-engineer reporter actually writes ("the levee broke and floodwater poured in", "a levee
        # break inundated the plant", "multiple levee breaks along the river"; the phrasing burned into
        # public memory by Katrina and "When the Levee Breaks"). It is the EXACT parallel of the
        # "dam break" plain-English twin just added beside "dam failure" — yet the levee cluster got
        # only "levee failure"/"levee breach" and this twin was missed, so "levee break" matched NO
        # floored token and dropped to LOW: \blevee\s+failure\b / \blevee\s+breach\b cannot match "levee
        # break" (different second word), and neither "levee" nor "break" is floored on its own. Added
        # both forms at the "levee failure"/"dam break" critical floor; the plural "levee breaks" needs
        # its own entry (\blevee\s+break\b won't match "levee breaks"), the singular->plural discipline
        # applied throughout. Both two-word phrases denote EXCLUSIVELY the flood-control-structure
        # catastrophe — "levee" has zero benign meaning in facility/incident text — so this closes the
        # miss with no operational false-positive risk. The reordered/verb form "levee broke" is
        # DELIBERATELY left out (the same discipline that excluded "dam broke"/"burst dam"). Also
        # completed the dam cluster's own plural coverage: "dam burst" floored but its plural "dam
        # bursts" did NOT (\bdam\s+burst\b won't match "dam bursts") — the identical singular->plural
        # miss, added beside it. Surfaced in the 2026-08-30 flood-control-structure rule-probe (plain-
        # English twin of the levee-failure / dam-break cluster).
        #
        # "dike"/"dyke" failure/breach/break (+ plurals) is the British/Dutch synonym of "levee" and
        # was DELIBERATELY deferred when the levee cluster landed (the note above at the levee entry
        # flags it) because the BARE word is polysemous — "dike"/"dyke" alone can mean an igneous
        # intrusion (geology) or is a slur, so flooring the bare token would over-fire. But the levee
        # discipline resolves this cleanly: it is the TWO-WORD phrases that carry the catastrophe, and
        # "dike failure"/"dike breach"/"dike break" (like "levee failure") have ZERO benign meaning as
        # phrases — a breached dike is the exact flood-control-embankment give-way that drowned SW
        # Netherlands in the 1953 North Sea flood (~1,800 dead) and is the standard term in Dutch/UK
        # flood reporting. So this adds ONLY the qualified phrases, both spellings, never the bare
        # word — the identical bare-token-excluded / two-word-phrase-floored discipline used for
        # "levee". Without this, "dike failure"/"dyke breach" matched NO floored token and dropped to
        # LOW: \blevee\s+failure\b etc. cannot match a different first word, and — the same worse-than-
        # absent class fixed for "levee breach"/"dam breach" — the singular "dike breach" hit ONLY the
        # security/intrusion "breach" token, mis-labeling a flood catastrophe a perimeter breach. Each
        # plural needs its own entry (\bdike\s+failure\b won't match "dike failures"), the singular->
        # plural discipline applied throughout. Mirrors the levee set (failure/breach/break, no
        # "burst" — "dike burst" is not the idiomatic phrasing). Surfaced in the 2026-08-30 flood-
        # control-structure rule-probe (British/Dutch synonym twin of the levee cluster).
        #
        # "floodwall" failure/breach/break (+ plurals) is the reinforced-concrete/steel flood-control
        # structure whose give-way IS the canonical modern flood catastrophe — the 2005 Katrina 17th
        # Street Canal and London Avenue Canal FLOODWALL failures (not the earthen levees) flooded ~80%
        # of New Orleans and killed ~1,800. It is a distinct engineered structure from the earthen
        # "levee"/"dike" (a vertical wall vs an embankment), so it needs its own token. Critically, this
        # is the CLOSED-compound tokenization miss (superstorm-beside-storm / sandstorm class), NOT the
        # spaced form: "flood wall failure" ALREADY floors critical because \bflood\b fires on its own
        # "flood" word — but the closed compound "floodwall" has no boundary after "flood"
        # (…flood|wall…), so \bflood\b cannot fire inside it, and neither "levee failure"/"dam failure"
        # matches (different first word). So "floodwall failure"/"floodwall break" dropped to LOW, and —
        # the same worse-than-absent under-floor+mis-category class fixed for levee/dam/dike "breach" —
        # "floodwall breach" hit ONLY the security/intrusion "breach" token (HIGH, WRONG category),
        # actively under-flooring a flood-control-structure catastrophe one tier LOW and mislabeling it a
        # perimeter breach. Added the closed-compound phrases at the "dam failure"/"levee failure"
        # critical floor; each plural needs its own entry (\bfloodwall\s+failure\b won't match
        # "floodwall failures"), the singular->plural discipline applied throughout. Mirrors the
        # levee/dike set (failure/breach/break, no "burst" — "floodwall burst" is not idiomatic).
        # "floodwall" is a whole compound with ZERO benign English meaning, so this closes the miss with
        # no operational false-positive risk. Surfaced in the 2026-08-30 flood-control-structure rule-
        # probe (closed-compound concrete twin of the levee/dike embankment cluster).
        "critical": ["flood", "flooding", "flooded", "floodwater", "floodwaters",
                     "submerged", "sewage backup", "burst main", "dam failure",
                     "dam break", "dam breaks", "dam breach", "dam breaches",
                     "dam burst", "dam bursts",
                     "levee failure", "levee failures", "levee breach", "levee breaches",
                     "levee break", "levee breaks",
                     "dike failure", "dike failures", "dike breach", "dike breaches",
                     "dike break", "dike breaks",
                     "dyke failure", "dyke failures", "dyke breach", "dyke breaches",
                     "dyke break", "dyke breaks",
                     "floodwall failure", "floodwall failures",
                     "floodwall breach", "floodwall breaches",
                     "floodwall break", "floodwall breaks"],
        # "ice jam"/"ice jams" is the directly-named NWS river-ice flood hazard — an accumulation of
        # broken ice that dams a river, backs water up over its banks, and can release in a sudden
        # destructive surge when it breaks (the NWS issues "Ice Jam Flood" warnings; ice-jam floods
        # kill people and inundate riverside facilities every winter/spring thaw). Yet a report writing
        # "an ice jam on the river is backing water toward the intake" reached NO floored token and
        # dropped to LOW: "ice jam" contains no "flood"/"flooding" substring (the water/flood critical
        # tokens can't fire), \bice storm\b (the weather-HIGH winter phrase) does not match "ice jam",
        # and neither "ice" nor "jam" is floored on its own — the SAME whole-hazard absent-term miss
        # class as "storm surge" beside "flood" and "levee failure" beside "dam failure". Added the
        # phrase at water/flood HIGH beside the general water hazards, NOT critical: if the report
        # actually says the jam IS flooding ("ice-jam flooding swamped the plant"), the bare
        # "flood"/"flooding" token independently escalates it to critical, so HIGH is the conservative
        # warning-stage floor for the jam named on its own. The plural "ice jams" needs its own entry
        # (\bice\s+jam\b does not match the trailing "s"), the singular->plural discipline applied
        # throughout the taxonomy. The two-word phrase denotes EXCLUSIVELY the river-ice hazard — the
        # polysemous "jam" (traffic jam / paper jam / preserves) never appears beside "ice" in that
        # benign sense, so this closes the miss with no operational false-positive risk. Surfaced in the
        # 2026-08-30 river-ice flood rule-probe (whole-hazard sibling of storm surge / dam failure).
        # "seiche"/"seiches" is the directly-named NOAA/NWS standing-wave flood hazard — a wind- or
        # pressure-driven oscillation of an enclosed or semi-enclosed body of water (a lake, bay, or
        # harbor) that sloshes back and forth, dropping the water at one shore and driving a sudden surge
        # up the other (the 1954 Lake Michigan seiche swept people off Chicago piers; Lake Erie seiches
        # routinely flood Buffalo/Toledo waterfronts by several feet). Yet a report writing "a seiche on
        # the lake pushed water over the intake berm" reached NO floored token and dropped to LOW:
        # "seiche" shares no substring with "flood"/"flooding"/"surge" (the water/flood critical tokens
        # can't fire), and it is not a substring of any weather or water token — the SAME whole-hazard
        # absent-term miss class as "storm surge" beside "flood" and "ice jam" beside the river hazards.
        # Floored at water/flood HIGH beside the general water hazards, NOT critical: if the report says
        # the seiche IS flooding ("seiche flooding inundated the dock"), the bare "flood"/"flooding" token
        # independently escalates it to critical, so HIGH is the conservative warning-stage floor for the
        # seiche named on its own. The plural "seiches" needs its own entry (\bseiche\b does not match the
        # trailing "s"), the singular->plural discipline applied throughout the taxonomy. "seiche" is a
        # technical limnology/oceanography term with ZERO benign or figurative English meaning (unlike the
        # deliberately-excluded polysemous "gale"/"tidal wave"), so it closes the miss with no operational
        # false-positive risk. Surfaced in the 2026-08-30 standing-wave flood rule-probe (whole-hazard
        # sibling of storm surge / ice jam).
        # "freshet"/"freshets" is the directly-named NWS/USGS river-flood hazard — the sudden rise and
        # overflow of a stream or river driven by heavy rain or spring snowmelt (NWS river forecasts warn
        # of the annual "spring freshet"; a freshet can overtop banks, scour intakes, and inundate
        # riverside facilities). Yet a report writing "a freshet on the river overtopped the intake
        # screens" reached NO floored token and dropped to LOW: "freshet" shares no substring with
        # "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire), and it is not a
        # substring of any weather or water token — the SAME whole-hazard absent-term miss class as
        # "storm surge" beside "flood", "ice jam" beside the river hazards, and "seiche" beside the coast.
        # Floored at water/flood HIGH beside the general water hazards, NOT critical: if the report says
        # the freshet IS flooding ("the freshet flooded the pump house"), the bare "flood"/"flooding"
        # token independently escalates it to critical, so HIGH is the conservative warning-stage floor for
        # the freshet named on its own. The plural "freshets" needs its own entry (\bfreshet\b does not
        # match the trailing "s"), the singular->plural discipline applied throughout the taxonomy.
        # "freshet" is a technical hydrology term with ZERO benign or figurative English meaning (unlike
        # the deliberately-excluded polysemous "gale"/"tidal wave"), so it closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-31 snowmelt/river-flood rule-probe
        # (whole-hazard sibling of storm surge / ice jam / seiche).
        # "king tide"/"king tides" is the directly-named NOAA coastal-flood hazard — the highest
        # predicted (perigean spring) tides of the year, which push seawater over low-lying shoreline
        # infrastructure and are the routine driver of "sunny-day"/nuisance coastal inundation (NOAA
        # runs a public "King Tides" reporting program precisely because they flood waterfront roads
        # and facilities on a clear day). Yet a report writing "a king tide is pushing seawater over
        # the intake berm" reached NO floored token and dropped to LOW: "king tide" shares no substring
        # with "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire), \bstorm\s+tide\b
        # (the critical hurricane-companion phrase) does not match "king tide" (different first word),
        # and neither "king" nor bare "tide" is floored on its own — the SAME whole-hazard absent-term
        # miss class as "storm surge" beside "flood", "ice jam" beside the river hazards, and
        # "seiche"/"freshet" beside the coast/river. Floored at water/flood HIGH beside the general
        # water hazards, NOT critical: if the report says the king tide IS flooding ("a king tide
        # flooded the low-lying yard"), the bare "flood"/"flooding" token independently escalates it to
        # critical, so HIGH is the conservative warning-stage floor for the king tide named on its own
        # (verified live: the bare-"flood" sentence already scores critical). The plural "king tides"
        # needs its own entry (\bking\s+tide\b does not match the trailing "s"), the singular->plural
        # discipline applied throughout the taxonomy. "king tide" is an established coastal-oceanography
        # term denoting EXCLUSIVELY this flood hazard — zero benign or figurative meaning as a phrase
        # (unlike the deliberately-excluded polysemous "gale"/"tidal wave") — so it closes the miss with
        # no operational false-positive risk. Surfaced in the 2026-08-31 coastal-flood rule-probe
        # (whole-hazard sibling of storm surge / storm tide / seiche / freshet).
        # "sneaker wave"/"sneaker waves" is the directly-named NWS/NOAA Pacific-coast life-threat hazard —
        # a sudden, disproportionately large and fast surge of water that rushes far up a beach or over
        # coastal rocks without warning, sweeping people off the shore and drowning them (the NWS issues
        # dedicated "Sneaker Wave" statements/advisories for the OR/WA/CA coast precisely because they
        # kill beachgoers and shoreline crews on otherwise calm days). Yet a report writing "a sneaker
        # wave swept a worker off the jetty" reached NO floored token and dropped to LOW: "sneaker wave"
        # shares no substring with "flood"/"flooding"/"surge" (the water/flood critical tokens can't
        # fire), and it is not a substring of any weather or water token — the SAME whole-hazard absent-
        # term miss class as "storm surge" beside "flood", "ice jam" beside the river hazards, and
        # "seiche"/"freshet"/"king tide" beside the coast/river. Floored at water/flood HIGH beside the
        # general water hazards, NOT critical: if the report says the wave IS flooding ("a sneaker wave
        # flooded the low pier") or sweeps a worker to their death, the bare "flood"/"flooding" or the
        # fatality/injury tokens independently escalate it to critical (both verified live), so HIGH is
        # the conservative warning-stage floor for the sneaker wave named on its own. The plural "sneaker
        # waves" needs its own entry (\bsneaker\s+wave\b does not match the trailing "s"), the singular->
        # plural discipline applied throughout the taxonomy. "sneaker wave" is an established coastal-
        # safety term denoting EXCLUSIVELY this hazard — zero benign or figurative meaning as a phrase
        # (unlike the deliberately-excluded polysemous "gale"/"tidal wave") — so it closes the miss with
        # no operational false-positive risk. Surfaced in the 2026-08-31 coastal-life-threat rule-probe
        # (whole-hazard sibling of storm surge / seiche / freshet / king tide).
        # "rip current"/"rip currents" is the directly-named NWS surf-zone life-threat hazard — a
        # narrow, powerful channel of water flowing swiftly away from shore through the surf zone that
        # drags swimmers and shoreline workers out to sea (the NWS issues dedicated "Rip Current
        # Statement" products for this hazard, which drowns more people on U.S. beaches than any other
        # surf danger). Yet a report writing "a rip current dragged a swimmer off the outfall apron"
        # reached NO floored token and dropped to LOW: "rip current" shares no substring with
        # "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire), and it is not a
        # substring of any weather or water token — the SAME whole-hazard absent-term miss class as
        # "storm surge" beside "flood", "ice jam" beside the river hazards, and
        # "seiche"/"freshet"/"king tide"/"sneaker wave" beside the coast/river. Floored at water/flood
        # HIGH beside the general water hazards, NOT critical: if the report says the current IS
        # flooding ("a rip current flooded the low apron") or the bare fatality/injury tokens fire, it
        # independently escalates to critical, so HIGH is the conservative warning-stage floor for the
        # rip current named on its own. The plural "rip currents" needs its own entry (\brip\s+current\b
        # does not match the trailing "s"), the singular->plural discipline applied throughout the
        # taxonomy. "rip current" is an established surf-safety term denoting EXCLUSIVELY this hazard —
        # zero benign or figurative meaning as a phrase, and note the bare polysemous "current"
        # (electrical current, current events, ocean current) is deliberately NOT floored, only the full
        # phrase — so it closes the miss with no operational false-positive risk. Surfaced in the
        # 2026-08-31 surf-zone-life-threat rule-probe (whole-hazard sibling of storm surge / seiche /
        # freshet / king tide / sneaker wave).
        # "high surf warning"/"high surf warnings" is the directly-named NWS coastal life-threat
        # product — issued for large, dangerous breaking waves and pounding shorebreak that sweep
        # people off jetties/piers/rocks, batter shoreline structures, and drown swimmers and workers
        # (routinely paired with the rip current statement below). Yet "a high surf warning is in
        # effect for the outfall jetty" reached NO floored token and dropped to LOW: the phrase shares
        # no substring with "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire),
        # "rip current"/"storm surge" are different words, and there is no bare "surf" or "warning"
        # token — the SAME whole-product absent-term miss class as the rip current statement and the
        # weather-side gale/red-flag/freeze warnings. Floored at water/flood HIGH beside rip current,
        # NOT critical: if the report says the waves ARE flooding ("high surf flooded the low apron")
        # or the bare fatality/injury tokens fire, it independently escalates to critical, so HIGH is
        # the conservative warning-stage floor for the product named on its own — the exact watch/
        # warning-vs-critical discipline used for rip current, gale warning, and the freeze/wind/heat
        # warnings. ONLY the full product phrase floors: the bare polysemous "high surf" (a surf-report
        # or recreational phrase — "the high surf was perfect for the surfers", "great high surf today")
        # is deliberately NOT floored, mirroring bare "gale"/"current"/"heat" left unfloored while only
        # their product phrases fire, so a beach-conditions mention stays LOW. Each plural needs its own
        # entry (\bhigh\s+surf\s+warning\b does not match the trailing "s"), the singular->plural
        # discipline applied throughout the taxonomy.
        # 2026-09-01 coastal-surf advisory follow-up (completes the high-surf warning/advisory ladder).
        # "high surf advisory"/"high surf advisories" is the ADVISORY-tier NWS coastal-surf product — one
        # NWS gradient below the HIGH "high surf warning": elevated/hazardous surf and strong shorebreak
        # that make the water dangerous for swimmers and small craft, but below the warning's life-threat
        # criteria. It floors water/flood MEDIUM (see the medium list below), the advisory->MEDIUM /
        # warning->HIGH gradient already applied to the heat (heat advisory MEDIUM / heat warning HIGH),
        # wind (wind advisory / high wind warning), and avalanche (avalanche watch MEDIUM / avalanche
        # warning HIGH) families. Named alone it dropped LOW: the phrase shares no substring with any
        # floored token (\bhigh\s+surf\s+warning\b is a different final word, "flood"/"surge" don't match,
        # bare "surf"/"advisory" are not tokens), the SAME whole-product absent-term miss as the warning.
        # ONLY the full three-word phrase floors — the bare polysemous "high surf" stays LOW (unchanged
        # FP guard), so a scattered "high surf ... advisory" stays LOW too (new adjacency FP guard). Each
        # plural is a distinct token (\bhigh\s+surf\s+advisory\b won't match the trailing "s"), the
        # singular->plural discipline applied throughout. Completes the warning-first/advisory-later split
        # deferred at the 2026-09-01 high-surf-warning ship (whole-product sibling of rip current / storm
        # surge / gale warning).
        # 2026-09-01 coastal-life-threat umbrella follow-up (caps the coastal-surf product family).
        # "beach hazards statement"/"beach hazards statements" is the directly-named NWS UMBRELLA coastal
        # life-threat product — the single statement the NWS issues to cover the surf-zone killers already
        # floored individually here (rip currents, high surf, sneaker waves, longshore/inlet currents):
        # "a beach hazards statement is in effect" tells shoreline workers and swimmers the water is
        # actively dangerous. Yet named alone it reached NO floored token and dropped to LOW: the phrase
        # shares no substring with "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire),
        # "rip current"/"high surf warning" are different words, and there is no bare "beach"/"hazard"/
        # "statement" token — the SAME whole-product absent-term miss as the rip current statement, high
        # surf warning, and the weather-side gale/red-flag/freeze warnings. Floored at water/flood HIGH
        # beside its member hazards, NOT critical: if the report says the water IS flooding, or the bare
        # fatality/injury tokens fire, those independently escalate to critical, so HIGH is the conservative
        # warning-stage floor for the product named on its own — the exact warning-vs-critical discipline
        # used for rip current / high surf warning / gale warning. ONLY the full product phrase floors: the
        # bare polysemous "beach" (a beach-vacation/recreation word — "the beach was crowded", "beach day")
        # is deliberately NOT floored, mirroring bare "surf"/"gale"/"current" left unfloored while only
        # their product phrases fire, so a beach-conditions mention stays LOW (FP guard). Note the canonical
        # NWS spelling pluralizes "Hazards" inside the product name; the whole-name plural "statements"
        # needs its own entry (\bbeach\s+hazards\s+statement\b does not match the trailing "s"), the
        # singular->plural discipline applied throughout the taxonomy. "beach hazards statement" is an
        # established NWS product name denoting EXCLUSIVELY this coastal life-threat advisory — so it closes
        # the miss with no operational false-positive risk (whole-product sibling of rip current / high
        # surf warning / sneaker wave / storm surge).
        # 2026-09-01 marine sea-state life-threat rule-probe (open-water peer of the coastal-surf family).
        # "hazardous seas" is the directly-named NWS OFFSHORE sea-state life-threat product — a Hazardous
        # Seas Warning covers large, steep combined seas (heights above the marine threshold, e.g. ~10 ft+,
        # or dangerously steep short-period waves) that capsize vessels, sweep crew overboard, and swamp
        # small craft in open water, the mariner-facing sibling of the shoreline "high surf warning" already
        # floored HIGH and the operational partner of the weather-side "gale warning"/"storm warning". Yet
        # named alone it reached NO floored token and dropped to LOW (verified live): "hazardous seas" shares
        # no substring with "flood"/"flooding"/"surge" (the water/flood critical tokens can't fire), the HIGH
        # coastal phrases "high surf warning"/"sneaker wave"/"rip current" are different words, and there is
        # no bare "seas"/"hazardous"/"warning" token — the SAME whole-product absent-term miss as the beach
        # hazards statement, rip current statement, high surf warning, and the weather-side gale/freeze
        # warnings. Floored at water/flood HIGH beside its coastal-surf peers, NOT critical: a forecastable,
        # mitigable marine hazard (alter course, seek harbor, secure the deck); if the report says the water
        # IS flooding, or the bare fatality/overboard-injury tokens fire, those independently escalate to
        # critical, so HIGH is the conservative warning-stage floor for the product named on its own — the
        # exact warning-vs-critical discipline used for high surf warning / beach hazards statement / gale
        # warning. Unlike the recreational "high surf" (deliberately left LOW because surfers prize it), the
        # phrase "hazardous seas" carries ZERO benign meaning — no one names calm safe water "hazardous seas"
        # — so the bare two-word phrase floors directly, the same unambiguous-phrase discipline that floored
        # "storm surge"/"atmospheric river"/"gale-force winds" as bare phrases while their polysemous single
        # words stayed unfloored. One entry suffices: \bhazardous\s+seas\b already fires inside the product
        # names "hazardous seas warning"/"hazardous seas warnings" (the \b sits before the trailing space),
        # so the warning + plural forms are covered without separate entries — the bare-phrase-covers-suffix
        # economy of "storm surge" (which also floors "storm surge warning"). Surfaced in the 2026-09-01
        # marine sea-state rule-probe (open-water sibling of high surf warning / beach hazards statement /
        # sneaker wave / storm surge).
        "high":     ["water damage", "burst pipe", "pipe burst", "leak", "leaking",
                     "ice jam", "ice jams",
                     "seiche", "seiches",
                     "freshet", "freshets",
                     "king tide", "king tides",
                     "sneaker wave", "sneaker waves",
                     "rip current", "rip currents",
                     "high surf warning", "high surf warnings",
                     "beach hazards statement", "beach hazards statements",
                     "hazardous seas",
                     "standing water", "ceiling collapse from water", "overflow"],
        "medium":   ["drip", "dripping", "damp", "moisture", "condensation", "minor leak",
                     "high surf advisory", "high surf advisories"],
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
        # "debris flow"/"mudflow" (+ plurals) are the water-saturated flowing members of the same
        # earth-movement family as the already-HIGH "rockslide" — a fast torrent of mud, rock, and
        # debris racing down a slope that buries roads, right-of-ways, and structures (the 2018
        # Montecito debris flows killed 23; post-wildfire debris flows are a chronic Western hazard).
        # They name the EXACT hazard the polysemous "mudslide"/"landslide" were DELIBERATELY excluded
        # for (see the rockslide note above): "mudslide" couldn't be floored because the word is a
        # cocktail/ice-cream and "landslide" because of "landslide victory" — but the technical
        # synonyms "debris flow"/"mudflow" have ZERO benign English meaning, so they close that same
        # miss cleanly. Both previously matched nothing and dropped to LOW: "debris flow" contains no
        # floored substring (bare "debris"/"flow" are unfloored; \blava\s+flow\b / \bpyroclastic\s+flow\b
        # can't match a different first word), and "mudflow" is a single compound sharing no substring
        # with any floored token. Added at the SAME conservative HIGH earth-failure floor as rockslide
        # (a debris flow ranges from a blocked road to a fatal burial; one that buries or injures a
        # worker independently floors critical via injury/medical). Each plural is its own token
        # (\bmudflow\b won't match "mudflows", \bdebris\s+flow\b won't match "debris flows"), the same
        # singular->plural discipline as rockslide/rockslides and sinkhole/sinkholes. The volcanic
        # debris flow "lahar" already floors CRITICAL over in weather; these are the generic
        # non-volcanic hazard. Surfaced in the 2026-08-30 earth-movement rule-probe.
        # A "rockfall" — individual rock or a mass free-falling, bouncing, and rolling down a cliff,
        # highwall, or road cut onto whatever is below — is the FREE-FALL member of the same
        # earth-movement family as the already-HIGH "rockslide" (a sliding coherent mass): a directly
        # named geotechnical/USGS/DOT hazard a reporter writes verbatim ("a rockfall struck the haul
        # road", "rockfall onto the rail line trapped a crew"), yet it matched nothing and dropped to
        # LOW while its siblings sinkhole/cave-in/rockslide/debris flow all floor at HIGH — the SAME
        # whole-hazard absent-term miss class, one failure-mode over. Added at the SAME conservative
        # HIGH earth-failure floor as rockslide (a rockfall ranges from a few pebbles on an empty road
        # to a fatal strike, so HIGH is the defensible floor the LLM/human can raise — and one that
        # buries or injures a worker independently floors critical via the injury/medical terms). The
        # plural "rockfalls" needs its own entry (\brockfall\b won't match "rockfalls"), the same
        # singular->plural tokenization discipline as rockslide/rockslides and sinkhole/sinkholes.
        # DELIBERATELY the ONE-WORD "rockfall"/"rockfalls" ONLY — NEVER the spaced "rock fall" (bare
        # "rock" and "fall" are unfloored, and "prices fall"/"a rock could fall" are routine benign
        # usages); \brockfall\b as one word has zero benign English meaning, so it closes the miss with
        # zero false-positive risk. Surfaced in the 2026-08-30 earth-movement rule-probe (free-fall
        # sibling of the rockslide/debris-flow cluster).
        "high":     ["crack in wall", "structural crack", "sagging", "buckling", "subsidence",
                     "sinkhole", "sinkholes", "cave-in", "cave-ins", "rockslide", "rockslides",
                     "rockfall", "rockfalls",
                     "debris flow", "debris flows", "mudflow", "mudflows",
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
        # "storm tide" is the NWS/NHC companion measure to storm surge — the TOTAL observed water level
        # during a storm (storm surge + the astronomical tide), i.e. the actual coastal-inundation depth
        # that overtops seawalls and drowns low-lying sites. It is historically the deadliest coastal
        # hazard (Katrina/Ike/Camille), a directly-named catastrophe on exactly the same footing as its
        # already-critical twin "storm surge". Yet it scored only HIGH, not critical: it contains the bare
        # word "storm" (weather HIGH floor), so the phrase matched \bstorm\b and floored one level LOW of
        # the surge it accompanies — the SAME active UNDER-floor as "storm surge" before it was added, WORSE
        # than a pure absent-term miss. Added the whole phrase at the storm-surge/hurricane critical floor;
        # the plural "storm tides" needs its own entry (\bstorm\s+tide\b won't match "storm tides"), the same
        # singular->plural tokenization discipline applied throughout this list. "storm tide" has ZERO benign
        # English meaning (unlike the polysemous half "tide" — a red tide, a rising tide of demand, tide
        # detergent — which is DELIBERATELY EXCLUDED; only the adjacent two-word storm-tide phrase fires),
        # so it closes the miss with no operational false-positive risk (bare "storm" continues to floor
        # HIGH on its own). Surfaced in the 2026-08-29 coastal-flood rule-probe alongside storm surge.
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
        #
        # "bombogenesis" is the METEOROLOGICAL PROCESS NAME for the exact event "bomb cyclone" denotes — the >=24 mb/24 h
        # central-pressure crash that IS a bomb cyclone (the term is literally named in the bomb-cyclone comment above as
        # its definition) — yet it was never added as a token, so the SAME storm scored critical when a reporter wrote
        # "bomb cyclone" and dropped to LOW when a forecaster/LLM wrote "the system underwent rapid bombogenesis": the
        # identical whole-event synonym miss as typhoon-beside-hurricane and temblor-beside-earthquake, the same
        # catastrophe scored critical-or-LOW purely on which name it reached for. Forecast discussions and imported NWS
        # AFDs use "bombogenesis"/"bombogenetic" as the routine technical term, so it is exactly the miss the taxonomy
        # exists to close. Added at the bomb-cyclone critical floor. It is a coined single word (bomb + cyclogenesis) with
        # ZERO benign English meaning and no floored substring (bare "bomb"/"cyclone" both deliberately excluded, neither
        # a substring of it), so it fires only on the meteorological event. It is an UNCOUNTABLE process noun (like the
        # already-critical "flooding") — a report writes "rapid bombogenesis", never "two bombogeneses" — so unlike the
        # countable bomb-cyclone/derecho it gets NO artificial plural entry. Surfaced in the 2026-08-29 bomb-cyclone-
        # synonym rule-probe.
        #
        # "cyclonic storm" is the North Indian Ocean (Bay of Bengal / Arabian Sea) REGIONAL NAME for the SAME tropical
        # cyclone that is "hurricane" in the Atlantic, "typhoon" in the NW Pacific, and "medicane" in the Mediterranean —
        # it is the India Meteorological Department's OFFICIAL category name, escalating "cyclonic storm" -> "severe" ->
        # "very severe" -> "extremely severe" -> "super cyclonic storm" (Amphan 2020 was a super cyclonic storm; the 1970
        # Bhola and 1999 Odisha cyclonic storms each killed tens of thousands). Yet the phrase matched NO critical token
        # and, WORSE than a pure absent-term miss, ACTIVELY UNDER-FLOORED to HIGH: the bare word "storm" sits at the
        # weather HIGH floor, so "cyclonic storm" hit \bstorm\b and floored one level LOW of the hurricane it IS — the
        # exact storm-surge-matching-"storm" under-floor class. This completes the regional tropical-cyclone-name set
        # (hurricane/typhoon/tropical cyclone/medicane already critical); international incident reports from Indian /
        # Bangladeshi facilities and imported IMD bulletins routinely write "cyclonic storm". Added the whole two-word
        # phrase at the hurricane/typhoon critical floor — a single "cyclonic storm" token also covers the qualified
        # escalation names ("super/severe/very severe cyclonic storm" all carry \bcyclonic\s+storm\b as a substring). The
        # plural "cyclonic storms" needs its own entry (\bcyclonic\s+storm\b won't match the trailing "s"), the same
        # singular->plural discipline applied to tropical cyclones/typhoons/derechos. The two-word phrase denotes
        # EXCLUSIVELY the meteorological catastrophe — the adjective "cyclonic" beside "storm" has ZERO benign English
        # meaning (the polysemous industrial senses are "cyclonic separator"/"cyclonic vacuum", never "cyclonic storm"),
        # so like the bare-"cyclone"-excluded/"tropical cyclone"-added discipline it closes the miss with no operational
        # false-positive risk. Surfaced in the 2026-08-29 tropical-cyclone-regional-name rule-probe.
        "critical": ["tornado", "tornadoes", "tornados", "hurricane", "hurricanes",
                     "typhoon", "typhoons",
                     "tropical cyclone", "tropical cyclones",
                     "cyclonic storm", "cyclonic storms",
                     "medicane", "medicanes",
                     "bomb cyclone", "bomb cyclones", "bombogenesis",
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
                     "storm surge",
                     "storm tide", "storm tides"],
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
        # "duststorm"/"duststorms" is the one-word variant spelling of the "dust storm" the just-added "sandstorm" is
        # the sand-specific twin of — the identical wind-driven wall-of-dust hazard (zero-visibility highway pileups,
        # a respiratory/ops hazard). The SPACED "dust storm" already floors HIGH because \bstorm\b fires on its own
        # "storm" token, but the CLOSED compound "duststorm" has no word boundary before "storm" (…t|storm…), so
        # \bstorm\b cannot fire inside it and no other token is a substring of it — so it previously dropped to LOW
        # while the same event written "dust storm" or "sandstorm" floors HIGH. This is the EXACT closed-compound
        # tokenization miss already closed for sandstorm/snowstorm/windstorm-beside-storm. Floored at HIGH, NOT
        # critical, on the identical rationale as its sandstorm sibling: a visibility/traffic/respiratory hazard a
        # human/LLM can raise, not the guaranteed mass-casualty catastrophe warranting CRITICAL (a duststorm that
        # injures a worker independently floors critical via injury/medical). Added both forms at the "storm" HIGH
        # floor beside sandstorm; the plural "duststorms" needs its own entry (\bduststorm\b does not match
        # "duststorms"), the same singular->plural discipline applied to sandstorms/snowstorms/windstorms.
        # "duststorm" is a whole word denoting EXCLUSIVELY the meteorological event — zero benign English meaning in
        # facility/incident reporting — so this closes the miss with no operational false-positive risk. Surfaced in
        # the 2026-08-30 severe-weather closed-compound spelling-variant rule-probe.
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
        # "lake-effect snow"/"lake effect snow" names the banded localized heavy-snow regime — feet of snow in
        # hours off a warm lake, whiteout visibility, roof-collapse loads, and days of lost access (Buffalo Nov
        # 2014 ~7 ft / 13 dead; Nov 2022 ~6+ ft). It is the severe named event on the footing of its HIGH siblings
        # snowstorm/blizzard/ice storm, yet it previously scored only MEDIUM — an UNDER-FLOOR inversion identical in
        # class to "heat wave" scoring beneath the MEDIUM "heat advisory": bare \bsnow\b (the MEDIUM tier) DOES fire
        # on the trailing "snow" word, so the feet-of-snow banded event scored the SAME MEDIUM as a routine dusting,
        # while \bstorm\b cannot fire (no "storm" substring) and no HIGH winter token (snowstorm/blizzard/ice storm)
        # is a substring of it. Floored at HIGH beside snowstorm/blizzard, NOT critical, on the identical rationale
        # as those siblings: a winter-weather access/visibility/structural hazard rather than the guaranteed
        # mass-casualty catastrophe that warrants the CRITICAL floor (hurricane/derecho/tsunami); HIGH is the
        # defensible floor a human/LLM can raise, and one that injures a worker independently floors critical via
        # injury/medical. Both the hyphenated ("lake-effect snow", tokenized \blake\-effect\s+snow\b) and spaced
        # ("lake effect snow", \blake\s+effect\s+snow\b) spellings are added — the matcher treats the hyphen
        # literally, so the spaced form needs its own entry, the same both-spellings discipline as nor'easter/
        # noreaster. No plural entry: like the already-HIGH "freezing rain"/"black ice" and the MEDIUM "snow" it is
        # a mass noun ("lake-effect snows" does not occur in incident text). The bare "snow" MEDIUM hit still shows
        # alongside the new HIGH hit (both auditable, HIGH wins) — zero benign English meaning in facility/incident
        # reporting, so this closes the under-floor with no operational false-positive risk. Surfaced in the
        # 2026-08-30 winter-weather under-floor rule-probe.
        #
        # "avalanche warning"/"snow avalanche" name the snow-mass-movement severe hazard — a slope of snow/ice
        # (and entrained rock/debris) releasing and racing downslope at highway speed, burying roads/rail/right-of-
        # ways and worksites and killing by burial/trauma/asphyxia (backcountry, mountain highways, and slope-adjacent
        # construction/mining/utility sites). It is the snow sibling of the earth-mass-movement HIGH structural
        # tokens rockslide/debris flow/mudflow and the winter-severe-weather peer of snowstorm/blizzard/lake-effect
        # snow. Yet the NWS/avalanche-center product "an avalanche warning is in effect" reached NO floored token and
        # dropped to LOW (verified live), and the physical "a snow avalanche buried the access road" scored only
        # MEDIUM off bare \bsnow\b — the SAME under-floor as lake-effect snow scoring at bare-snow MEDIUM. Floored
        # HIGH (beside snowstorm/blizzard/lake-effect snow and the warning-grade freeze warning / high wind warning /
        # red flag warning), NOT critical: a slope-access/burial hazard a human/LLM can raise; a worker actually
        # buried or struck independently floors critical via injury/medical. Floored ONLY as the two qualified,
        # zero-polysemy phrases — the NWS product "avalanche warning" and the explicitly-physical "snow avalanche".
        # The bare root "avalanche"/"avalanches" is DELIBERATELY EXCLUDED as figurative-polysemous: "an avalanche of
        # emails / support tickets / paperwork / complaints" is routine ops/facilities language, so \bavalanche\b
        # would over-fire on benign text — the same qualified-phrase discipline that floored "gale-force winds"/"gale
        # warning" while leaving the polysemous bare "gale" (a name / "a gale of laughter") unfloored, and that kept
        # bare "landslide"/"mudslide" out while their zero-benign twins debris flow/mudflow shipped. A new FP-guard
        # test (test_bare_avalanche_figurative_stays_low) re-verifies "an avalanche of {emails,support tickets,
        # paperwork}" stays LOW. No plural entry: "avalanche warning" is a product name and "snow avalanche" a mass
        # phrase (neither pluralizes in incident text); the advisory-grade "avalanche watch" was a deferred follow-up
        # (the watch->MEDIUM / warning->HIGH sibling, out of scope that one-tier ship) — NOW SHIPPED at weather MEDIUM
        # (see the "avalanche watch"/"avalanche watches" rationale on the medium list). Surfaced in the 2026-09-01
        # 5:1x AM whole-hazard-absent rule-probe (a major named natural hazard entirely missing from the taxonomy).
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
        #
        # "downburst"/"microburst"/"macroburst" (and plurals) are the NWS/AMS-named DOWNDRAFT severe-wind
        # hazards — a strong column of sinking air that hits the ground and spreads out as damaging
        # straight-line winds (a microburst is a downburst <4 km across, a macroburst >4 km; both can exceed
        # 100 mph and have brought down aircraft and roofs). They are the closed-compound siblings of the
        # just-added "windstorm": the SAME severe-wind class one tier below the named-catastrophe derecho at
        # CRITICAL, floored at HIGH on the identical windstorm/haboob rationale (a damaging-wind hazard, not
        # the guaranteed mass-casualty catastrophe that warrants CRITICAL). Every HIGH severe-wind signal that
        # SHOULD have caught them cannot: "high winds" is a literal two-word phrase that does not match the
        # single token; \bwind\b cannot fire because there is no "wind" token in "…burst"; \bstorm\b cannot
        # fire because there is no "storm" token either; and the flood-taxonomy phrase \bburst\b ("burst pipe"/
        # "burst main") cannot fire inside the CLOSED compound "downburst" (…n|burst…, no word boundary before
        # "burst") — so they previously dropped to LOW. This is the SAME closed-compound tokenization miss as
        # windstorm/sandstorm/snowstorm-beside-storm. Each plural needs its own entry (\bdownburst\b does not
        # match "downbursts"), the same singular->plural discipline applied to windstorms/hurricanes/derechos.
        # Each "…burst" here denotes EXCLUSIVELY the meteorological downdraft event — zero benign English
        # meaning in facility/incident reporting — so this closes the miss with no operational false-positive
        # risk. Surfaced in the 2026-08-28 severe-wind closed-compound rule-probe (downdraft sibling sweep).
        #
        # "thunderstorm"/"thunderstorms" and "hailstorm"/"hailstorms" are the precipitation-storm closed
        # compounds — the remaining directly-named "-storm" severe-weather events that the generic HIGH
        # signals SHOULD catch and cannot. "thunderstorm" is the canonical NWS-warned storm ("severe
        # thunderstorm warning": damaging straight-line wind, hail, deadly cloud-to-ground lightning); it is
        # LITERALLY a storm, no less severe than the generic "storm" already floored HIGH, yet the CLOSED
        # compound has no word boundary before "storm" (…r|storm…) so \bstorm\b cannot fire inside it and it
        # previously dropped to LOW. "hailstorm" is the direct twin of the already-HIGH "hail": a barrage of
        # hail heavy enough to shatter skylights/glazing, dent equipment, and injure workers — but \bhail\b
        # cannot fire because there is no word boundary AFTER "hail" in "hailstorm" (hail|storm, "s" follows),
        # and \bstorm\b cannot fire before "storm" either, so it too dropped to LOW. This is the SAME
        # closed-compound tokenization miss as sandstorm/snowstorm/windstorm-beside-storm and hail-inside-
        # hailstorm mirrors snow-inside-snowstorm exactly. Floored at HIGH, NOT critical, on the identical
        # rationale as the other "-storm" siblings: a damaging severe-weather hazard rather than the
        # guaranteed mass-casualty catastrophe that warrants CRITICAL (hurricane/derecho/tsunami); HIGH is the
        # defensible floor a human/LLM can raise, and one that injures a worker independently floors critical
        # via injury/medical. Added both forms of each at the "storm"/"hail" HIGH floor beside macroburst; each
        # plural needs its own entry (\bthunderstorm\b does not match "thunderstorms"), the same singular->plural
        # discipline applied to windstorms/sandstorms/snowstorms. Each is a whole word denoting EXCLUSIVELY the
        # meteorological event — zero benign English meaning in facility/incident reporting — so this closes the
        # miss with no operational false-positive risk. Surfaced in the 2026-08-28 precipitation-storm
        # closed-compound rule-probe.
        #
        # "squall"/"squalls" is the NWS/AMS-named severe-convective hazard — a sudden violent burst of
        # wind (typically with rain, hail, or snow), and, as a "squall line", the linear multicell storm
        # system that produces damaging straight-line winds, large hail, and embedded tornadoes (a squall
        # line is the parent structure a derecho grows out of). It is a directly-named severe-weather
        # event no less severe than the generic "storm" already floored HIGH, yet it matched nothing and
        # dropped to LOW: a bare loanword-style single token that shares no substring with "storm" (so
        # \bstorm\b cannot fire inside it) and is absent from the list entirely — the SAME absent-term
        # miss class as haboob/typhoon, one tier below the named-catastrophe derecho at CRITICAL. Floored
        # at HIGH, NOT critical, on the identical rationale as its haboob/windstorm/thunderstorm siblings:
        # a damaging severe-convective hazard rather than the guaranteed mass-casualty catastrophe that
        # warrants the CRITICAL floor (hurricane/derecho/tsunami); HIGH is the defensible floor a human/LLM
        # can raise, and one that injures a worker independently floors critical via injury/medical. Added
        # the bare "squall" plus the plural "squalls" (\bsquall\b does not match "squalls"), the same
        # singular->plural discipline applied to hurricanes/typhoons/derechos/haboobs/thunderstorms. NO
        # separate "squall line" entry is needed: \bsquall\b already fires on the "squall" token inside the
        # phrase "squall line"/"squall lines" (the space is a word boundary), so both the bare event and the
        # squall-line system are covered by the two entries. "squall" is a whole word denoting EXCLUSIVELY
        # the meteorological event — zero benign English meaning in facility/incident reporting — so this
        # closes the miss with no operational false-positive risk. Surfaced in the 2026-08-28
        # severe-convective rule-probe.
        #
        # "rainstorm"/"rainstorms" is the last of the directly-named "-storm" precipitation compounds the
        # generic HIGH signals SHOULD catch and cannot. It is LITERALLY a storm — the word "storm" sits right
        # inside it — and a heavy rainstorm brings the flash-flooding, washed-out access, and lightning that
        # make it no less severe than the generic "storm" already floored HIGH; yet the CLOSED compound has no
        # word boundary before "storm" (…n|storm…) so \bstorm\b cannot fire inside it, and \brain\b cannot fire
        # either because there is no word boundary after "rain" (rain|storm, "s" follows) — the SAME closed-
        # compound tokenization miss as snowstorm/windstorm/thunderstorm-beside-storm, and rain-inside-rainstorm
        # mirrors snow-inside-snowstorm/hail-inside-hailstorm exactly. It previously dropped to LOW (the spaced
        # phrasing "heavy rain" only reaches the weather MEDIUM tier, so a report writing the single closed word
        # "rainstorm" scored strictly LOWER than the same event written "storm"). Floored at HIGH, NOT critical,
        # on the identical rationale as the other "-storm" siblings: a damaging severe-weather hazard rather than
        # the guaranteed mass-casualty catastrophe that warrants CRITICAL (hurricane/derecho/tsunami); HIGH is
        # the defensible floor a human/LLM can raise, and one that injures a worker independently floors critical
        # via injury/medical. Added both forms at the "storm" HIGH floor beside squall/thunderstorm; the plural
        # "rainstorms" needs its own entry (\brainstorm\b does not match "rainstorms"), the same singular->plural
        # discipline applied to thunderstorms/snowstorms/squalls. "rainstorm" is a whole word denoting
        # EXCLUSIVELY the meteorological event — zero benign English meaning in facility/incident reporting — so
        # this closes the miss with no operational false-positive risk. Surfaced in the 2026-08-28
        # precipitation-storm closed-compound rule-probe (rain-sibling sweep).
        #
        # "cloudburst"/"cloudbursts" is the rain-sibling of rainstorm the same sweep left uncovered: a
        # SUDDEN, VIOLENT torrential downpour (the AMS/NWS term for an extreme short-duration rainfall,
        # historically the trigger of the deadliest flash floods — the 2013 Kedarnath and 2022 Pakistan
        # cloudbursts each killed thousands). It is a directly-named severe-precipitation hazard on the
        # same footing as rainstorm/thunderstorm, yet the CLOSED compound reaches NO floored token: there
        # is no "storm" in it so \bstorm\b cannot fire, no word boundary after "cloud" (cloud|burst) and
        # no "rain" token, and the flood-taxonomy phrases "burst pipe"/"pipe burst"/"burst main" cannot
        # fire inside the closed word (…d|burst, no boundary before "burst" and no adjacent pipe/main) —
        # the SAME closed-compound tokenization miss as downburst-beside-burst and rainstorm-beside-storm,
        # so a report writing "a cloudburst overwhelmed the drains" previously dropped to LOW while the
        # spaced "heavy rain" only reaches MEDIUM. Floored at HIGH, NOT critical, on the identical rationale
        # as the other precipitation siblings: a damaging severe-weather hazard a human/LLM can raise, not
        # the guaranteed mass-casualty catastrophe that warrants CRITICAL, and a cloudburst that injures a
        # worker or floods the site independently floors higher via injury/medical or water/flood. Added
        # both forms beside rainstorm; the plural "cloudbursts" needs its own entry (\bcloudburst\b does not
        # match "cloudbursts"), the same singular->plural discipline applied to rainstorms/thunderstorms.
        # "cloudburst" is a whole word denoting EXCLUSIVELY the meteorological deluge — zero benign English
        # meaning in facility/incident reporting — so this closes the miss with no operational false-positive
        # risk. Surfaced in the 2026-08-29 precipitation-storm closed-compound rule-probe (rain-sibling sweep).
        #
        # "monsoon"/"monsoons" is the bare absent-loanword severe-weather hazard the closed-compound sweep
        # never reaches — the direct sibling of haboob/typhoon/squall (a directly-named meteorological event
        # adopted into English, one tier below the named-catastrophe derecho/hurricane at CRITICAL). A monsoon
        # is the seasonal reversal that drives the violent convective downpours, flash floods, dust storms, and
        # microbursts of the US desert Southwest ("monsoon flooding closed I-10"; the Arizona monsoon routinely
        # kills via flash floods and dust-storm pileups) and the deadly South-Asian rains — an acute severe-
        # weather hazard no less severe than the generic "storm" already floored HIGH. Yet it matched nothing
        # and dropped to LOW: as a single loanword token it shares no substring with any floored signal — no
        # "storm" so \bstorm\b cannot fire, no "rain"/"wind" token, and the spaced "heavy rain" it brings only
        # reaches the weather MEDIUM tier — the SAME bare absent-term miss class as haboob/typhoon/squall.
        # Floored at HIGH, NOT critical, on the identical rationale as its haboob/squall siblings: a damaging
        # severe-weather hazard a human/LLM can raise, not the guaranteed mass-casualty catastrophe that
        # warrants CRITICAL (hurricane/derecho/tsunami), and a monsoon that injures a worker or floods the site
        # independently floors higher via injury/medical or water/flood. Added the bare "monsoon" plus the
        # plural "monsoons" (\bmonsoon\b does not match "monsoons"), the same singular->plural discipline applied
        # to haboobs/typhoons/squalls. "monsoon" is a whole word denoting EXCLUSIVELY the meteorological event —
        # zero benign common-noun meaning in facility/incident reporting — so this closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-29 severe-weather bare-loanword rule-probe.
        # "waterspout"/"waterspouts" is a directly-named NWS marine severe-weather hazard — a tornado over water
        # (a rotating column that spins up under a convective cloud and can move ashore as a landspout tornado).
        # The NWS issues Special Marine Warnings for them and they routinely capsize boats, tear off roofs at the
        # coast, and injure people. It is a named severe-weather event on the same footing as its HIGH storm-family
        # siblings (squall/haboob/monsoon), yet as a single closed compound it reaches no floored token: it has no
        # "storm" so \bstorm\b cannot fire, no "rain"/"wind"/"hail" token, and "spout" is not floored — so it
        # previously dropped to LOW, the SAME bare-token miss class as haboob/monsoon-beside-storm. Floored at HIGH,
        # NOT critical, on the identical rationale as its squall/microburst siblings: a typically short-lived,
        # over-water/coastal wind hazard a human/LLM can raise, not the guaranteed mass-casualty catastrophe that
        # warrants CRITICAL (its stronger land cousin "tornado" already floors critical, and a waterspout that
        # injures a worker or floods the site independently floors higher via injury/medical or water/flood). Added
        # the bare "waterspout" plus the plural "waterspouts" (\bwaterspout\b does not match "waterspouts"), the
        # same singular->plural discipline applied to haboobs/typhoons/squalls/monsoons. "waterspout" is a whole
        # word denoting EXCLUSIVELY the meteorological event — zero benign English meaning (the archaic "roof
        # waterspout"/downspout sense is spelled "downspout"/"spout", never "waterspout", in modern facility text) —
        # so this closes the miss with no operational false-positive risk. Surfaced in the 2026-08-29 severe-weather
        # marine-hazard rule-probe.
        #
        # "freezing rain" is the NWS-warned glaze-ice hazard that PRODUCES the already-floored "ice storm" (HIGH):
        # supercooled rain that freezes on contact, coating roads/walkways/handrails in glaze ice and loading power
        # lines and tree limbs until they fall — the mechanism behind the deadliest winter road pileups and the
        # multi-day outages of the 1998 North American and 2021 Texas ice storms. It is a directly-named severe-
        # weather hazard on the same footing as its sibling "ice storm", yet the spaced phrase reached NO floored
        # token: there is no "storm" in it so \bstorm\b cannot fire, and bare "rain" is NOT floored (only the spaced
        # "heavy rain" sits at the weather MEDIUM tier, and \bheavy\s+rain\b does not match "freezing rain") — so a
        # report writing "freezing rain coated every walkway" previously dropped to LOW, scoring strictly BELOW the
        # same glaze-ice event written "ice storm". This is the SAME whole-hazard absent-term miss class as
        # "pyroclastic surge" beside "pyroclastic flow" and "tropical cyclone" beside hurricane — the identical
        # winter hazard scored HIGH-or-LOW purely on which name the reporter reached for. Floored at HIGH beside
        # "ice storm", NOT critical, on the identical rationale as its ice-storm/blizzard siblings: a
        # glaze-ice/downed-line/deadly-road hazard a human/LLM can raise, not the guaranteed mass-casualty
        # catastrophe that warrants CRITICAL (hurricane/derecho/tsunami), and a freezing-rain event that injures a
        # worker or downs a line independently floors higher via injury/medical or electrical. No separate plural
        # entry is needed: "freezing rain" is a mass noun (like the already-HIGH "hail"/"snow"), reported in the
        # singular, so a "freezing rains" token does not occur in incident text. The two-word phrase denotes
        # EXCLUSIVELY the meteorological hazard — zero benign English meaning in facility/incident reporting — so
        # this closes the miss with no operational false-positive risk. Surfaced in the 2026-08-29 winter-weather
        # glaze-ice rule-probe (precipitation sibling of the already-HIGH ice storm).
        # "nor'easter"/"noreaster" (and plurals) is the directly-named severe Atlantic coastal storm — the
        # cyclone that drives blizzard whiteouts, hurricane-force winds, and coastal flooding up the U.S.
        # Northeast (the deadly 1991 "Perfect Storm", the 2018 bomb-cyclone nor'easters). It is a whole-storm
        # named hazard on the same footing as its already-HIGH siblings "snowstorm"/"blizzard"/"ice storm",
        # yet as a closed/apostrophe compound it reached NO floored token: \bstorm\b cannot fire (there is no
        # "storm" token — "nor'easter" ends in "easter"), and no "snow"/"wind"/"rain"/"blizzard" substring is
        # present either — so a report writing "a nor'easter knocked out the feeders" previously dropped to
        # LOW, scoring strictly BELOW the same storm written "snowstorm". This is the SAME whole-hazard
        # absent-term miss class as haboob-beside-sandstorm and monsoon-beside-storm. Floored at HIGH, NOT
        # critical, on the identical rationale as its snowstorm/blizzard/ice-storm siblings: a generic severe
        # storm's guaranteed harm is disruption/outage/access-loss, not the certain mass-casualty catastrophe
        # of a tornado/hurricane; a nor'easter that injures someone still escalates via injury/medical. Both
        # dominant spellings are added — the apostrophe form "nor'easter" and the bare "noreaster" — plus each
        # plural ("nor'easters"/"noreasters"), since \bnor'easter\b does not match "nor'easters"; the same
        # singular->plural discipline applied to windstorms/snowstorms/haboobs. Each is a whole word denoting
        # EXCLUSIVELY the coastal storm — zero benign English meaning — so this closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-29 winter-weather rule-probe (coastal-storm
        # sibling of the already-HIGH snowstorm/blizzard).
        #
        # "heat wave"/"heatwave" (+ plurals) is the directly-named severe prolonged-extreme-heat event — the
        # deadliest weather hazard in the U.S. by average annual deaths (the 1995 Chicago heat wave ~739 dead,
        # the 2003 European heat wave ~70,000, the 2021 Pacific-Northwest "heat dome"). It is the severe,
        # mass-casualty form of the already-tracked "heat advisory" (which floors only MEDIUM), yet it matched
        # nothing and dropped to LOW — an inversion where a mere watch-level "heat advisory" outranked the
        # actual killing event, because \bheat\s+advisory\b does NOT match "heat wave"/"heatwave" and no other
        # token fired. This is the same severe-form-below-its-own-advisory miss the taxonomy exists to close
        # (blizzard/ice storm HIGH sit above the MEDIUM snow/frost). Floored at HIGH, NOT critical: a heat wave
        # is a prolonged, forecastable public-health hazard (cooling/hydration/rest mitigate it), not the
        # guaranteed instantaneous mass-casualty catastrophe of a tornado/hurricane — HIGH is the defensible
        # floor a human/LLM can raise, and a worker who collapses from it independently floors critical via
        # injury/medical. Both the spaced "heat wave" and the closed compound "heatwave" are added (both are
        # common spellings), plus each plural — \bheat\s+wave\b does not match "heat waves" and \bheatwave\b
        # does not match "heatwaves" — the same singular->plural discipline applied to windstorms/haboobs/
        # nor'easters. In facility/incident text the phrase denotes EXCLUSIVELY the meteorological event (the
        # Motown song / band titles never appear in a hazard report), so this closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-29 extreme-heat rule-probe (severe sibling
        # of the already-MEDIUM heat advisory).
        # "cold wave"/"cold snap"/"arctic blast"/"arctic outbreak" (+ plurals) are the directly-named
        # severe extreme-cold events — the exact symmetric sibling of the just-added "heat wave" and the
        # cold-side mass-casualty hazard the NWS now warns as an "Extreme Cold Warning" (renamed from Wind
        # Chill Warning in 2024). Extreme cold is a leading weather killer (hypothermia deaths) and a top
        # infrastructure threat: the Feb-2021 Texas arctic outbreak froze wellheads and generation, killed
        # ~200+ people, and blacked out millions; cold snaps routinely burst pipes, freeze feeders, and spike
        # heating demand past capacity. Yet every one of these matched nothing and dropped to LOW: each is a
        # two-word phrase containing NO floored token — \bstorm\b cannot fire (no "storm" substring), there is
        # no "snow"/"wind"/"rain" token, and the MEDIUM "frost" is not a substring — so a report writing "an
        # arctic blast froze the feeders" or "the cold snap burst the sprinkler mains" scored strictly below
        # the same-severity "heat wave" purely on hot-vs-cold wording. This is the mirror of the heat-wave miss
        # (severe event with no token of its own) and the same absent-phrase class as nor'easter/derecho.
        # Floored at HIGH, NOT critical, on the identical heat-wave rationale: extreme cold is a prolonged,
        # forecastable public-health/infrastructure hazard (heating/shelter/insulation mitigate it), not the
        # guaranteed instantaneous catastrophe of a tornado/hurricane — HIGH is the defensible floor, and a
        # worker who suffers hypothermia/frostbite independently floors critical via injury/medical. Each
        # concept adds its plural as a distinct entry (\bcold\s+wave\b does not match "cold waves"), the same
        # singular->plural discipline applied to heat waves/nor'easters/windstorms. All four are unambiguous
        # multi-word phrases with zero benign meaning in incident/facility text; the polysemous bare "cold"
        # (a common cold, a cold start, cold storage) and bare "wave"/"blast" are DELIBERATELY left unfloored
        # (only the qualified phrases fire), and \barc\s+blast\b — the electrical CRITICAL token — cannot match
        # "arctic blast" (\barc\b has no boundary inside "arctic"). Surfaced in the 2026-08-29 extreme-cold
        # rule-probe (cold-side symmetric sibling of the just-added heat wave).
        # "black ice" is the directly-named glaze-ice road/walkway hazard — a thin, transparent layer of ice
        # that blends into the pavement, so drivers/pedestrians never see it until they lose traction. It is
        # the surface-level product of the already-HIGH "freezing rain" and "ice storm" (supercooled rain
        # freezing on contact, or refreeze of melt), and one of the deadliest winter road hazards (chronic
        # multi-vehicle pileups, slip-and-fall injuries on iced walkways/loading docks). Yet the phrase reached
        # NO floored token and dropped to LOW: there is no "storm" substring so \bstorm\b cannot fire,
        # \bice\s+storm\b does not match "black ice" (different second word), and the MEDIUM "frost" is not a
        # substring — so a report writing "patchy black ice on the access road" scored strictly below the same
        # glaze-ice event written "freezing rain"/"ice storm". This is the same whole-hazard absent-term miss
        # class as freezing-rain-beside-ice-storm. Floored at HIGH beside its freezing-rain/ice-storm producers,
        # NOT critical, on the identical glaze-ice rationale: a slip/traction hazard a human/LLM can raise, and
        # one that injures a worker or causes a crash independently floors critical via injury/medical. NO plural
        # entry — "black ice" is a mass noun like the already-HIGH "freezing rain"/"hail"/"snow" ("black ices"
        # does not occur in incident text). In facility/incident reporting the phrase denotes EXCLUSIVELY the ice
        # hazard (the AC/DC album, cocktail, and film titles never appear in a hazard report), and the adjacency
        # requirement means "black ink ... ice maker" cannot fire it — so this closes the miss with no operational
        # false-positive risk. Surfaced in the 2026-08-29 winter-weather glaze-ice rule-probe.
        #
        # "polar vortex" (+ plurals "polar vortexes"/"polar vortices") is the directly-named extreme-cold
        # driver — the stratospheric low whose collapse/stretch spills arctic air south and IS the phenomenon
        # the media/NWS name for the killing deep-freezes ("the polar vortex froze the feeders"): the Jan-2019
        # Midwest event that dropped Chicago to -23 F, and the Feb-2021 Texas outbreak that killed ~200+ and
        # blacked out millions. In incident/media text it is used interchangeably with the just-added "arctic
        # outbreak"/"cold wave" (both HIGH), yet it matched nothing and dropped to LOW: the two-word phrase
        # carries NO floored token — \bstorm\b cannot fire (no "storm" substring), there is no "snow"/"wind"/
        # "rain" token, MEDIUM "frost" is not a substring, and \barctic\s+(blast|outbreak)\b cannot match a
        # different second word — so a report writing "the polar vortex burst the intake mains" scored strictly
        # below the same-severity "arctic outbreak" purely on which name the reporter reached for. This is the
        # SAME whole-hazard absent-term miss class as freezing-rain-beside-ice-storm and cold-wave-beside-heat-
        # wave. Floored at HIGH beside its arctic-outbreak/cold-wave siblings, NOT critical, on the identical
        # extreme-cold rationale: a prolonged, forecastable public-health/infrastructure hazard (heating/shelter
        # mitigate it), not the guaranteed instantaneous catastrophe of a tornado/hurricane — HIGH is the
        # defensible floor, and a worker who suffers hypothermia/frostbite independently floors critical via
        # injury/medical. Each plural is a distinct entry (\bpolar\s+vortex\b matches neither "polar vortexes"
        # nor the Latin plural "polar vortices"), the same singular->plural discipline applied to cold waves/
        # arctic outbreaks/haboobs. The QUALIFIED two-word phrase only — the bare "vortex" is DELIBERATELY left
        # unfloored (it already appears benignly in this file's own "fire vortex"/firenado prose and in routine
        # "vortex shedding"/"vortex tube"/"vortex mixer" engineering text, which must NOT fire); "polar vortex"
        # itself denotes EXCLUSIVELY the meteorological phenomenon — zero benign meaning in facility/incident
        # reporting — so this closes the miss with no operational false-positive risk. Surfaced in the
        # 2026-08-29 extreme-cold rule-probe (named-driver sibling of the just-added arctic outbreak/cold wave).
        #
        # "freezing fog" is the directly-named glaze-ice producer — fog whose supercooled droplets freeze on
        # contact with any surface, coating roads/catwalks/aircraft and power lines in clear rime/glaze ice. It
        # is an NWS-named advisory hazard ("Freezing Fog Advisory") and the same glaze-ice family as the
        # already-HIGH "freezing rain"/"ice storm"/"black ice": a low-visibility driving hazard that also ices
        # walkways, ladders, and conductors. Yet the phrase reached NO floored token and dropped to LOW: there
        # is no "storm" substring so \bstorm\b cannot fire, \bfreezing\s+rain\b does not match "freezing fog"
        # (different second word), bare "fog" is not floored, and MEDIUM "frost" is not a substring — so a report
        # writing "dense freezing fog glazed the intake catwalk" scored strictly below the same glaze-ice event
        # written "freezing rain"/"black ice". This is the identical whole-hazard absent-term miss class as
        # black-ice-beside-ice-storm. Floored at HIGH beside its freezing-rain/black-ice/ice-storm siblings, NOT
        # critical, on the identical glaze-ice rationale: a visibility/traction hazard a human/LLM can raise, and
        # one that injures a worker or downs a line independently floors critical via injury/medical. NO plural
        # entry — "freezing fog" is a mass noun like the already-HIGH "freezing rain"/"black ice"/"hail" ("freezing
        # fogs" does not occur in incident text). In facility/incident reporting the phrase denotes EXCLUSIVELY the
        # meteorological event (zero benign meaning — there is no "freezing fog" machine/product/idiom), so this
        # closes the miss with no operational false-positive risk. Surfaced in the 2026-08-29 winter-weather
        # glaze-ice rule-probe (fog-borne sibling of the just-added black ice).
        #
        # "atmospheric river" is the NWS/CW3E-named narrow moisture plume that drives the West Coast's most
        # destructive flooding — the long-duration rain/mountain-snow producer behind levee breaks, debris flows,
        # and mass evacuations (the Jan 2023 California ARs killed 20+ and caused billions in damage; NWS/CW3E now
        # rate them AR1–AR5, AR4/AR5 = "hazardous to extreme"). It is a rain-bearing SYNOPTIC DRIVER exactly like
        # the already-HIGH "monsoon", yet the phrase reached NO floored token and dropped to LOW: there is no
        # "storm" word so \bstorm\b cannot fire, "river" is not "rain", and nothing floored is a substring of it —
        # so a report writing "an atmospheric river stalled over the watershed" scored strictly below the same
        # flood-driving pattern written "monsoon". This is the identical whole-hazard absent-term miss class as
        # monsoon/derecho beside their siblings. Floored at HIGH beside "monsoon", NOT critical, on the identical
        # driver rationale: the flooding it produces already floors critical via "flash flood"/water taxonomy, and
        # a human/LLM can raise it — the pattern itself is the minimum-severity signal. The plural "atmospheric
        # rivers" is a distinct token (\batmospheric\s+river\b does not match it, trailing \b), so it needs its own
        # entry, the same singular->plural discipline applied to monsoons/derechos. The phrase denotes EXCLUSIVELY
        # this meteorological event (zero benign English meaning — no "atmospheric river" product/idiom), so it
        # closes the miss with no operational false-positive risk. Surfaced in the 2026-08-29 flood-driver rule-probe.
        #
        # "landspout"/"landspouts" is the direct LAND analogue of the already-HIGH "waterspout" — an NWS-named
        # non-supercell tornado that spins up from the ground under a growing cumulus (the same vortex a waterspout
        # is, just over land instead of water). It reached NO floored token and dropped to LOW: no "storm"/"tornado"
        # substring, and nothing floored is a substring of it — so the same land-vortex hazard scored HIGH-or-LOW
        # purely on whether the reporter wrote "waterspout" or "landspout". Floored at HIGH beside "waterspout" (NOT
        # the critical "tornado" floor — landspouts are typically the weaker EF0–EF1 end, the same conservative call
        # already made for waterspout). The plural "landspouts" is a distinct token (\blandspout\b won't match it),
        # the same singular->plural discipline applied to waterspouts. Zero benign English meaning, so it closes the
        # miss with no operational false-positive risk. Surfaced in the 2026-08-29 severe-weather sibling rule-probe.
        #
        # "heat dome"/"heat domes" is the named synoptic DRIVER of an extreme heat wave — the stalled upper-level
        # high-pressure ridge that traps heat over a region (the June 2021 Pacific Northwest heat dome killed
        # hundreds and set all-time records). It is a heat-bearing driver exactly like the already-HIGH "heat wave"
        # it produces, yet the phrase reached NO floored token and dropped to LOW: no "storm" word, "dome" is not
        # floored, and nothing floored is a substring — so a report writing "a heat dome parked over the region"
        # scored below the same extreme-heat event written "heat wave". Floored at HIGH beside "heat wave", the same
        # driver rationale as "monsoon"/"atmospheric river". The plural "heat domes" is a distinct token. The phrase
        # denotes EXCLUSIVELY this meteorological event (zero benign "heat dome" idiom), so it closes the miss with
        # no operational false-positive risk. Surfaced in the 2026-08-29 severe-weather sibling rule-probe.
        #
        # "thundersnow" is the NWS-named convective winter phenomenon — a thunderstorm whose precipitation falls as
        # snow, a marker of intense (often 2-4+ in/hr) snowfall rates and the lightning hazard riding with it. It is
        # a directly-named severe-weather event on the footing of its siblings "snowstorm"/"thunderstorm", yet it
        # reached NO floored token and dropped to LOW: it is a single closed compound, so \bsnow\b (MEDIUM) cannot
        # fire on "thunder"+"snow" (no internal boundary) and \bthunderstorm\b does not match it either — the same
        # closed-compound tokenization miss already closed for snowstorm/hailstorm/windstorm. Floored at HIGH beside
        # "snowstorm"/"thunderstorm". It is one whole word with zero benign English meaning (mass noun, no plural
        # entry, like "freezing rain"), so it closes the miss with no operational false-positive risk. Surfaced in
        # the 2026-08-29 severe-weather sibling rule-probe.
        #
        # "gustnado"/"gustnados"/"gustnadoes" is the NWS/storm-spotter name (gust + tornado) for the short-lived
        # ground whirlwind that spins up along a thunderstorm's gust front / outflow boundary — a real damaging-wind
        # hazard that overturns high-profile vehicles, tears roofing, and hurls debris (it is NOT connected to the
        # cloud base like a true tornado, which is why it sits below the CRITICAL "tornado" floor). It is the
        # gust-front sibling of the already-HIGH vortex hazards "landspout"/"waterspout"/"microburst", yet it reached
        # NO floored token and dropped to LOW: a coined closed compound, so \btornado\b cannot fire on "gust"+"nado"
        # (no internal boundary) and \bstorm\b is nowhere in it — the same coined/closed-compound tokenization miss
        # closed for firenado (critical) and landspout (HIGH). Floored at HIGH beside "landspout"/"microburst" (NOT
        # the critical "tornado"/"firenado" floor — a gustnado is the weaker, cloud-detached end of the vortex
        # family, the same conservative EF0-EF1 call already made for landspout/waterspout). Each spelling is a
        # distinct token (\bgustnado\b matches neither "gustnados" nor "gustnadoes"), the same singular->plural
        # discipline applied to firenados/firenadoes/landspouts. It denotes EXCLUSIVELY this meteorological event
        # (zero benign English meaning), so it closes the miss with no operational false-positive risk. Surfaced in
        # the 2026-08-30 severe-weather gust-front rule-probe.
        #
        # "supercell"/"supercells" is the NWS/AMS-named PARENT severe thunderstorm — a storm with a deep, persistent
        # rotating updraft (a mesocyclone) that is the single most dangerous convective storm type, the structure that
        # spawns the most violent (EF4-EF5) tornadoes, giant hail, and damaging downbursts (a "supercell tracking toward
        # the metro" is standard NWS severe-weather wording). It is a directly-named severe-weather storm on exactly the
        # same footing as its already-HIGH siblings "thunderstorm"/"squall"/"downburst", yet as a single closed compound
        # it reached NO floored token: there is no "storm" in it so \bstorm\b cannot fire, "cell" is not floored, and
        # \bthunderstorm\b/\bsquall\b are different words — so a report writing "a supercell developed over the tank farm"
        # previously dropped to LOW while the same storm written "thunderstorm" floors HIGH. This is the SAME closed-
        # compound/absent-term tokenization miss as squall/haboob/thunderstorm-beside-storm. Floored at HIGH, NOT
        # critical, on the identical rationale as its thunderstorm/squall siblings: a severe-convective storm a human/LLM
        # can raise, not the guaranteed mass-casualty catastrophe that warrants CRITICAL (its offspring "tornado"/
        # "derecho" already floor critical, and a supercell that spawns a tornado or injures a worker independently
        # floors higher via those signals or injury/medical). Added both forms at the "storm" HIGH floor beside squall;
        # the plural "supercells" needs its own entry (\bsupercell\b does not match "supercells"), the same
        # singular->plural discipline applied to thunderstorms/squalls/downbursts. In operational facility safety/incident
        # reporting "supercell" denotes EXCLUSIVELY the meteorological storm — the narrow crystallography/battery senses
        # do not occur in this domain, the same negligible cross-domain tolerance already accepted for
        # "typhoon"/"derecho"/"molotov" — so this closes the miss with no operational false-positive risk. Surfaced in
        # the 2026-08-30 severe-convective rule-probe (parent-storm sibling of the already-HIGH thunderstorm/squall).
        #
        # "whiteout"/"whiteouts" is the NWS-named zero-visibility winter condition — the exact hazard cited as the
        # defining danger of the already-HIGH blizzard/snowstorm/lake-effect-snow siblings ("blizzard whiteouts",
        # "whiteout visibility, deadly highway pileups"): wind-driven snow reduces visibility to near zero, blinding
        # drivers and crews into the deadly multi-vehicle pileups and lost-worker searches that make it, like the
        # visibility-hazard haboob (also HIGH), a severe event in its own right. Yet as a CLOSED compound it reaches
        # no floored token: there is no "storm" in it so \bstorm\b cannot fire, no "snow" token (…ite|out, the MEDIUM
        # \bsnow\b needs a "snow" substring that "whiteout" lacks), and neither blizzard nor snowstorm is a substring
        # of it — so a report writing "a whiteout closed the highway and stranded the night crew" previously dropped
        # to LOW, scoring strictly BELOW the same event written "blizzard". This is the SAME closed-compound
        # tokenization miss as snowstorm/windstorm-beside-storm. Floored at HIGH beside blizzard/lake-effect snow,
        # NOT critical, on the identical rationale as those winter siblings: a visibility/access hazard a human/LLM
        # can raise, not the guaranteed mass-casualty catastrophe that warrants CRITICAL (hurricane/derecho/tsunami),
        # and a whiteout that injures a worker or causes a pileup independently floors higher via injury/medical.
        # Added both forms (\bwhiteout\b does not match "whiteouts"), the same singular->plural discipline applied to
        # snowstorms/windstorms/haboobs. In operational facility/incident reporting "whiteout" denotes EXCLUSIVELY the
        # meteorological condition — the correction-fluid sense is the branded "Wite-Out"/hyphenated "white-out", not
        # the closed word — so this closes the miss with no operational false-positive risk. Surfaced in the
        # 2026-08-30 winter-weather closed-compound rule-probe (visibility sibling of the already-HIGH blizzard).
        #
        # "sleet" is the directly-named winter precipitation (ice pellets that bounce and accumulate into a
        # slick, treacherous coating on roads, catwalks, and docks) — the winter-precip peer of the already-
        # MEDIUM "snow" and "frost". Yet it matched nothing and dropped to LOW: "sleet" shares no substring
        # with any floored token (\bsnow\b/\bfrost\b are different words, and the HIGH glaze-ice tokens
        # "freezing rain"/"black ice"/"freezing fog" are different phrases), so a report writing "sleet
        # coated the access road" scored strictly BELOW the same-severity event written "snow"/"frost" — an
        # UNDER-FLOOR inversion in the SAME class as heat-wave-beneath-heat-advisory and lake-effect-snow-at-
        # bare-snow. Floored at MEDIUM (exactly its snow/frost winter-precip peers, NOT the HIGH glaze-ice
        # tier): sleet is visible bouncing ice pellets, less treacherous than the invisible glaze ice of
        # freezing rain/black ice (which floor HIGH), and MEDIUM is the honest raise-able floor a human/LLM
        # can lift when an event is severe — a heavy sleet event that injures a worker independently floors
        # higher via injury/medical. "sleet" is a mass noun (like snow/frost/hail — "sleets" does not occur
        # in incident text), so no plural entry. The NWS synonym "ice pellets" is DELIBERATELY EXCLUDED: ice-
        # making equipment legitimately dispenses literal "ice pellets" (pellet/nugget ice), so \bice\s+pellets\b
        # would over-fire on routine facilities text — the SAME polysemy-exclusion discipline that kept
        # "cyclone"/"mudslide"/"landslide" out while their zero-benign twins typhoon/mudflow were added; only
        # the unambiguous "sleet" (an ice machine never produces "sleet") fires. Surfaced in the 2026-08-30
        # winter-precipitation rule-probe (MEDIUM peer of snow/frost).
        # "gale-force winds"/"gale force winds"/"gale warning" -> weather HIGH. A gale is the directly-named
        # NWS severe-wind hazard (a Gale Warning covers sustained winds of 34-47 knots / 39-54 mph — enough
        # to topple cranes, tear roofing, capsize small craft, and down lines), the wind-family peer of the
        # already-HIGH "high winds"/"windstorm"/"squall". Yet these phrases matched NO floored token and
        # dropped to LOW: \bhigh\s+winds\b does not match "gale-force winds" (different first word),
        # \bwindstorm\b/\bstorm\b share no substring with them, and "gale warning" reaches no token either —
        # so a report writing "gale-force winds battered the rig" scored strictly BELOW the same event
        # written "high winds", a whole-hazard absent-term miss in the SAME class as heat-wave-beside-heat-
        # advisory and freezing-rain-beside-ice-storm. Floored HIGH beside "high winds" (a damaging severe-
        # wind hazard a human/LLM can raise; one that injures a worker or downs a line independently floors
        # higher via injury/medical or electrical). Added BOTH the hyphenated "gale-force winds" and the
        # spaced "gale force winds" — the matcher treats the hyphen literally, so the spaced form needs its
        # own entry (the both-spellings discipline of lake-effect snow / nor'easter) — plus the NWS product
        # name "gale warning". The bare root "gale" is DELIBERATELY EXCLUDED: it is polysemous (a proper name
        # — "Gale from accounting"; the figurative "a gale of laughter"), so \bgale\b would over-fire on
        # routine text — the SAME qualified-phrase discipline that floored "storm surge"/"arctic blast"/"cold
        # snap"/"heat advisory" while leaving their polysemous bare halves unfloored; only the unambiguous
        # zero-benign phrases fire. Surfaced in the 2026-08-30 severe-wind rule-probe (HIGH peer of high winds).
        #
        # "extreme cold warning"/"wind chill warning" name the NWS life-threatening-cold PRODUCTS — the
        # directly-named warning-tier hazard for exactly the dangerous cold the whole cold family already
        # floors HIGH (arctic blast/cold wave/polar vortex). "Extreme Cold Warning" is the CURRENT NWS
        # product (it replaced "Wind Chill Warning" in winter 2024-25; issued for life-threatening cold /
        # dangerous wind chills — frostbite in minutes, hypothermia), and "Wind Chill Warning" is both the
        # prior NWS name and the standing Environment Canada product, so it still appears in historical +
        # imported bulletins — the both-names discipline of nor'easter/noreaster. Yet neither phrase reached
        # a floored token: no "storm" substring so \bstorm\b cannot fire, "cold"/"wind"/"chill"/"warning" are
        # not floored alone, and \bcold\s+wave\b / \barctic\s+(blast|outbreak)\b / \bpolar\s+vortex\b cannot
        # match a different phrase — so a report writing "Extreme Cold Warning; wind chills to -40F" dropped
        # to LOW, scoring strictly BELOW the same event written "arctic blast" (the SAME whole-hazard
        # absent-term / NWS-product-name miss class as the already-floored "gale warning" and "severe storm
        # warning"). Floored HIGH beside arctic blast/cold wave/polar vortex, NOT critical (a forecastable
        # public-health cold hazard mitigable by shelter/PPE, not the guaranteed instantaneous mass-casualty
        # catastrophe of a tornado/hurricane; a worker with frostbite/hypothermia independently floors
        # critical via injury/medical). Singular product names only — no plural entry, the same discipline as
        # "gale warning"/"severe storm warning". The bare "wind chill" is DELIBERATELY EXCLUDED as
        # polysemous-by-severity (a routine "wind chill of 25F this morning" must stay LOW — FP guard below),
        # as is the bare "extreme cold" (a vaguer descriptor / cold-storage context); only the unambiguous
        # zero-benign WARNING products fire — the qualified-phrase discipline of gale/gale warning. The
        # advisory tier (wind chill advisory / cold weather advisory, the MEDIUM cold peer of the MEDIUM
        # "heat advisory") was DEFERRED to a later probe — NOW SHIPPED: "wind chill advisory" floored MEDIUM
        # 2026-08-31 (see medium list), and its 2024-rename current name "cold weather advisory" floored MEDIUM
        # later the same day. Surfaced in the 2026-08-31 extreme-cold rule-probe.
        # "flash freeze"/"flash freezes" is the directly-named NWS winter hazard — a rapid, large temperature
        # crash (often behind an arctic front) that freezes standing water and wet/slushy roadways into ice
        # almost instantly, the meteorological EVENT that PRODUCES the already-HIGH "black ice" and glaze-ice
        # surfaces. Yet a report writing "a flash freeze glazed the access road" reached NO floored token and
        # dropped to LOW (verified live): "flash freeze" shares no substring with any floored token —
        # \bflash\s+flood\b (the critical water phrase) does not match "flash freeze" (different second word),
        # there is no bare "freeze" token, and the HIGH glaze-ice phrases "freezing rain"/"black ice"/
        # "freezing fog" are different words — so the same treacherous glaze-ice event scored strictly BELOW
        # its own PRODUCT written "black ice"/"freezing rain" (both HIGH), the identical whole-hazard absent-
        # term / under-floor inversion already fixed for black-ice-beside-ice-storm and sleet-beneath-snow.
        # Floored HIGH beside the glaze-ice family, NOT critical: if the flash freeze IS flooding first ("the
        # flash freeze flooded the lot") the bare "flooded" token independently escalates to critical
        # (verified live), and a worker injured on the ice floors higher via injury/medical — so HIGH is the
        # conservative warning-stage floor for the flash freeze named on its own. Both "flash freeze" and the
        # plural event form "flash freezes" get entries (\bflash\s+freeze\b does not match the trailing "s";
        # unlike the mass-noun black ice/freezing rain, a flash freeze is a countable event — "two flash
        # freezes this week" — so the plural occurs in incident text), the singular->plural discipline of the
        # event-noun siblings storm surge/seiche/king tide. "flash freeze" denotes EXCLUSIVELY this hazard
        # (the culinary "flash-freeze" sense — freezing food fast — does not appear in facilities/infra
        # incident text, and even there it is written hyphenated as a verb), so it closes the miss with no
        # operational false-positive risk. Surfaced in the 2026-08-31 glaze-ice rule-probe (event sibling of
        # black ice / freezing rain / freezing fog).
        # "volcanic ash"/"volcanic ashfall" is the directly-named USGS/NWS/NOAA downwind hazard of an
        # eruption — the tephra plume that grounds aircraft (jet-engine flameout is the reason NOAA runs the
        # Volcanic Ash Advisory Centers), collapses roofs under wet-ash load, contaminates water, and is a
        # respiratory/eye hazard (NWS issues Ashfall Advisories and Ashfall Warnings). Yet a report writing
        # "heavy volcanic ash is falling on the plant" or "the volcanic ash cloud grounded all flights"
        # reached NO floored token and dropped to LOW (verified live): all the volcanic entries above
        # ("volcanic eruption", "pyroclastic flow", "lahar", "lava flow", "limnic eruption") name the ERUPTION
        # or its ground flows, none is a substring of "volcanic ash"/"volcanic ashfall", and there is no bare
        # "ash" token — so the same active volcanic emergency scored strictly BELOW its own eruption written
        # "eruption"/"lahar" (all critical), the SAME whole-hazard absent-term miss class as storm surge beside
        # flood and flash freeze beside black ice. Floored HIGH, NOT critical: ashfall severity is
        # dose-dependent (a light dusting is a nuisance, a roof-loading fall is life-threatening), and if the
        # eruption itself is in the report the "volcanic eruption"/"pyroclastic"/"lahar" tokens independently
        # escalate to critical (verified live) while a worker with an ash-triggered respiratory collapse floors
        # via injury/medical — so HIGH is the conservative advisory-stage floor for ashfall named on its own,
        # the same discipline used for the coastal life-threats (storm surge / sneaker wave / rip current).
        # DELIBERATELY qualified with "volcanic": the bare closed compound "ashfall" and the two-word "ash
        # fall" are EXCLUDED as domain-polysemous — in a facilities/infra incident report "ash"/"ash fall" can
        # mean incinerator, furnace, or combustion residue, which must not fire a weather HIGH — whereas
        # "volcanic ash"/"volcanic ashfall" carry ZERO benign meaning, so this closes the miss with no
        # operational false-positive risk. Both entries are needed: \bvolcanic\s+ash\b does not match the
        # closed compound "volcanic ashfall" (no word boundary between "ash" and "fall"), so the ashfall spelling
        # gets its own token. Surfaced in the 2026-08-31 volcanic-hazard rule-probe (downwind sibling of the
        # eruption/pyroclastic/lahar criticals).
        # "red flag warning" names the NWS fire-weather PRODUCT — the warning issued when low humidity, strong
        # wind, and dry fuels combine into critical conditions for rapid wildfire ignition and spread (a Red Flag
        # Warning tells crews that any ember will run). It is the fire-side sibling of the just-added cold PRODUCTS
        # "extreme cold warning"/"wind chill warning": a directly-named NWS warning that a human/LLM incident
        # report routinely cites by product name. Yet a report writing "a red flag warning is in effect for the
        # tank-farm district" reached NO floored token and dropped to LOW (verified live): "red flag warning"
        # shares no substring with any floored token — the critical fire tokens name the FIRE itself
        # ("wildfire"/"bushfire"/"conflagration"/"structure fire"), none is a substring of "red flag warning", and
        # there is no bare "warning"/"flag" token — so the same fire-weather emergency scored strictly BELOW the
        # fire it forecasts written "wildfire" (critical), the SAME whole-hazard absent-term miss class as "extreme
        # cold warning" beside the cold criticals and "volcanic ash" beside the eruption criticals. Floored HIGH,
        # NOT critical: a Red Flag Warning is a forecast of conditions, not an active burn — if a wildfire actually
        # ignites the "wildfire"/"conflagration" tokens independently escalate to critical (verified live: "fire
        # crews staging under a red flag warning" already scores critical on the fire context) and a burned worker
        # floors via injury/medical, so HIGH is the conservative warning-stage floor for the product named on its
        # own, the identical discipline used for extreme cold warning / wind chill warning. DELIBERATELY floored
        # only as the full three-word phrase: the bare idiom "red flag" (a warning sign, a beach/racing flag, a
        # code-review red flag) is domain-polysemous and MUST NOT fire a weather HIGH — "red flag warning" carries
        # ZERO benign meaning, the same qualified-phrase discipline that floored "volcanic ash" (not bare "ash")
        # and "storm surge" (not bare "surge"). No separate plural entry: like the sibling products "extreme cold
        # warning"/"wind chill warning" the NWS product name is used as a mass term and the bare-phrase entry
        # already covers the operative singular. Surfaced in the 2026-08-31 fire-weather rule-probe (product
        # sibling of the extreme cold warning / wind chill warning NWS-named warnings).
        # "excessive heat warning"/"extreme heat warning" name the NWS warning-grade extreme-heat PRODUCT — the
        # life-threatening-heat counterpart of the already-MEDIUM "heat advisory", issued when a dangerous,
        # prolonged heat event ("extreme heat warning" is the current NWS product name adopted in 2024; "excessive
        # heat warning" is the legacy name still in wide use, so both spellings are added). It is the HOT-side
        # mirror of the already-HIGH cold PRODUCTS "extreme cold warning"/"wind chill warning": a directly-named
        # NWS warning a human/LLM incident report routinely cites by product name. Yet a report writing "an
        # excessive heat warning is in effect for the site all week" reached NO floored token and dropped to LOW
        # (verified live): "excessive heat warning" shares no substring with any floored weather token — the HIGH
        # heat events name the phenomenon itself ("heat wave"/"heatwave"/"heat dome"), none is a substring of
        # "excessive heat warning"/"extreme heat warning" (\bheat\s+wave\b / \bheat\s+dome\b cannot match), the
        # MEDIUM "heat advisory" is a different final word (\bheat\s+advisory\b cannot match "...heat warning"),
        # and there is no bare "heat"/"warning" token — so the warning-grade heat product scored strictly BELOW
        # its own advisory-grade sibling "heat advisory" (MEDIUM), the exact advisory-beneath-warning INVERSION
        # the cold family already fixed by pairing "wind chill advisory" (MEDIUM) with "wind chill warning" (HIGH).
        # This is the same whole-hazard absent-term miss class as "extreme cold warning" beside the cold criticals
        # and "red flag warning" beside the fire criticals. Floored HIGH, NOT critical: an Extreme Heat Warning is
        # a forecast/duration product, not a confirmed mass-casualty event — if the killing heat is itself in the
        # report the "heat wave"/"heat dome" tokens independently sit HIGH, and a worker with heat stroke floors
        # via injury/medical, so HIGH is the conservative warning-stage floor for the product named on its own,
        # the identical discipline used for extreme cold warning / wind chill warning / red flag warning.
        # DELIBERATELY floored only as the full qualified phrases: the bare "heat" is domain-polysemous (a heat
        # exchanger, "turn up the heat", body heat) and MUST NOT fire a weather HIGH — "excessive heat warning" /
        # "extreme heat warning" carry ZERO benign meaning, the same qualified-phrase discipline that floored
        # "red flag warning" (not bare "red flag") and "volcanic ash" (not bare "ash"). No separate plural entry:
        # like the sibling products "extreme cold warning"/"wind chill warning"/"red flag warning" the NWS product
        # name is used as a mass term and the bare-phrase entry already covers the operative singular. Surfaced in
        # the 2026-08-31 9:4x PM heat-product rule-probe (warning-grade heat sibling of the MEDIUM heat advisory,
        # symmetric hot-side counterpart of the HIGH extreme cold warning / wind chill warning).
        # "high wind warning" names the NWS warning-grade damaging-wind PRODUCT — the warning issued for sustained
        # winds >=40 mph (or gusts >=58 mph), the same destructive wind the whole wind family already floors HIGH
        # ("high winds"/"gale-force winds"/"windstorm"/"squall"). It is the WARNING-grade sibling of the just-added
        # MEDIUM advisory-grade "wind advisory" (see medium list), completing the wind family's advisory->MEDIUM /
        # warning->HIGH pair exactly as the heat family (heat advisory MEDIUM / extreme heat warning HIGH) and the
        # cold family (wind chill advisory MEDIUM / wind chill warning HIGH) already do. Yet a report writing "a high
        # wind warning is in effect for the crane district" reached NO floored token and dropped to LOW (verified
        # live): the floored token is the PLURAL "high winds", and \bhigh\s+winds\b cannot match the singular "high
        # wind" in "high wind warning" (different word — winds vs wind), \bwindstorm\b/\bstorm\b share no substring,
        # and there is no bare "wind"/"warning" token — so the warning-grade product scored strictly BELOW the same
        # event written "high winds" (HIGH), the SAME whole-hazard absent-term / NWS-product-name miss class as
        # "extreme cold warning" beside the cold criticals and "gale warning" beside the HIGH winds. Floored HIGH
        # beside "high winds", NOT critical (a forecastable damaging-wind product mitigable by securing loose gear /
        # sheltering, not a guaranteed mass-casualty catastrophe; a worker injured or a line downed by the wind
        # independently floors higher via injury/medical or electrical). Singular product name only — no plural
        # entry, the same discipline as "gale warning"/"extreme cold warning"/"red flag warning". The bare "high
        # wind" (singular) is DELIBERATELY EXCLUDED as a vaguer descriptor a routine forecast uses ("high wind gusts
        # possible" is not itself an incident), and the bare "wind" is domain-polysemous ("second wind", "a wind of
        # change", "wind the clock") — only the unambiguous zero-benign PRODUCT name fires, the qualified-phrase
        # discipline of gale warning / red flag warning. Surfaced in the 2026-09-01 12:4x AM wind-product rule-probe
        # (warning-grade wind sibling of the MEDIUM wind advisory, completing the wind family advisory/warning pair).
        # "freeze warning" names the NWS warning-grade lethal/damaging-cold PRODUCT — issued when sub-freezing
        # temperatures will kill crops and sensitive vegetation, burst exposed pipes, and threaten unsheltered people
        # or animals. It is the warning-grade sibling of the advisory-grade "frost advisory" (which already floors
        # MEDIUM via the bare "frost" token), so before this add the frost/freeze pair was INVERTED: "a frost advisory
        # is in effect" scored MEDIUM while the more dangerous "a freeze warning is in effect" dropped to LOW — a
        # warning-grade product ranking BELOW its own advisory-grade sibling, the exact advisory-beneath-warning
        # inversion the cold family (wind chill advisory MEDIUM / wind chill warning HIGH) and heat family (heat
        # advisory MEDIUM / extreme heat warning HIGH) already fixed. It previously dropped LOW because the phrase
        # shares no substring with any floored token — the HIGH freeze phrases "flash freeze"/"freezing rain"/
        # "freezing fog" are different words (\bflash\s+freeze\b / \bfreezing\s+rain\b cannot match "freeze warning"),
        # \bstorm\b/\bblizzard\b share no substring, and the bare verb "freeze" is DELIBERATELY unfloored (hiring
        # freeze, freeze-frame, "freeze the sample"; guarded by test_bare_freeze_stays_low) — so the standalone NWS
        # product name matched nothing, the same whole-hazard absent-term / NWS-product-name miss the wind/cold/heat
        # warnings fixed. Floored HIGH beside the cold PRODUCTS "extreme cold warning"/"wind chill warning". Singular
        # product name only — no plural entry, the same gale warning / extreme cold warning / high wind warning
        # discipline. Surfaced in the 2026-09-01 2:1x AM winter-product rule-probe (warning-grade freeze sibling of
        # the MEDIUM frost advisory, completing the frost/freeze advisory/warning pair).
        # "heavy freezing spray warning" names the NWS marine life-threat PRODUCT — issued when freezing
        # spray (sea spray/fog freezing on contact) is expected to accumulate ICE on a vessel fast enough
        # (typ. >=2 cm/hr) to endanger it: ice loading high on the superstructure raises the center of
        # gravity toward capsize and glazes decks/rails/rigging into a crew-fall trap. It is the marine
        # cold-icing peer of the already-HIGH cold PRODUCTS "extreme cold warning"/"wind chill warning".
        # Yet the phrase reached NO floored token and dropped to LOW (verified live): the HIGH glaze-ice
        # phrases "freezing rain"/"freezing fog" are different words (\bfreezing\s+rain\b cannot match
        # "freezing spray"), \bflash\s+freeze\b/\bfreeze\s+warning\b are different phrases, bare "freeze"
        # is DELIBERATELY unfloored, and "spray"/"heavy"/"warning" are not floored alone — so a report
        # writing "a heavy freezing spray warning is in effect; ice building on the superstructure" scored
        # strictly BELOW the same-severity cold event written "extreme cold warning", the SAME whole-hazard
        # absent-term / NWS-product-name miss class the wind/cold/heat/freeze warnings fixed. Floored HIGH
        # beside the cold warning products, NOT critical (a forecastable, mitigable marine hazard — de-ice,
        # alter course, seek harbor; a crew member injured by an ice fall or the vessel actually foundering
        # independently floors higher via injury/medical or the critical water tokens). Added the plural
        # "heavy freezing spray warnings" (\b...warning\b does not match the trailing "s"), the singular->
        # plural discipline applied throughout the taxonomy. The bare "freezing spray" is DELIBERATELY
        # EXCLUDED as polysemous: "freezing spray" / "freeze spray" is also a common aerosol component-
        # cooling / fault-finding product ("hit the connector with freezing spray to find the intermittent
        # fault"), so \bfreezing\s+spray\b alone would over-fire on routine electronics/maintenance text —
        # the SAME qualified-phrase / zero-benign discipline that floored the full "high surf warning" while
        # leaving the recreational bare "high surf" LOW, and kept "ice pellets" out for the ice-machine
        # collision. ONLY the unambiguous full NWS product phrase floors. The advisory tier "freezing spray
        # advisory" (lighter icing, one NWS gradient down) floors MEDIUM beside the cold advisories,
        # completing the marine-icing advisory->MEDIUM / warning->HIGH ladder (the same ladder built for
        # high surf, wind chill, heat, and freeze). Surfaced in the 2026-09-01 6:4x PM marine-cold rule-probe.
        # "special marine warning"/"special marine warnings" names the NWS short-fuse marine WARNING PRODUCT — the
        # marine analog of a Severe Thunderstorm Warning, issued for a brief but intense hazard over coastal/bay/lake
        # waters: severe thunderstorm winds >=34 kt (39+ mph), waterspouts, or hail >=1 inch bearing down on vessels in
        # the next ~2 hours. It is the imminent-severe top of the marine-wind fuse the taxonomy already builds — small
        # craft advisory MEDIUM -> gale watch MEDIUM -> gale warning HIGH — and the marine sibling of the land severe-
        # storm warnings, so it belongs at HIGH beside "gale warning" and its own constituent hazards (the phenomena a
        # SMW warns on — "thunderstorm"/"waterspout"/"hail"/"squall" — already floor HIGH). Yet a report writing "a
        # special marine warning was issued for the bay" reached NO floored token and dropped to LOW (verified live):
        # "special marine warning" shares no substring with any floored token — bare \bstorm\b/\bwaterspout\b/\bhail\b
        # are different words, "gale warning"/"severe storm warning" are different phrases, and "special"/"marine"/
        # "warning" are not floored alone — the SAME whole-hazard absent-term / NWS-product-name miss class the wind/
        # cold/heat/freeze warnings and the gale/small-craft marine ladder fixed. Floored HIGH, NOT critical: a SMW is a
        # brief-fuse forecast product mitigable by getting vessels off the water / into harbor; if the severe storm or
        # waterspout it warns on actually strikes, the thunderstorm/waterspout/hail tokens independently floor HIGH and
        # a swamped/injured crew escalates via injury/medical or the critical water tokens. "warning" is countable so the
        # plural "special marine warnings" is a distinct token (\b...warning\b cannot match the trailing "s") and gets its
        # own entry, the singular->plural discipline applied throughout. The full three-word product phrase carries ZERO
        # benign polysemy (unlike its separable component words), so only the adjacent phrase fires — the same qualified-
        # phrase discipline as gale warning / red flag warning / high surf warning (a benign "special", "marine", and
        # "warning" separated by other words stays LOW; adjacency FP-guard added). Surfaced in the 2026-09-02 12:4x AM
        # marine-product rule-probe (imminent-severe warning tier atop the small-craft/gale marine-wind ladder).
        "high":     ["storm", "lightning strike", "lightning struck", "struck by lightning",
                     "supercell", "supercells",
                     "hail", "high winds", "fallen tree",
                     "gale-force winds", "gale force winds", "gale warning",
                     "high wind warning",
                     "haboob", "haboobs",
                     "sandstorm", "sandstorms",
                     "duststorm", "duststorms",
                     "snowstorm", "snowstorms",
                     "lake-effect snow", "lake effect snow",
                     "avalanche warning", "snow avalanche",
                     "whiteout", "whiteouts",
                     "windstorm", "windstorms",
                     "thunderstorm", "thunderstorms",
                     "thundersnow",
                     "hailstorm", "hailstorms",
                     "downburst", "downbursts",
                     "microburst", "microbursts",
                     "macroburst", "macrobursts",
                     "squall", "squalls",
                     "rainstorm", "rainstorms",
                     "cloudburst", "cloudbursts",
                     "monsoon", "monsoons",
                     "atmospheric river", "atmospheric rivers",
                     "waterspout", "waterspouts",
                     "landspout", "landspouts",
                     "gustnado", "gustnados", "gustnadoes",
                     "funnel cloud", "funnel clouds",
                     "freezing rain", "black ice", "freezing fog",
                     "flash freeze", "flash freezes",
                     "freeze warning",
                     "nor'easter", "nor'easters", "noreaster", "noreasters",
                     "heat wave", "heat waves", "heatwave", "heatwaves",
                     "heat dome", "heat domes",
                     "excessive heat warning", "extreme heat warning",
                     "cold wave", "cold waves", "cold snap", "cold snaps",
                     "arctic blast", "arctic blasts", "arctic outbreak", "arctic outbreaks",
                     "polar vortex", "polar vortexes", "polar vortices",
                     "extreme cold warning", "wind chill warning",
                     "heavy freezing spray warning", "heavy freezing spray warnings",
                     "volcanic ash", "volcanic ashfall",
                     "red flag warning",
                     "special marine warning", "special marine warnings",
                     "downed line", "ice storm", "blizzard"],
        # "dense fog" names the NWS Dense Fog Advisory hazard — visibility collapse (typically < 1/4 mile) that
        # produces the deadly multi-vehicle chain-reaction pileups fog is known for. It is a directly-named
        # ADVISORY-tier product a human/LLM incident report routinely cites ("a dense fog advisory is in effect",
        # "dense fog dropped visibility to zero on the highway"). Yet it reached NO floored token and dropped to
        # LOW (verified live): "dense fog" shares no substring with any floored token — the HIGH glaze-ice phrase
        # "freezing fog" is a different first word (\bfreezing\s+fog\b cannot match "dense fog"), the HIGH
        # visibility phrase "whiteout" is a different word, and bare "fog" is DELIBERATELY unfloored (benign: "fog
        # of war", "brain fog", "light fog"). Floored MEDIUM, NOT HIGH: the taxonomy mirrors NWS product tiers —
        # ADVISORY → MEDIUM (this sits beside "heat advisory"/"frost"), WARNING → HIGH. Dense fog is a
        # single-hazard (visibility) advisory, one gradient BELOW its dual-hazard sibling "freezing fog" (HIGH),
        # which adds black-ice road glaze on top of the visibility loss — the exact freezing-fog-vs-dense-fog
        # gradient NWS itself draws (Dense Fog Advisory vs the warning-grade freezing/whiteout products). No
        # plural entry: fog is a mass noun (like the HIGH "freezing fog"), so the bare-phrase entry covers it, and
        # \bdense\s+fog\b already fires inside "dense fog advisory". The qualified phrase "dense fog" carries ZERO
        # benign meaning (unlike bare "fog"), the same qualified-phrase discipline that floored "volcanic ash"
        # (not bare "ash") and "red flag warning" (not bare "red flag"). Surfaced in the 2026-08-31 12:4x
        # visibility-advisory rule-probe (advisory-tier sibling of the HIGH freezing fog / whiteout products; the
        # first MEDIUM-tier close of a directly-named-hazard under-floor in this run's weather sweep).
        #
        # "funnel cloud"/"funnel clouds" is the NWS-named tornado PRECURSOR — the rotating, funnel-shaped
        # condensation cloud descending from a cumulonimbus base that is NOT (yet) in contact with the ground; it
        # is the visible vortex the NWS warns on ("a funnel cloud was reported near the tank farm") in the minutes
        # before it becomes a tornado. Yet it reached NO floored token and dropped to LOW (verified live): "funnel
        # cloud" shares no substring with the CRITICAL "tornado" (different words) or with "storm", "funnel" is not
        # floored, "cloud" is not floored, and nothing floored is a substring of it — so the same warning-stage
        # vortex scored HIGH-or-LOW purely on whether the reporter wrote "waterspout"/"landspout"/"gustnado" or
        # "funnel cloud". Floored at weather HIGH beside its vortex-family siblings waterspout/landspout/gustnado,
        # NOT critical: a funnel cloud has not touched down — the instant it does it IS a tornado and the bare
        # "tornado" token independently escalates to critical (verified live: "the funnel cloud touched down as a
        # tornado" scores critical), so HIGH is the conservative warning-stage floor for the funnel cloud named on
        # its own, the same conservative EF0-precursor call already made for landspout/waterspout. DELIBERATELY
        # floored only as the full two-word phrase: bare "funnel" (funnel cake, a sales funnel, the object) and bare
        # "cloud" (cloud computing, cloud cover) are domain-polysemous and MUST NOT fire a weather HIGH — "funnel
        # cloud" together carries ZERO benign meaning, the same qualified-phrase discipline as "red flag warning"
        # (not bare "red flag") / "volcanic ash" (not bare "ash") / "storm surge" (not bare "surge"). The plural
        # "funnel clouds" is a distinct token (\bfunnel\s+cloud\b won't match the trailing "s"), the same
        # singular->plural discipline applied throughout the vortex family. Surfaced in the 2026-08-31 vortex-
        # precursor rule-probe (warning-stage sibling of the HIGH waterspout/landspout/gustnado vortices).
        #
        # "graupel" (soft hail / snow pellets / "grits") is the directly-named frozen-precipitation TYPE formed
        # when supercooled droplets rime onto a falling snow crystal into a soft opaque pellet. Its hazard is the
        # SAME slick-surface/reduced-traction class as "sleet" (already MEDIUM): graupel accumulates like ball
        # bearings on roads, catwalks, and stairs (a slip/fall + traffic-traction nuisance), and it is a marker of
        # convective winter instability (it commonly falls in thundersnow bursts). Yet a report writing "graupel
        # pellets coated the loading-dock stairs" reached NO floored token and dropped to LOW (verified live):
        # "graupel" shares no substring with any floored token — it is a wholly distinct word from sleet / hail /
        # snow — so the SAME frozen-precip traction hazard scored MEDIUM-or-LOW purely on whether the reporter
        # reached for the plain "sleet"/"hail" or the meteorological term "graupel". Floored MEDIUM beside "sleet"/
        # "snow", NOT HIGH: like sleet it is a precipitation TYPE / advisory-grade nuisance, one gradient below the
        # warning-grade glaze-ice products "freezing rain"/"black ice"/"flash freeze" (HIGH) that coat surfaces in
        # a bonded ice sheet; an actual injury or vehicle wreck on the graupel independently escalates via the
        # injury/medical floor. No plural entry: graupel is a mass noun (like the MEDIUM "sleet"/"frost"/"snow" and
        # the HIGH "freezing fog"), so the bare-word entry covers it. "graupel" carries ZERO benign meaning — it
        # denotes EXCLUSIVELY this precipitation type, no polysemy to guard (unlike the deliberately-unfloored
        # bare "funnel"/"cloud"/"surge"). Surfaced in the 2026-08-31 3:4x frozen-precip rule-probe (advisory-tier
        # traction sibling of the MEDIUM sleet, one gradient below the HIGH glaze-ice warning products).
        #
        # "wind chill advisory"/"wind chill advisories" names the NWS ADVISORY-tier dangerous-cold product — the
        # advisory-grade warning that wind chills have dropped to the frostbite/hypothermia-risk range (the threshold
        # a step below the life-threatening WARNING level). It is the exact advisory-tier sibling of the already-HIGH
        # WARNING products "wind chill warning"/"extreme cold warning" — the SAME cold hazard, one NWS gradient
        # lower. Yet a report writing "a wind chill advisory is in effect for the yard crew" reached NO floored token
        # and dropped to LOW (verified live): "wind chill advisory" shares no substring with the HIGH "wind chill
        # warning" (different final word — \bwind\s+chill\s+warning\b cannot match "wind chill advisory"), bare "wind
        # chill" is DELIBERATELY unfloored (a routine "wind chill of 25F", a wind-chill chart — the existing FP
        # guard), and no floored token is a substring of it. Floored MEDIUM, NOT HIGH: the taxonomy mirrors NWS
        # product tiers — ADVISORY -> MEDIUM (this sits beside "heat advisory"/"frost"/"dense fog"), WARNING -> HIGH
        # (its "wind chill warning"/"extreme cold warning" siblings) — the exact advisory-vs-warning gradient just
        # codified for dense fog (MEDIUM) beneath freezing fog (HIGH). UNLIKE the mass-noun "dense fog"/"graupel"
        # (no plural), "advisory" is a COUNTABLE noun and NWS routinely issues multi-zone "wind chill advisories"
        # ("wind chill advisories were issued for the northern counties"), a distinct token \bwind\s+chill\s+advisory\b
        # cannot match — so the plural is added per the countable-noun singular->plural discipline (this also closes
        # the latent singular-only gap the older "heat advisory" entry still carries). The qualified phrase carries
        # ZERO benign meaning (unlike bare "wind chill"), the same qualified-phrase discipline as "dense fog" (not
        # bare "fog") / "red flag warning" (not bare "red flag"). Surfaced in the 2026-08-31 5:1x cold-advisory
        # rule-probe (advisory-tier sibling of the HIGH wind chill warning / extreme cold warning products).
        #
        # "wintry mix" names the NWS advisory-grade mixed-precipitation event — the forecast phrasing for a
        # simultaneous fall of snow, sleet, and/or freezing rain that glazes and slushes roads, catwalks, and
        # stairs. It is the winter-precip TYPE sibling of the already-MEDIUM "sleet"/"snow"/"graupel" (a
        # traction/visibility nuisance that NWS carries in a Winter Weather Advisory), one gradient below the
        # warning-grade glaze-ice products "freezing rain"/"black ice"/"ice storm" (HIGH). Yet a report writing
        # "a wintry mix is expected across the site approach roads tonight" reached NO floored token and dropped
        # to LOW (verified live): "wintry mix" shares no substring with any floored weather token (its component
        # words "wintry"/"mix" are each unfloored and benign — "product mix", "concrete mix"), and no floored
        # token is a substring of it — the SAME advisory-tier precipitation miss class as "graupel" beside
        # "sleet". Floored MEDIUM, NOT HIGH: the taxonomy mirrors NWS product tiers — ADVISORY -> MEDIUM (this
        # sits beside "sleet"/"snow"/"graupel"/"dense fog"), WARNING -> HIGH (the ice-storm/freezing-rain glaze
        # products) — the exact advisory-vs-warning gradient already codified for dense fog (MEDIUM) beneath
        # freezing fog (HIGH); an actual wreck or injury on the glazed surface independently escalates via the
        # injury/medical floor. No plural entry: "wintry mix" is a mass/collective phrase (like the MEDIUM
        # "sleet"/"snow"/"graupel") — reporters do not write "wintry mixes". The QUALIFIED two-word phrase floors
        # only as a whole (\bwintry\s+mix\b), never bare "mix", the same qualified-phrase discipline as "dense
        # fog" (not bare "fog") / "red flag warning" (not bare "red flag") — zero operational false-positive
        # risk. Surfaced in the 2026-08-31 6:4x winter-precip rule-probe (advisory-tier mixed-precip sibling of
        # the MEDIUM sleet/snow/graupel, one gradient below the HIGH glaze-ice warning products).
        # "freezing drizzle" names the NWS advisory-grade glaze-ice precipitation — fine supercooled drops that
        # freeze on contact into a thin, treacherous coating on catwalks, ladders, handrails, and vehicle glass.
        # It is the light, advisory-tier cousin of the HIGH warning-grade "freezing rain" (NWS bands the two by
        # accumulation: freezing drizzle → Winter Weather Advisory, sustained freezing rain → Ice Storm Warning).
        # Yet a report writing "freezing drizzle is glazing the catwalk handrails" reached NO floored token and
        # dropped to LOW (verified live): "freezing drizzle" shares no substring with any floored weather token —
        # the HIGH glaze-ice phrase "freezing rain" is a different final word (\bfreezing\s+rain\b cannot match
        # "freezing drizzle"), "black ice"/"ice storm"/"freezing fog" are different phrases, and no floored token
        # is a substring of it — the SAME advisory-tier glaze miss class as "wintry mix"/"graupel" beside their
        # HIGH warning siblings. Floored MEDIUM, NOT HIGH: the taxonomy mirrors NWS product tiers — ADVISORY →
        # MEDIUM (this sits beside "sleet"/"snow"/"graupel"/"wintry mix"/"dense fog"), WARNING → HIGH (the
        # freezing-rain/ice-storm glaze products); a light glaze that actually injures a worker floors higher via
        # the independent injury/medical token. No plural entry: "freezing drizzle" is a mass noun (like the HIGH
        # "freezing rain" and the MEDIUM "sleet"/"snow"/"graupel"/"wintry mix") — reporters do not write "freezing
        # drizzles". Floored only as the QUALIFIED two-word phrase (\bfreezing\s+drizzle\b), never bare "drizzle",
        # which is domain-benign (light rain, "drizzle olive oil over the salad") and MUST stay LOW — the same
        # qualified-phrase discipline as "dense fog" (not bare "fog") / "wintry mix" (not bare "mix") / "red flag
        # warning" (not bare "red flag") — zero operational false-positive risk. Surfaced in the 2026-08-31 8:1x PM
        # winter-precip rule-probe (advisory-tier glaze sibling of the MEDIUM wintry mix/graupel, one gradient
        # below the HIGH freezing rain/black ice/ice storm warning products).
        # "cold weather advisory"/"cold weather advisories" is the CURRENT (2024 NWS hazard-simplification) name for
        # the ADVISORY-tier dangerous-cold product — the rename of "wind chill advisory" (already MEDIUM), issued
        # when cold is a nuisance/health risk but below the frostbite/hypothermia threshold that triggers the HIGH
        # WARNING product "extreme cold warning" (itself the 2024 rename of "wind chill warning"). It is the MEDIUM
        # cold peer of the MEDIUM "heat advisory" and the advisory-grade sibling of the HIGH "extreme cold warning" —
        # the SAME cold hazard, one NWS gradient lower. This was EXPLICITLY DEFERRED by the extreme-cold cluster
        # comment above ("the still-missing advisory tier — wind chill advisory / cold weather advisory — the MEDIUM
        # cold peer of the MEDIUM heat advisory — is DEFERRED to a later probe"); the wind-chill-advisory half shipped
        # 2026-08-31, and this closes the current-name half. Yet a report writing "a cold weather advisory is in
        # effect for the yard crew" reached NO floored token and dropped to LOW (verified live): "cold weather
        # advisory" shares no substring with any floored weather token — the HIGH cold events "cold wave"/"cold
        # snap"/"arctic blast" are not substrings, the HIGH product "extreme cold warning" is a different phrase
        # (\bextreme\s+cold\s+warning\b cannot match), the legacy MEDIUM "wind chill advisory" is a different first
        # two words, and bare "cold" is DELIBERATELY unfloored (a routine "it's cold outside"/"cold weather gear" must
        # stay LOW — FP guard below) — so the CURRENT-name advisory product scored strictly BELOW its own legacy name
        # "wind chill advisory" (MEDIUM), the exact current-name-beneath-legacy-name miss the heat family just closed
        # by pairing the legacy "excessive heat warning" with the current "extreme heat warning". Floored MEDIUM
        # beside "wind chill advisory"/"heat advisory" (the ADVISORY -> MEDIUM / WARNING -> HIGH gradient), NOT HIGH;
        # an actual cold injury (frostbite/hypothermia) on a worker independently escalates via injury/medical. UNLIKE
        # the mass-noun graupel/sleet, "advisory" is a COUNTABLE noun and NWS issues multi-zone "cold weather
        # advisories", a distinct token \bcold\s+weather\s+advisory\b cannot match — so the plural is added per the
        # singular->plural discipline (mirroring the wind chill advisory/advisories pair). Floored ONLY as the
        # qualified three-word phrase, never bare "cold"/"cold weather" (polysemous — "cold weather gear", "a cold
        # morning", "the cold shoulder", a "cold" illness — new FP-guard test test_bare_cold_stays_low re-verifies 4
        # benign sentences stay LOW), the same qualified-phrase discipline as wind chill advisory / dense fog / red
        # flag warning. Surfaced in the 2026-08-31 11:1x PM cold-advisory-current-name rule-probe (current-name half
        # of the deferred advisory-tier cold pair; legacy "wind chill advisory" half shipped earlier the same day).
        # "wind advisory"/"wind advisories" names the NWS ADVISORY-tier damaging-wind product — issued for sustained
        # winds of 31-39 mph (or gusts 46-57 mph), strong enough to topple trees/signs and make high-profile driving
        # dangerous but one gradient below the HIGH "high wind warning". It is the ADVISORY-grade sibling of the HIGH
        # wind family ("high winds"/"gale-force winds"/"high wind warning"), the exact wind counterpart of the MEDIUM
        # "heat advisory" (paired with HIGH heat wave) and "wind chill advisory" (paired with HIGH wind chill
        # warning) — completing the wind family's advisory->MEDIUM / warning->HIGH pair. Yet a report writing "a wind
        # advisory is in effect for the site" reached NO floored token and dropped to LOW (verified live): "wind
        # advisory" shares no substring with any floored token — the HIGH "high winds" is a different phrase
        # (\bhigh\s+winds\b cannot match), \bwindstorm\b/\bstorm\b share no substring, "wind damage" is a different
        # final word, and bare "wind" is DELIBERATELY unfloored (a routine "the wind picked up"/"second wind" must
        # stay LOW — FP guard). Floored MEDIUM beside "heat advisory"/"wind chill advisory" (the ADVISORY -> MEDIUM /
        # WARNING -> HIGH gradient), NOT HIGH; an actual injury or downed line from the wind independently escalates
        # via injury/medical or electrical. UNLIKE the mass-noun graupel/sleet, "advisory" is a COUNTABLE noun and
        # NWS issues multi-zone "wind advisories", a distinct token \bwind\s+advisory\b cannot match — so the plural
        # is added per the singular->plural discipline (mirroring the wind chill advisory/advisories pair). Floored
        # ONLY as the qualified two-word phrase, never bare "wind" (polysemous — "second wind", "a wind of change",
        # "wind the clock", "wind down") — the same qualified-phrase discipline as wind chill advisory / heat
        # advisory / red flag warning. Surfaced in the 2026-09-01 12:4x AM wind-product rule-probe (advisory-tier
        # wind sibling of the MEDIUM heat/cold advisories; paired with the new HIGH high wind warning).
        # "winter weather advisory"/"winter weather advisories" names the NWS ADVISORY-tier winter-precip product —
        # issued for a mix of snow/sleet/freezing rain / light accumulations that cause slick roads and hazardous
        # travel but stay below warning criteria. It is the ADVISORY-grade sibling of the HIGH winter-storm family
        # ("winter storm warning"/"blizzard"/"ice storm"/"snowstorm"), the winter-storm counterpart of the MEDIUM
        # "heat advisory" (paired with HIGH heat wave), "wind advisory" (paired with HIGH high wind warning), and
        # "wind chill advisory" (paired with HIGH wind chill warning) — completing the winter-storm family's
        # advisory->MEDIUM / warning->HIGH pair (the warning half "winter storm warning" already floors HIGH via
        # the bare \bstorm\b token). Yet a report writing "a winter weather advisory is in effect for the overnight
        # crew" reached NO floored token and dropped to LOW (verified live): "winter weather advisory" shares no
        # substring with any floored token — \bstorm\b/\bblizzard\b/\bice\s+storm\b/\bsnowstorm\b share no substring
        # (there is no "storm" in "winter weather advisory"), the MEDIUM "snow"/"sleet"/"wintry mix" are different
        # words, and bare "winter"/"winter weather" is DELIBERATELY unfloored (a routine "the winter weather on
        # their break" / "winter maintenance advisory" must stay LOW — FP guard). Floored MEDIUM beside "wind
        # advisory"/"wind chill advisory"/"heat advisory" (the ADVISORY -> MEDIUM / WARNING -> HIGH gradient), NOT
        # HIGH; an actual slip-injury or vehicle wreck on the slick surface independently escalates via
        # injury/medical. UNLIKE the mass-noun graupel/sleet, "advisory" is a COUNTABLE noun and NWS issues
        # multi-zone "winter weather advisories" ("winter weather advisories were issued for the northern
        # counties"), a distinct token \bwinter\s+weather\s+advisory\b cannot match — so the plural is added per the
        # singular->plural discipline (mirroring the wind advisory/advisories pair). Floored ONLY as the qualified
        # three-word phrase, never bare "winter"/"winter weather" (polysemous — "the winter weather", "winter
        # weather gear", "winter maintenance") — the same qualified-phrase discipline as wind advisory / wind chill
        # advisory / heat advisory. Surfaced in the 2026-09-01 3:4x AM winter-storm-advisory rule-probe (advisory-
        # tier winter-storm sibling of the MEDIUM heat/wind/cold advisories; the winter-storm family's advisory
        # half, its warning half "winter storm warning" already HIGH via bare "storm").
        #
        # "avalanche watch"/"avalanche watches" names the avalanche-center/NWS WATCH-tier snow-slope product — issued
        # when avalanche conditions are developing/possible (a step below the "avalanche warning" that floors weather
        # HIGH, which means avalanches are imminent or occurring). It is the anticipatory sibling of that HIGH warning
        # and completes the avalanche family's watch->MEDIUM / warning->HIGH pair, the same advisory/watch -> MEDIUM /
        # warning -> HIGH gradient the wind, cold, heat, frost/freeze, and winter-storm families already carry (the
        # 2026-09-01 5:1x AM cycle that shipped "avalanche warning"/"snow avalanche" HIGH explicitly deferred this
        # watch half). Yet a report writing "an avalanche watch is in effect for the northern ranges" reached NO
        # floored token and dropped to LOW (verified live): "avalanche watch" shares no substring with any floored
        # token — \bavalanche\s+warning\b cannot match the different final word, \bsnow\s+avalanche\b is a different
        # phrase, bare "avalanche"/"avalanches" is DELIBERATELY unfloored (figurative "an avalanche of emails/tickets/
        # paperwork" — FP guard test_bare_avalanche_figurative_stays_low), and bare "watch" is not a token (a security
        # "watch"/night watch/wristwatch must stay LOW). Floored MEDIUM beside the other watch/advisory products, NOT
        # HIGH; if a slope actually releases the physical "snow avalanche"/"avalanche warning" independently floors
        # HIGH and a worker buried floors critical via injury/medical. "watch" is a COUNTABLE noun and avalanche
        # centers issue multi-zone "avalanche watches", a distinct token \bavalanche\s+watch\b cannot match — so the
        # plural is added per the singular->plural discipline (mirroring wind advisory/advisories). Floored ONLY as the
        # qualified two-word phrase (\bavalanche\s+watch\b), never bare "avalanche" or bare "watch" — the same
        # qualified-phrase discipline as avalanche warning / snow avalanche. This is the FIRST "watch"-tier token; it
        # rides at the advisory MEDIUM floor because an avalanche WATCH and an advisory carry the same be-prepared
        # urgency one gradient below a WARNING. Surfaced in the 2026-09-01 6:4x AM avalanche-watch rule-probe
        # (completing the avalanche watch/warning pair deferred by the 5:1x AM avalanche-warning ship).
        # "freeze watch"/"freeze watches" names the NWS WATCH-tier freeze product — issued when significant,
        # widespread freezing temperatures are POSSIBLE within the next 24-36 hours during the growing season, a
        # step below the "freeze warning" that floors weather HIGH (freezing temps imminent or occurring, lethal to
        # crops/pipes/exposed workers). It is the anticipatory sibling of that HIGH warning and completes the
        # freeze family's watch->MEDIUM / warning->HIGH pair — the same watch/advisory -> MEDIUM / warning -> HIGH
        # gradient the wind, cold, heat, winter-storm, and avalanche families already carry (the 2026-09-01 cycle
        # that shipped "freeze warning" HIGH left the watch half open, and the avalanche watch ship just codified
        # this exact watch-tier pattern). Yet a report writing "a freeze watch is in effect for the tank farm" reached
        # NO floored token and dropped to LOW (verified live): "freeze watch" shares no substring with any floored
        # token — the HIGH "freeze warning" is a different final word (\bfreeze\s+warning\b cannot match "freeze
        # watch"), the HIGH glaze phrases "flash freeze"/"freezing rain"/"freezing fog" are different words
        # (\bflash\s+freeze\b / \bfreezing\s+rain\b cannot match "freeze watch"), the bare verb "freeze" is
        # DELIBERATELY unfloored (hiring freeze, freeze frame, freeze-tag — see the freeze warning rationale), and bare
        # "watch" is not a token (a security "watch"/night watch/wristwatch must stay LOW). Floored MEDIUM beside the
        # other watch/advisory products, NOT HIGH; if the temperature actually drops the "freeze warning" independently
        # floors HIGH and a cold-injured worker floors critical via injury/medical. "watch" is a COUNTABLE noun and
        # forecast offices issue multi-zone "freeze watches", a distinct token \bfreeze\s+watch\b cannot match — so the
        # plural is added per the singular->plural discipline (mirroring avalanche watch/watches, wind advisory/
        # advisories). Floored ONLY as the qualified two-word phrase (\bfreeze\s+watch\b), never bare "freeze" or bare
        # "watch" — the same qualified-phrase discipline as freeze warning / avalanche watch. The figurative
        # non-adjacent "a hiring freeze ... put new reqs on a watch" carries neither adjacent phrase and stays LOW (FP
        # guard test_freeze_watch_needs_adjacency). Surfaced in the 2026-09-01 8:2x AM freeze-watch rule-probe
        # (completing the freeze watch/warning pair left open by the freeze-warning ship, mirroring avalanche watch).
        # "high wind watch"/"high wind watches" names the NWS WATCH-tier high-wind product — issued when sustained
        # winds >=40 mph or gusts >=58 mph are POSSIBLE within the next 12-48 hours, a step below the "high wind
        # warning" that floors weather HIGH (those winds imminent or occurring, the damaging severe-wind hazard that
        # downs lines and topples equipment on exposed crews). It is the anticipatory sibling of that HIGH warning and
        # completes the wind family's watch->MEDIUM / warning->HIGH ladder: the family already carries the ADVISORY
        # tier ("wind advisory" MEDIUM) and the WARNING tier ("high wind warning" HIGH beside bare "high winds"), but
        # the WATCH tier between them was open — the same watch/advisory->MEDIUM / warning->HIGH gradient the freeze,
        # avalanche, cold, heat, and winter-storm families already carry (the freeze/avalanche watch ships codified this
        # exact watch-tier pattern). Yet a report writing "a high wind watch is posted for the tower crew" reached NO
        # floored token and dropped to LOW (verified live): "high wind watch" shares no substring with any floored
        # token — the HIGH "high wind warning" is a different final word (\bhigh\s+wind\s+warning\b cannot match "high
        # wind watch"), the HIGH bare "high winds" is PLURAL (\bhigh\s+winds\b needs the "s" and cannot match the
        # singular "high wind" inside "high wind watch"), the MEDIUM "wind advisory" is a different phrase, and bare
        # "watch" is not a token (a security "watch"/night watch/wristwatch must stay LOW). Floored MEDIUM beside the
        # other watch/advisory products, NOT HIGH; if the winds actually arrive the "high wind warning"/"high winds"
        # independently floor HIGH and a struck worker floors critical via injury/medical. "watch" is a COUNTABLE noun
        # and forecast offices issue multi-zone "high wind watches", a distinct token \bhigh\s+wind\s+watch\b cannot
        # match — so the plural is added per the singular->plural discipline (mirroring freeze watch/watches, avalanche
        # watch/watches, wind advisory/advisories). Floored ONLY as the qualified three-word phrase
        # (\bhigh\s+wind\s+watch\b), never bare "wind"/"watch" — the same qualified-phrase discipline as high wind
        # warning / freeze watch / avalanche watch. The figurative non-adjacent "kept a high wind at his back ... on
        # watch" carries neither adjacent phrase and stays LOW (FP guard test_high_wind_watch_needs_adjacency).
        # Surfaced in the 2026-09-01 9:4x AM high-wind-watch rule-probe (completing the wind family's watch tier left
        # open beneath high wind warning, mirroring the freeze/avalanche watch ships).
        # "gale watch"/"gale watches" names the NWS marine WATCH-tier gale product — issued when sustained winds of
        # 34-47 knots (39-54 mph, gale force) are POSSIBLE within the next 12-48 hours, a step below the "gale warning"
        # that floors weather HIGH (those winds imminent or expected, the severe-wind marine hazard that capsizes small
        # craft and endangers harbor/offshore crews). It is the anticipatory sibling of that HIGH warning and completes
        # the gale product's watch->MEDIUM / warning->HIGH pair — the same watch->MEDIUM / warning->HIGH gradient the
        # wind, freeze, cold, heat, and avalanche families already carry (the freeze/high-wind/heat/cold watch ships
        # codified this exact watch-tier pattern). Yet a report writing "a gale watch is posted for the harbor crew"
        # reached NO floored token and dropped to LOW (verified live): "gale watch" shares no substring with any floored
        # token — the HIGH "gale warning" is a different final word (\bgale\s+warning\b cannot match "gale watch"), the
        # HIGH "gale-force winds"/"gale force winds" are different words, and bare "watch" is not a token (a security
        # "watch"/night watch/wristwatch must stay LOW). The bare root "gale" is DELIBERATELY EXCLUDED as polysemous
        # (a proper name — "Gale from accounting"; the figurative "a gale of laughter"), exactly as it is for the
        # HIGH "gale warning"/"gale-force winds" entries — so only the adjacent two-word phrase fires. Floored MEDIUM
        # beside the other watch/advisory products, NOT HIGH; if the winds actually arrive the "gale warning"/
        # "gale-force winds" independently floor HIGH and a struck/overboard worker floors critical via injury/medical.
        # "watch" is a COUNTABLE noun and forecast offices issue multi-zone "gale watches", a distinct token
        # \bgale\s+watch\b cannot match — so the plural is added per the singular->plural discipline (mirroring freeze
        # watch/watches, high wind watch/watches, avalanche watch/watches). Floored ONLY as the qualified two-word
        # phrase (\bgale\s+watch\b), never bare "gale"/"watch" — the same qualified-phrase discipline as gale warning /
        # high wind watch / freeze watch. The figurative non-adjacent "a gale of laughter ... stood watch" carries
        # neither adjacent phrase and stays LOW (FP guard test_gale_watch_needs_adjacency). Surfaced in the 2026-09-01
        # 8:1x PM gale-watch rule-probe (completing the gale product's watch tier left open beneath gale warning,
        # mirroring the freeze/high-wind/cold/heat watch ships).
        # "small craft advisory"/"small craft advisories" names the NWS ADVISORY-tier marine wind/sea product — issued
        # when sustained winds of ~22-33 knots (25-38 mph) and/or hazardous seas make conditions dangerous for small
        # vessels, one NWS gradient BELOW the WATCH-tier "gale watch" (gale-force winds POSSIBLE) and two below the
        # HIGH warning-tier "gale warning"/"hazardous seas" (imminent/occurring). It is the entry rung of the marine-
        # wind ladder — small craft advisory MEDIUM -> gale watch MEDIUM -> gale warning HIGH — and the marine sibling
        # of the land advisory products "wind advisory"/"wind chill advisory"/"heat advisory" already floored MEDIUM.
        # Yet a report writing "a small craft advisory is in effect for the bay crew" reached NO floored token and
        # dropped to LOW (verified live): "small craft advisory" shares no substring with any floored token — the HIGH
        # "gale warning"/"hazardous seas"/"high surf warning" are different words, the MEDIUM "gale watch"/"wind
        # advisory" are different phrases, and there is no bare "craft"/"advisory"/"small" token. The bare noun "small
        # craft" is DELIBERATELY EXCLUDED as polysemous — it is the ordinary maritime term for a small boat ("the crew
        # launched a small craft", "the small craft was moored overnight"), routine ops language that must stay LOW —
        # so only the adjacent qualified phrase fires, the same qualified-phrase discipline that floored "gale watch"/
        # "wind advisory" while leaving bare "gale"/"wind"/"craft" unfloored. Floored MEDIUM beside "gale watch"/"wind
        # advisory", NOT HIGH; if the winds/seas actually endanger a vessel the "gale warning"/"hazardous seas"
        # independently floor HIGH and a capsizing/overboard worker floors critical via injury/medical. "advisory" is a
        # COUNTABLE noun and forecast offices issue multi-zone "small craft advisories", a distinct token
        # \bsmall\s+craft\s+advisory\b cannot match — so the plural is added per the singular->plural discipline
        # (mirroring wind advisory/advisories, gale watch/watches). The non-adjacent "a small craft ... under advisory
        # from counsel" carries neither the boat-hazard sense nor the phrase and stays LOW (FP guard
        # test_small_craft_advisory_needs_adjacency). Surfaced in the 2026-09-01 11:1x PM small-craft-advisory rule-
        # probe (completing the marine-wind ladder's advisory rung left open beneath gale watch, mirroring the land
        # wind/heat/cold advisory ships).
        "medium":   ["heavy rain", "wind damage", "snow", "sleet", "frost", "heat advisory", "dense fog",
                     "graupel", "wintry mix", "freezing drizzle",
                     "wind advisory", "wind advisories",
                     "wind chill advisory", "wind chill advisories",
                     "cold weather advisory", "cold weather advisories",
                     "winter weather advisory", "winter weather advisories",
                     "avalanche watch", "avalanche watches",
                     "freeze watch", "freeze watches",
                     "high wind watch", "high wind watches",
                     "excessive heat watch", "excessive heat watches",
                     "extreme heat watch", "extreme heat watches",
                     "wind chill watch", "wind chill watches",
                     "extreme cold watch", "extreme cold watches",
                     "gale watch", "gale watches",
                     "small craft advisory", "small craft advisories",
                     "freezing spray advisory", "freezing spray advisories"],
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
