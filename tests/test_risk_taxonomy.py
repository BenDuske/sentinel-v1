"""Grounded rule-layer taxonomy — each category maps to its expected severity FLOOR.

Pure, deterministic, no LLM/network. This is the differentiator: a defensible, auditable floor.
"""
from sentinel import risk

_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _atleast(actual, floor):
    return _RANK[actual] >= _RANK[floor]


def test_no_signal_defaults_to_low():
    sev, reasons = risk.rule_layer("Routine note: replaced the lobby light bulb.")
    assert sev == "low"
    assert reasons and "no risk taxonomy signals" in reasons[0].lower()


# (text, minimum-expected-severity, category-substring-expected-in-rationale)
CASES = [
    ("Two workers injured, one hospitalized after a fall", "high", "injury/medical"),
    ("Fatality on site after equipment failure", "critical", "injury/medical"),
    # Respiratory distress (still breathing but struggling) must reach the HIGH floor — the lay
    # phrasings a reporter writes previously matched nothing and dropped to LOW while apnea ("not
    # breathing") sat at critical.
    ("Employee is having trouble breathing at their desk", "high", "injury/medical"),
    ("Visitor can't breathe and is turning blue", "high", "injury/medical"),
    ("Resident complains of shortness of breath and chest tightness", "high", "injury/medical"),
    ("Worker gasping for air near the loading dock", "high", "injury/medical"),
    # Lay synonyms for the acute-emergency critical floor: "heart attack" (== "cardiac arrest")
    # and anaphylaxis previously scored LOW while "cardiac arrest" scored CRITICAL.
    ("Employee is having a heart attack at their desk", "critical", "injury/medical"),
    ("Anaphylaxis after a bee sting; epinephrine administered", "critical", "injury/medical"),
    # "myocardial infarction" is the clinical twin of "heart attack" (critical) — an EMS/medical
    # report writes it this way, yet it previously matched nothing and dropped to LOW. Both cases
    # isolate on the term (no other critical/high token fires).
    ("Radiology tech suffered a myocardial infarction mid-shift", "critical", "injury/medical"),
    ("Acute myocardial infarction confirmed by the responding paramedic", "critical", "injury/medical"),
    # "ventricular fibrillation" is the lethal shockable rhythm of a pulseless cardiac arrest (critical)
    # — an AED/monitor report writes it this way, yet it previously matched nothing and dropped to LOW.
    # Both cases isolate on the term (no other critical/high token fires).
    ("Patient found in ventricular fibrillation; AED advised a shock", "critical", "injury/medical"),
    ("Confirmed ventricular fibrillation on the cardiac monitor", "critical", "injury/medical"),
    # "cardiac tamponade" / "pericardial tamponade" is an immediately life-threatening compression of
    # the heart (critical) an EMS/ED/echo report names directly, yet it previously matched nothing and
    # dropped to LOW. Both cases isolate on the term (no other critical/high token fires); the bare
    # word "tamponade" is NOT floored (polysemous therapeutic maneuver — balloon/uterine tamponade).
    ("Entrant developed cardiac tamponade after the chest impact", "critical", "injury/medical"),
    ("Responding medic confirmed pericardial tamponade on the echo", "critical", "injury/medical"),
    # Heart-wall rupture is the terminal MECHANICAL complication of an MI (critical) — "cardiac
    # rupture" / "myocardial rupture" / "ventricular rupture" — an almost uniformly fatal blow-out an
    # ED/cath-lab/autopsy report names directly, yet each previously matched nothing and dropped to
    # LOW. Each case isolates on the term (no other critical/high token fires); the bare word
    # "rupture" is NOT floored (polysemous — water main / disc / spleen rupture across categories).
    ("Patient arrested from cardiac rupture minutes after the infarct", "critical", "injury/medical"),
    ("Autopsy confirmed left ventricular free-wall myocardial rupture", "critical", "injury/medical"),
    ("Surgeon documented a ventricular rupture at thoracotomy", "critical", "injury/medical"),
    # The lay name for an intracranial hemorrhage — a "brain bleed" — floors at HIGH like its
    # clinical umbrella "intracranial/cerebral/brain hemorrhage" (NOT critical: can be a slow chronic
    # subdural), yet previously matched nothing and dropped to LOW because the bare token is "bleed",
    # not the already-floored "bleeding". Each case isolates on the added phrase; bare "bleed"/"brain"
    # stay unfloored, so a benign non-adjacent sentence (FP guard below) stays LOW.
    ("CT confirmed a large brain bleed", "high", "injury/medical"),
    ("Both patients presented with brain bleeds on imaging", "high", "injury/medical"),
    ("Neurologist noted an active bleed on the brain", "high", "injury/medical"),
    ("The scan showed she had bled on the brain overnight", "high", "injury/medical"),
    # Plain-English synonyms for "unconscious" (critical): "lost consciousness" / "loss of
    # consciousness" previously matched nothing and dropped to LOW while "unconscious" scored
    # critical. A transient faint reaches the HIGH floor.
    ("Worker lost consciousness on the floor", "critical", "injury/medical"),
    ("Reporter noted a brief loss of consciousness after the fall", "critical", "injury/medical"),
    ("Employee fainted at her desk and was helped up", "high", "injury/medical"),
    # "CPR" is an unambiguous lay marker of a life-threatening arrest and must reach the same
    # critical floor as "cardiac arrest"/"not breathing"; the bare acronym previously dropped to LOW.
    ("Coworker collapsed; bystanders started CPR immediately", "critical", "injury/medical"),
    ("We are performing CPR while waiting for the ambulance", "critical", "injury/medical"),
    # A convulsion reaches the HIGH floor — the "convuls-" forms safely cover what the polysemous
    # "seizure" cannot, and previously matched nothing and dropped to LOW.
    ("Visitor is convulsing on the floor of the atrium", "high", "injury/medical"),
    ("Resident had convulsions after the fall", "high", "injury/medical"),
    # EMS/lay phrasings for cardiac arrest must reach the same critical floor as "cardiac arrest"/
    # "cpr" — "no pulse" / "pulseless" / "no heartbeat" previously matched nothing and dropped to LOW.
    ("Collapsed worker has no pulse; CPR in progress", "critical", "injury/medical"),
    ("Bystander reports the man is pulseless and not moving", "critical", "injury/medical"),
    ("Patient found with no heartbeat in the stairwell", "critical", "injury/medical"),
    # Apnea reported in the natural past tense / contraction must reach the same critical floor as
    # "not breathing" — "stopped breathing" / "no longer breathing" / "isn't breathing" previously
    # matched nothing (the substring "not breathing" does not cover them) and dropped to LOW.
    ("The infant stopped breathing and turned blue", "critical", "injury/medical"),
    ("Found the resident on the floor, no longer breathing", "critical", "injury/medical"),
    ("He isn't breathing — starting rescue breaths now", "critical", "injury/medical"),
    # Named vascular emergencies must reach the same critical floor as "cardiac arrest"/"heart
    # attack" — "aneurysm" (incl. the "aneurism" lay spelling) and pulmonary "embolism" previously
    # matched nothing and dropped to LOW while the polysemous "stroke"/"seizure" stay excluded.
    ("Employee collapsed with a suspected brain aneurysm", "critical", "injury/medical"),
    ("Reporter wrote it up as a possible aneurism in the break room", "critical", "injury/medical"),
    ("Worker down; medics suspect a pulmonary embolism", "critical", "injury/medical"),
    # The clinical phrase "respiratory arrest" (and "cardiopulmonary arrest") must reach the same
    # critical floor as its twin "cardiac arrest" — an equally life-threatening arrest that
    # previously matched nothing and dropped to LOW purely on which clinical term the reporter chose
    # ("respiratory arrest" is not a substring of the apnea phrasings that already floor).
    ("Patient went into respiratory arrest before EMS arrived", "critical", "injury/medical"),
    ("Responders report cardiopulmonary arrest at the scene", "critical", "injury/medical"),
    # "cardiorespiratory arrest" is the British/international synonym of "cardiopulmonary arrest" and
    # must reach the same critical floor — the identical immediately-fatal event, previously matched
    # nothing and dropped to LOW purely on which orthographic tradition the reporter learned (it is
    # not a substring of "cardiopulmonary arrest" nor of the deliberately-excluded bare "arrest").
    ("EMS logged a cardiorespiratory arrest on the loading dock", "critical", "injury/medical"),
    # The plural noun "burns" must reach the same HIGH floor as the singular "burn"/"burned" —
    # "severe burns" / "third-degree burns" previously matched nothing (\bburn\b does not match
    # "burns") and dropped to LOW. "impaled" is an unambiguous severe-trauma term at the same floor.
    ("Worker suffered severe burns on both hands", "high", "injury/medical"),
    ("Two people treated for third-degree burns after the flash", "high", "injury/medical"),
    ("Worker impaled on a length of rebar at the site", "high", "injury/medical"),
    # The NOUN "impalement" is the word-form twin of the participle "impaled" (already HIGH) and must
    # reach the same floor — an isolated report of it previously matched nothing (\bimpaled\b does not
    # match "impalement") and dropped to LOW, the same participle-vs-noun gap as concussion/concussed.
    # Both cases below isolate on the new noun (no other floored token), so removing it regresses LOW.
    ("Traumatic impalement on the fence stake; the crew freed the worker", "high", "injury/medical"),
    ("Responders reported an impalement after the guardrail sheared away", "high", "injury/medical"),
    # A "punctured lung" (traumatic pneumothorax) is serious acute chest trauma and must reach the
    # same HIGH floor as impaled/blood loss — an isolated report of it previously matched nothing and
    # dropped to LOW. The MULTI-WORD phrase cannot fire from bare polysemous "punctured" ("punctured
    # tire"). No other high/critical token appears in these cases, so they isolate on the new term.
    ("The worker suffered a punctured lung when the scaffold gave way", "high", "injury/medical"),
    ("Punctured lung suspected after the rib cage took the blow", "high", "injury/medical"),
    # A "puncture wound" is penetrating trauma and must reach the same conservative HIGH floor as
    # punctured lung/impaled — an isolated report previously matched nothing and dropped to LOW while
    # weapon-implying "stab wound"/"gunshot wound" already floor CRITICAL. HIGH (not critical) because
    # a puncture wound is routinely minor (nail/needle/bite) and carries no weapon connotation. The
    # MULTI-WORD phrase cannot fire from bare polysemous "puncture" ("punctured tire", "puncture-
    # resistant gloves"). No other high/critical token appears in these cases, so they isolate.
    ("The line worker sustained a deep puncture wound to the thigh", "high", "injury/medical"),
    ("Medics dressed multiple puncture wounds after the tool slipped", "high", "injury/medical"),
    # "decompression sickness" (DCS, "the bends") is the diving/hyperbaric/caisson injury and must reach
    # the same conservative HIGH floor as frostbite/hypothermia/punctured lung — an isolated report
    # previously matched nothing and dropped to LOW. HIGH (not critical) because DCS is routinely
    # survivable and treated by hyperbaric recompression. The MULTI-WORD phrase cannot fire from bare
    # polysemous "decompression" (archive/pressure-vessel/needle-chest decompression). No other high/
    # critical token appears in these cases, so they isolate on the new term.
    ("The diver developed decompression sickness after the rapid ascent", "high", "injury/medical"),
    ("Caisson worker suffered decompression sickness on the job", "high", "injury/medical"),
    # "crush syndrome" (traumatic rhabdomyolysis) is the systemic, potentially-fatal complication of
    # prolonged crushing/entrapment and must reach the same conservative HIGH floor as sepsis/near-
    # drowning/decompression sickness — an isolated report previously matched nothing and dropped to LOW.
    # HIGH (not critical) because, like sepsis, it evolves over hours after extrication and is treatable.
    # The MULTI-WORD phrase cannot fire from bare polysemous "crush"/"crushed". Both cases below avoid
    # "collapsed"/"injured"/"fall" so they isolate on the new term — removing it regresses them to LOW.
    ("Crush syndrome suspected after the worker was freed from under the machinery", "high", "injury/medical"),
    ("Medics treated the crew for crush syndrome following the prolonged entrapment", "high", "injury/medical"),
    # "sepsis" is a life-threatening infection response and must reach the HIGH floor; "septic shock"
    # is its terminal circulatory-collapse form and must reach CRITICAL alongside anaphylactic/
    # respiratory-arrest. Both previously matched nothing and dropped to LOW. The bare adjective
    # "septic" is deliberately EXCLUDED (septic tank/system polysemy) - whole-word matching keeps
    # \bsepsis\b and the two-word \bseptic shock\b from firing off it. No other high/critical token
    # appears in these cases, so they isolate on the new terms.
    ("The patient is in sepsis after the wound became infected", "high", "injury/medical"),
    ("Responders report the worker went into septic shock", "critical", "injury/medical"),
    # "near drowning"/"near-drowning" (a nonfatal submersion event) must reach the HIGH floor; both
    # the spaced and hyphenated forms previously matched nothing and dropped to LOW. Bare "drowning"/
    # "drowned" is deliberately EXCLUDED (polysemy: "drowning in debt", the idiom "drowned out"), so
    # these isolate on the new two-word/hyphenated phrase — no other high/critical token appears.
    ("Lifeguard reports a near drowning at the aquatics center", "high", "injury/medical"),
    ("Child recovered after a near-drowning; medics en route", "high", "injury/medical"),
    # The participle "amputated" must reach the same critical floor as the noun "amputation" — an
    # acute report is written "his arm was amputated" / "amputated finger", which previously matched
    # nothing and dropped to LOW purely on verb-vs-noun word form.
    ("Machinist's hand was amputated in the press", "critical", "injury/medical"),
    ("Amputated finger recovered at the scene; medics en route", "critical", "injury/medical"),
    # "decapitation"/"decapitated" is an unambiguous, virtually always-fatal trauma a reporter names
    # directly ("the worker was decapitated by the machine", "traumatic decapitation at the press"),
    # yet BOTH the noun and the participle previously matched nothing and dropped to LOW — a whole
    # severe-trauma word simply absent from the critical floor (same class as amputated/impaled).
    ("The worker was decapitated by the unguarded machine", "critical", "injury/medical"),
    ("Traumatic decapitation at the stamping press; responders en route", "critical", "injury/medical"),
    # "dismemberment"/"dismembered" is a catastrophic trauma on par with decapitation/amputation, yet
    # BOTH the noun and the participle previously matched nothing and dropped to LOW — a whole severe-
    # trauma word simply absent from the critical floor (same class as amputated/decapitated). Both
    # cases below isolate on the new terms (no other floored token), so removing them regresses to LOW.
    ("The worker was dismembered by the unguarded conveyor", "critical", "injury/medical"),
    ("Traumatic dismemberment at the rolling mill; responders en route", "critical", "injury/medical"),
    # "evisceration"/"eviscerated" is a catastrophic trauma on par with decapitation/dismemberment, yet
    # BOTH the noun and the participle previously matched nothing and dropped to LOW — a whole severe-
    # trauma word simply absent from the critical floor (same class as amputated/decapitated/dismembered).
    # Both cases below isolate on the new terms (no other floored token), so removing them regresses to LOW.
    ("The worker was eviscerated by the unguarded machine", "critical", "injury/medical"),
    ("Traumatic abdominal evisceration at the press; responders en route", "critical", "injury/medical"),
    # The PRESENT participles "amputating"/"decapitating"/"dismembering" must reach the same critical
    # floor as their nouns/past participles — an ACTIVE machine-trauma report writes "the press was
    # amputating fingers", which previously matched neither the noun nor the past participle ("-ing" is
    # a distinct token) and dropped to LOW/MEDIUM (same verb-form asymmetry class as exsanguinating).
    # Each case isolates on the new term (no other floored token), so removing it regresses below critical.
    ("The industrial press was amputating fingers on every cycle", "critical", "injury/medical"),
    ("The rotating blade was decapitating the worker who reached in", "critical", "injury/medical"),
    ("The auger was dismembering the worker who fell into it", "critical", "injury/medical"),
    # "strangulation" is a fatal airway-occlusion trauma (sibling of asphyxiation/suffocation) — the noun
    # previously matched nothing and dropped to LOW unless a coincident token (death/assault) fired. Both
    # cases below isolate on the noun (no other floored token), so removing "strangulation" regresses to LOW.
    ("Manual strangulation of the worker caught in the machine", "critical", "injury/medical"),
    ("Confined-space incident: strangulation on the conveyor guard", "critical", "injury/medical"),
    # "agonal" is the terminal gasping respiration of a dying/arresting patient (respiratory sibling of
    # the already-critical respiratory arrest / not-breathing) — the term previously matched nothing and
    # dropped to LOW. Both cases below isolate on "agonal" (no other floored token), so removing it
    # regresses to LOW. The word-boundary cases further down guard the diagonal/hexagonal substring class.
    ("Patient has agonal breathing; EMS performing resuscitation", "critical", "injury/medical"),
    ("Found the man with agonal respirations on the loading dock", "critical", "injury/medical"),
    # "tension pneumothorax" is an immediately-lethal obstructive-shock emergency (respiratory sibling of
    # the already-critical cardiac tamponade) — the qualified phrase previously matched nothing and
    # dropped to LOW. Both cases below isolate on the phrase (no other floored token), so removing
    # "tension pneumothorax" regresses to LOW.
    ("Responders report a tension pneumothorax; emergency needle decompression underway", "critical", "injury/medical"),
    ("Tension pneumothorax confirmed on scene; chest decompression performed by EMS", "critical", "injury/medical"),
    # "asystole" is the flatline rhythm that IS a pulseless cardiac arrest (critical) — the non-shockable
    # sibling of the already-critical "ventricular fibrillation" — an AED/monitor/EMS report writes it
    # this way, yet it previously matched nothing and dropped to LOW. Both cases isolate on the term (no
    # other floored token), so removing "asystole" regresses to LOW.
    ("Monitor showed asystole; the rhythm strip was flat", "critical", "injury/medical"),
    ("Asystole confirmed by the responding paramedic on the cardiac monitor", "critical", "injury/medical"),
    # "cardiogenic shock" / "hypovolemic shock" are the immediately-lethal shock siblings of the
    # already-critical "septic shock" — pump-failure shock (post-MI/arrest) and circulatory collapse
    # from massive blood loss — yet both previously matched nothing and dropped to LOW. Each case
    # isolates on the term (no other floored token), so removing the phrase regresses to LOW.
    ("EMS on scene reports the patient is in cardiogenic shock", "critical", "injury/medical"),
    ("Cardiogenic shock developed; pressors started per protocol", "critical", "injury/medical"),
    ("Responders note hypovolemic shock; rapid transfusion begun", "critical", "injury/medical"),
    ("Hypovolemic shock on arrival, transferred to the trauma bay", "critical", "injury/medical"),
    # "hemorrhagic shock" is the clinical synonym of "hypovolemic shock" for the blood-loss case —
    # circulatory collapse from massive hemorrhage. It previously fired only HIGH via the bare
    # "hemorrhagic" bleeding signal; the added critical phrase escalates it to the shock-sibling floor.
    # Each case isolates on the phrase, so removing it regresses to HIGH (still not critical).
    ("Patient in hemorrhagic shock from the leg wound; massive transfusion begun", "critical", "injury/medical"),
    ("Class IV hemorrhagic shock on arrival, transferred to the trauma bay", "critical", "injury/medical"),
    # "status epilepticus" is the non-stopping, life-threatening escalation of the already-HIGH
    # convulsion floor — a continuous seizure that causes brain injury/death if not aborted, the term
    # an EMS/ED report names directly — yet it previously matched nothing and dropped to LOW. Each
    # case isolates on the phrase (no other floored token), so removing it regresses to LOW.
    ("Patient in status epilepticus on arrival; benzodiazepines administered", "critical", "injury/medical"),
    ("Convulsive status epilepticus reported; the fits would not stop", "critical", "injury/medical"),
    # "aortic dissection" is a tear splitting the aortic wall — an immediately-fatal arterial
    # catastrophe (distinct from the already-critical "aneurysm" bulge) the term an EMS/CT report
    # names directly — yet it previously matched nothing and dropped to LOW. Each case isolates on
    # the phrase (no other floored token), so removing it regresses to LOW.
    ("Acute aortic dissection on CT; to the OR emergently", "critical", "injury/medical"),
    ("EMS reports a Type A aortic dissection on arrival", "critical", "injury/medical"),
    # "aortic rupture"/"ruptured aorta" is the frank wall blow-out — the terminal twin of the already-
    # critical "aortic dissection" — yet both word orders previously matched nothing and dropped to LOW.
    # Each case isolates on the phrase (no other floored token), so removing it regresses to LOW.
    ("Confirmed aortic rupture on CT with a massive hemothorax", "critical", "injury/medical"),
    ("The patient suffered a ruptured aorta before EMS arrived", "critical", "injury/medical"),
    # "subarachnoid hemorrhage" is the clinical name for a ruptured cerebral aneurysm (already
    # critical) — a hyperacute, ~50%-mortality catastrophe — yet it previously fired only HIGH via the
    # bare "hemorrhage" bleeding term. Both spellings must reach critical. Each case isolates on the
    # qualified phrase (no "aneurysm"/"death" token), so removing it regresses to the HIGH bleeding floor.
    ("Acute subarachnoid hemorrhage on CT; to the OR emergently", "critical", "injury/medical"),
    ("Hunt-Hess IV subarachnoid haemorrhage on arrival", "critical", "injury/medical"),
    # Brain HERNIATION is the terminal endpoint of raised intracranial pressure (brainstem crush) — a
    # neuro sibling of the already-critical subarachnoid hemorrhage — yet the qualified clinical phrases
    # previously matched nothing and dropped to LOW. Each case isolates on the herniation phrase (no
    # other floored token), so removing the terms regresses to LOW. The benign disc/inguinal hernia
    # cases in test_bare_herniation_stays_low guard the deliberately-excluded bare "herniation"/"hernia".
    ("CT shows uncal herniation; neurosurgery notified", "critical", "injury/medical"),
    ("Signs of brain herniation on the repeat scan", "critical", "injury/medical"),
    ("Transtentorial herniation noted; pupils blown", "critical", "injury/medical"),
    ("Tonsillar herniation confirmed on imaging", "critical", "injury/medical"),
    ("Cerebral herniation imminent per the neuro exam", "critical", "injury/medical"),
    # "exsanguination"/"exsanguinated" is the clinical term for fatal blood loss — the fatal endpoint
    # of "severe bleeding" (already critical) — yet both the noun and the participle previously matched
    # nothing and dropped to LOW/MEDIUM. Both cases isolate on the new terms (no "death"/"severe
    # bleeding" token present), so removing them regresses to LOW/MEDIUM.
    ("The patient exsanguinated before EMS arrived on scene", "critical", "injury/medical"),
    ("Massive exsanguination from the laceration; responders en route", "critical", "injury/medical"),
    # The PRESENT participle "exsanguinating" must reach the same critical floor as the noun
    # "exsanguination"/past participle "exsanguinated" — an active trauma report writes "patient is
    # actively exsanguinating", which previously matched neither ("-ating" is a distinct token) and
    # dropped to LOW (same verb-form asymmetry class as amputated/decapitated). Case isolates on the new
    # term (no "death"/"severe bleeding"/"hemorrhage" token), so removing it regresses to LOW.
    ("Patient is actively exsanguinating from the femoral wound", "critical", "injury/medical"),
    # "massive/catastrophic/uncontrolled hemorrhage" (+ British "haemorrhage") are the clinical
    # QUALIFIED synonyms of the already-critical lay "severe bleeding" — the same immediately-fatal
    # exsanguinating bleed, previously scored only HIGH via the bare "hemorrhage" term. Each case
    # isolates on the new phrase (no "death"/"severe bleeding"/"exsanguination" token present), so
    # removing the phrases regresses to HIGH (via bare "hemorrhage"), proving the escalation.
    ("Trauma bay: massive hemorrhage, activated the transfusion protocol", "critical", "injury/medical"),
    ("Catastrophic haemorrhage from the femoral wound; responders en route", "critical", "injury/medical"),
    ("Uncontrolled hemorrhage on scene, could not stop it before EMS", "critical", "injury/medical"),
    # "hemorrhage"/"hemorrhaging" is the clinical synonym of "bleeding" (already HIGH) and must
    # reach the same HIGH floor — "worker is hemorrhaging" / a bare "hemorrhage" previously matched
    # nothing (neither "bleeding" nor "severe bleeding" is a substring) and dropped to LOW. NOTE the
    # BARE term stays HIGH; the QUALIFIED "massive/catastrophic/uncontrolled hemorrhage" phrases
    # escalate to critical above (their own cases), so these examples avoid those qualifiers.
    ("Worker is hemorrhaging badly after the press incident", "high", "injury/medical"),
    ("Hemorrhage reported on the floor; responders en route", "high", "injury/medical"),
    # The adjective "hemorrhagic" (== the HIGH nouns "hemorrhage"/"hemorrhaging") must reach the same
    # HIGH floor — "hemorrhagic rash" / "hemorrhagic stroke" previously matched nothing (\bhemorrhage\b
    # does not match "hemorrhagic") and dropped to LOW. Both cases isolate on the adjective (no
    # "bleeding"/"hemorrhage"/"stroke" HIGH-or-above token present; "stroke" is deliberately unlisted).
    # NOTE the bare adjective stays HIGH here — the QUALIFIED "hemorrhagic shock" phrase escalates to
    # critical above (its own case), so these examples deliberately avoid the word "shock".
    ("Patient developed a hemorrhagic rash; responders en route", "high", "injury/medical"),
    ("Suspected hemorrhagic stroke after the fall", "high", "injury/medical"),
    # British spelling "haemorrhagic" must reach the same HIGH floor purely on en-GB orthography;
    # isolates on the adjective (no other critical/high token present).
    ("The technician turned haemorrhagic before help arrived", "high", "injury/medical"),
    # The participle "concussed" must reach the same HIGH floor as the noun "concussion" — an acute
    # report is written "worker was concussed" / "concussed and disoriented", which previously
    # matched nothing (\bconcussion\b does not match "concussed") and dropped to LOW purely on
    # verb-vs-noun word form.
    ("Worker was concussed after the beam struck his helmet", "high", "injury/medical"),
    ("Employee concussed and disoriented; medics called to the scene", "high", "injury/medical"),
    # The participle "overdosed" must reach the same HIGH floor as the noun "overdose" — an acute
    # report is written "worker overdosed" / "he overdosed on the dock", which previously matched
    # nothing (\boverdose\b does not match "overdosed") and dropped to LOW purely on verb-vs-noun
    # word form (the same gap already fixed for amputation/amputated and concussion/concussed).
    ("Worker overdosed in the restroom; naloxone administered", "high", "injury/medical"),
    ("Contractor overdosed on the loading dock, medics en route", "high", "injury/medical"),
    # The PLURAL "broken bones" must reach the same HIGH floor as the singular "broken bone" — the
    # injury is usually reported "multiple broken bones" / "several broken bones", which previously
    # matched nothing (\bbroken bone\b does not match "broken bones") and dropped to LOW purely on
    # singular-vs-plural tokenization (the same gap already fixed for "burn"->"burns").
    ("Worker suffered multiple broken bones when the pallet load shifted", "high", "injury/medical"),
    ("Several broken bones reported after the ladder gave way", "high", "injury/medical"),
    # The participle "fractured" and the plural "fractures" must reach the same HIGH floor as the
    # noun "fracture" — a break is reported "fractured his leg" / "multiple fractures", which
    # previously matched nothing (\bfracture\b does not match "fractured"/"fractures") and dropped to
    # LOW purely on verb-vs-noun + singular-vs-plural word form (the same gap already fixed for
    # concussion/concussed, overdose/overdosed, and burn/burns).
    ("Worker fractured his wrist on the stamping press", "high", "injury/medical"),
    ("Two employees treated for multiple fractures after the fall", "high", "injury/medical"),
    # The PLURAL "injuries" must reach the same HIGH floor as the singular "injury"/"injured" — a
    # multi-casualty report is written "multiple injuries" / "several injuries reported", which
    # previously matched nothing (\binjury\b does not match "injuries") and dropped to LOW purely on
    # singular-vs-plural tokenization (the same gap already fixed for burn/burns and fracture/
    # fractures). Whole-word matching inside "head injuries" also covers the plural of "head injury".
    ("Multiple serious injuries at the site after the scaffold gave way", "high", "injury/medical"),
    ("Several injuries reported when the pallet load shifted", "high", "injury/medical"),
    ("Workers sustained head injuries from falling debris", "high", "injury/medical"),
    # The British/international spellings must reach the same HIGH floor as their US twins —
    # "hospitalised" (== "hospitalized") and "haemorrhage"/"haemorrhaging" (== "hemorrhage"/
    # "hemorrhaging") previously matched nothing and dropped to LOW purely on en-GB orthography.
    ("Worker hospitalised after the scaffold gave way", "high", "injury/medical"),
    ("Patient is haemorrhaging badly; responders en route", "high", "injury/medical"),
    ("Haemorrhage reported on the floor", "high", "injury/medical"),
    # "hypothermia"/"hypothermic" is an acute exposure emergency with no benign meaning and must
    # reach the injury/medical HIGH floor — previously matched nothing and dropped to LOW.
    ("Employee found with severe hypothermia after exposure", "high", "injury/medical"),
    ("Worker pulled from the walk-in freezer, hypothermic and shivering", "high", "injury/medical"),
    # "frostbite"/"frostbitten" is the cold-exposure sibling of hypothermia and must reach the same
    # injury/medical HIGH floor — both forms previously matched nothing and dropped to LOW.
    ("Worker found with severe frostbite on both hands", "high", "injury/medical"),
    ("The technician was frostbitten after the freezer lockout", "high", "injury/medical"),
    # "degloved"/"degloving" is a severe avulsion trauma with no benign meaning and must reach the
    # injury/medical HIGH floor — both forms previously matched nothing and dropped to LOW (the
    # generic word "injury" floored "degloving injury", but "was degloved" alone scored LOW). These
    # cases isolate on the term (no other critical/high token present).
    ("His hand was degloved in the roller mechanism", "high", "injury/medical"),
    ("The press caught his forearm, degloving it to the wrist", "high", "injury/medical"),
    # "scald"/"scalded" is the thermal-burn sibling of "burn"/"burns" and must reach the same
    # injury/medical HIGH floor — both forms previously matched nothing and dropped to LOW while
    # "burn"/"burns" scored HIGH. These cases isolate on the term (no "burn" token present).
    ("The operator was scalded by steam from the ruptured line", "high", "injury/medical"),
    ("A scald to the forearm from the hot water valve", "high", "injury/medical"),
    # "blood loss" is the plain-English phrase for the same emergency as "bleeding"/"hemorrhage" and
    # must reach the same injury/medical HIGH floor — previously matched nothing and dropped to LOW.
    # These cases isolate on the phrase (no "bleeding"/"hemorrhage" token present).
    ("The victim suffered severe blood loss after the guard rail failed", "high", "injury/medical"),
    ("Massive blood loss reported at the scene of the machinery incident", "high", "injury/medical"),
    # "heat stroke"/"heatstroke"/"hyperthermia"/"heat exhaustion" is the HOT counterpart to
    # hypothermia — the same acute exposure emergency, one commit apart — and must reach the same
    # injury/medical HIGH floor; previously matched nothing and dropped to LOW.
    ("Roofer collapsed with heat stroke during the afternoon shift", "high", "injury/medical"),
    ("Worker found hyperthermic and confused in the boiler room", "high", "injury/medical"),
    ("Two staff treated for heat exhaustion after the outage", "high", "injury/medical"),
    ("Structure fire in the warehouse, building ablaze", "critical", "fire/smoke"),
    # "flames" is the lay word for an active fire and must reach the same critical floor as "fire"
    # — "in flames" / "visible flames" (no literal "fire" token) previously dropped to LOW.
    ("The building is in flames on the east side", "critical", "fire/smoke"),
    ("Visible flames on the roof of the annex", "critical", "fire/smoke"),
    # The verb "exploded" must reach the same critical floor as the noun "explosion" — an acute
    # report is written "the transformer exploded" / "a boiler exploded", which previously matched
    # nothing (\bexplosion\b does not match "exploded") and dropped to LOW purely on verb-vs-noun
    # word form.
    ("The transformer exploded near the substation", "critical", "fire/smoke"),
    ("A boiler exploded in the basement mechanical room", "critical", "fire/smoke"),
    # The plural "explosions" and the present participle "exploding" must reach the same critical
    # floor as the noun "explosion" / past-tense "exploded" — "secondary explosions" and "a
    # transformer exploding" are how a multi-blast or unfolding event is written, and previously
    # matched nothing (\bexplosion\b/\bexploded\b match neither) and dropped to LOW purely on word form.
    ("Secondary explosions rocked the plant after the initial blast", "critical", "fire/smoke"),
    ("Multiple explosions reported in the storage yard", "critical", "fire/smoke"),
    ("A transformer exploding on the roof of the annex", "critical", "fire/smoke"),
    ("Lithium batteries exploding in the charging bay", "critical", "fire/smoke"),
    # The synonym "detonation" and its verb forms must reach the same critical floor as the
    # explosion family — "the device detonated" / "a detonation was heard" / "a car bomb detonated"
    # is how a blast is written, and previously matched nothing (\bexplosion\b/\bexploded\b match no
    # "deton-" form) and dropped to LOW purely on word choice. Every "deton-" form is exclusively an
    # explosion (zero benign meaning).
    ("The device detonated near the main entrance", "critical", "fire/smoke"),
    ("A car bomb detonated in the parking structure", "critical", "fire/smoke"),
    ("A detonation was heard on the third floor", "critical", "fire/smoke"),
    ("Crews report a charge detonating in the quarry pit", "critical", "fire/smoke"),
    # "arson"/"arsonist" (an intentionally-set fire named by its crime) must reach the same critical
    # floor as the plain word "fire" — a report that names the act rather than the flame previously
    # matched nothing and dropped to LOW purely on word choice. Neither case has an independent
    # critical/high token, so they isolate on the new term; "arson" denotes only deliberate
    # fire-setting (zero benign meaning), the same class as the molotov/detonation fixes.
    ("Suspected arson at the vacant warehouse overnight", "critical", "fire/smoke"),
    ("An arsonist set the dumpster alight behind the loading dock", "critical", "fire/smoke"),
    # "thermal runaway" — the self-sustaining exothermic battery/BESS/EV runaway that drives fire and
    # off-gas explosion — must reach the same critical floor as the plain word "fire": a report that
    # names the runaway rather than the flame previously matched nothing and dropped to LOW purely on
    # word choice. Neither case carries an independent critical/high token (no "fire"/"explosion"/
    # "battery-fire" word), so they isolate on the new term; "thermal runaway" denotes only the
    # hazardous uncontrolled exothermic event (zero benign meaning), the same class as arc-blast/
    # hydrogen-sulfide whole-hazard fixes.
    ("The ESS battery rack went into thermal runaway during the overnight charge", "critical", "fire/smoke"),
    ("Thermal runaway detected in the lithium-ion cells inside the parking-garage EV", "critical", "fire/smoke"),
    # "flashover" and "backdraft" — the two lethal fire-BEHAVIOR events — must reach the same critical
    # floor as the plain word "fire": a report that names the behavior rather than the flame previously
    # matched nothing and dropped to LOW purely on word choice, the same whole-hazard class as thermal
    # runaway/arc-blast. Neither case carries an independent critical/high token (no "fire"/"smoke"/
    # "explosion" word), so each isolates on the new term; both are whole words with zero benign meaning
    # (every real sense — fire behavior, insulator flashover arc, flue backdraft/CO spill — is a hazard).
    ("Crews reported a flashover in the second-floor compartment", "critical", "fire/smoke"),
    ("Conditions in the stairwell were deteriorating toward flashover", "critical", "fire/smoke"),
    ("A backdraft threw the nozzleman off the landing when the door was forced", "critical", "fire/smoke"),
    ("Command warned of backdraft potential in the sealed basement", "critical", "fire/smoke"),
    ("Smoke detected near the electrical panel", "high", "fire/smoke"),
    ("Server room flooded, equipment submerged", "critical", "water/flood"),
    ("Burst pipe caused water damage to the ceiling", "high", "water/flood"),
    # floodwater/floodwaters (an active inundation) is a distinct whole-word token that \bflood\b
    # does not match; the SAME singular/compound tokenization gap class as burn/burns and the weather
    # plurals. Both cases isolate on the new terms — no independent critical token fires.
    ("Floodwaters rose to the second floor of the plant", "critical", "water/flood"),
    ("Rising floodwater poured through the loading dock doors", "critical", "water/flood"),
    ("Exposed wiring sparking in the breaker box", "high", "electrical/power"),
    # "electric shock" must reach the same HIGH floor as "electrical shock" — the electric/electrical
    # word choice previously left the more common lay phrasing at LOW.
    ("Worker got an electric shock from the panel", "high", "electrical/power"),
    ("He received repeated electric shocks servicing the unit", "high", "electrical/power"),
    # The verb/participle forms of "electrocution" (critical) must reach the SAME critical floor —
    # the participle "electrocuted" previously caught only the injury/medical HIGH floor, and
    # "electrocute"/"electrocuting" matched nothing and dropped to LOW; the same lethal event scored
    # critical-or-lower purely on grammatical form. Each case isolates on the electrocution term.
    ("A lineman was electrocuted by the exposed bus bar", "critical", "electrical/power"),
    ("The energized panel could electrocute anyone who touches it", "critical", "electrical/power"),
    ("Two contractors were electrocuting themselves on the live rail", "critical", "electrical/power"),
    ("This fault electrocutes crews the instant the breaker recloses", "critical", "electrical/power"),
    # "arc blast" is the concussive pressure-wave sibling of the already-critical "arc flash" (the
    # thermal/radiant component) — a distinct electrical hazard an OSHA/incident report names in its
    # own right, but \barc flash\b does not match "arc blast", so a directly-named blast dropped to
    # LOW. Both cases isolate on the term (no other independent critical/high token) → without it they
    # drop below the electrical critical floor. Surfaced in the 2026-08-20 rule-probe (sibling of arc flash).
    ("The arc blast hurled the electrician across the switchgear room", "critical", "electrical/power"),
    ("Arc blast reported during the breaker replacement at the substation", "critical", "electrical/power"),
    # A "downed power line" is the fallen live-conductor hazard — the contact-electrocution twin of
    # "live wire" (always treat every downed line as energized), yet a directly-named report dropped to
    # LOW: the only existing token was the generic "downed line" over in WEATHER at HIGH, and \bdowned
    # line\b does not match "downed POWER line". Added at electrical CRITICAL beside "live wire" in all
    # four forms a report writes (two-word/one-word, singular/plural). Each case isolates on the new term
    # (no other independent critical/high token — "road"/"yard"/"perimeter" do not floor) → without it
    # they drop to LOW. Surfaced in the 2026-08-21 23:3x rule-probe (sibling of arc blast / live wire).
    ("A downed power line lay across the access road", "critical", "electrical/power"),
    ("Crews reported downed power lines behind the north yard", "critical", "electrical/power"),
    ("A downed powerline was found at the site perimeter", "critical", "electrical/power"),
    ("Utility crews flagged downed powerlines near the retention pond", "critical", "electrical/power"),
    ("Gas leak reported; carbon monoxide alarm triggered", "critical", "gas/chemical"),
    # "hydrogen sulfide" (H2S, the lethal rotten-egg / sour gas) is the sibling always-a-hazard
    # multi-word gas name of "carbon monoxide" but previously had ZERO gas/chemical coverage (no bare
    # entry, no "…leak" phrase), so a directly-named release dropped to LOW/HIGH-off-"leak". Both
    # cases isolate on the term (no other independent critical token) → without it they drop below
    # the gas/chemical critical floor. Surfaced in the 2026-08-20 rule-probe.
    ("Hydrogen sulfide detected in the sewer wet well", "critical", "gas/chemical"),
    ("Crews evacuated the pad after hydrogen sulfide filled the vault", "critical", "gas/chemical"),
    # "phosgene" (COCl2) and "cyanide" (HCN + salts) are the two remaining canonical lethal toxic gases
    # missing from the family — sibling always-a-hazard named gases of "carbon monoxide"/"hydrogen
    # sulfide", so a directly-named release must reach the same critical floor. Bare "cyanide" subsumes
    # "hydrogen cyanide"/"cyanide gas"/"cyanide poisoning" via \bcyanide\b. Each case isolates on the new
    # term (no other independent critical token). Surfaced in the 2026-08-21 toxic-gas rule-probe.
    ("Phosgene detected downwind of the isocyanate reactor", "critical", "gas/chemical"),
    ("Hydrogen cyanide filled the plating shop after the tank ruptured", "critical", "gas/chemical"),
    ("Cyanide gas release forced evacuation of the fumigation bay", "critical", "gas/chemical"),
    # "hydrofluoric acid" (aqueous HF) and "hydrogen fluoride" (anhydrous gas) are the same lethal
    # industrial toxicant in two lexically-distinct forms a report names — sibling always-a-hazard named
    # chemicals of carbon monoxide / hydrogen sulfide / phosgene / cyanide, so a directly-named exposure
    # must reach the same critical floor (a bare splash previously dropped to LOW; "…burn" only reached
    # HIGH off "burn"). Each case isolates on the new term (no other independent critical token — "line"/
    # "gas"/"vessel" do not floor). Surfaced in the 2026-08-21 toxic-gas rule-probe.
    ("The operator was splashed with hydrofluoric acid on the etch line", "critical", "gas/chemical"),
    ("Exposure to hydrogen fluoride in the sealed process vessel", "critical", "gas/chemical"),
    # "oxygen deficient atmosphere" / "oxygen-deficient atmosphere" is the canonical OSHA confined-space
    # killer (O2 below ~19.5%), the leading cause of confined-space fatalities, yet a directly-named
    # report previously matched nothing and dropped to LOW. Both spaced and hyphenated forms must reach
    # the gas/chemical critical floor. Each case isolates on the new term (no other independent critical
    # token — "tank"/"gas meter"/"manway" do not floor). Surfaced in the 2026-08-22 confined-space
    # O2-displacement rule-probe.
    ("Atmospheric test showed an oxygen deficient atmosphere before tank entry", "critical", "gas/chemical"),
    ("Entrant overcome by an oxygen-deficient atmosphere in the vault", "critical", "gas/chemical"),
    # "grain engulfment" is the NOUN form of a leading OSHA agricultural fatality — the confined-space
    # burial/suffocation death an ag/OSHA report names directly ("a grain engulfment at the feed mill").
    # The verb sibling "engulfed" already floors critical (via fire/smoke's "engulfed"), but the noun
    # phrasing matched nothing and dropped to LOW. Added as the full two-word phrase at the confined-space
    # fatal floor beside oxygen-deficient-atmosphere / asphyxiation — NOT bare "engulfment" (it has a
    # benign biology sense: macrophage engulfment of apoptotic cells). Each case isolates on the new term
    # (no other independent floored token — "feed mill"/"storage bin"/"unloading" do not floor). Surfaced
    # in the 2026-08-22 confined-space/agricultural rule-probe.
    ("Investigators documented a grain engulfment at the feed mill this morning", "critical", "gas/chemical"),
    ("Grain engulfment in the number three storage bin during unloading", "critical", "gas/chemical"),
    # Named chemical-warfare agents ("nerve agent"/"mustard gas") are always-a-hazard siblings of the
    # already-floored "phosgene" (itself a WWI CW agent), so a directly-named release must reach the same
    # gas/chemical critical floor — a directly-named agent previously matched nothing and dropped to LOW.
    # The plural "nerve agents" is a distinct token from "nerve agent" (\bnerve\s+agent\b won't match the
    # trailing "s"). Each case isolates on the new term (no other independent critical token — "detected"/
    # "ventilation"/"exposure"/"loading dock" do not floor). Surfaced in the 2026-08-22 chemical-weapon
    # rule-probe.
    ("A nerve agent was detected in the ventilation duct", "critical", "gas/chemical"),
    ("Multiple nerve agents suspected across the ventilation system", "critical", "gas/chemical"),
    ("Responders treating mustard gas exposure at the loading dock", "critical", "gas/chemical"),
    # A gas ODOR is a HIGH floor regardless of word order — the noun-compound and "natural gas"
    # phrasings a person actually writes must reach the same floor as "smell of gas"/"odor of gas".
    ("Strong gas smell reported in the mechanical room", "high", "gas/chemical"),
    ("Tenant reports a gas odor near the boiler", "high", "gas/chemical"),
    ("Natural gas smell throughout the east wing", "high", "gas/chemical"),
    ("Smell of natural gas by the loading dock", "high", "gas/chemical"),
    # Verb-order gas-odor reports must reach the same HIGH floor as the noun-order forms above —
    # "I smell gas" / "smells like gas" is how a person actually reports it, and previously dropped
    # to LOW purely on word order.
    ("Tenant called: I smell gas in the second-floor hallway", "high", "gas/chemical"),
    ("Staff report they smell natural gas near the meter", "high", "gas/chemical"),
    ("It smells like gas in the mechanical room", "high", "gas/chemical"),
    # The verb "asphyxiated" must reach the same critical floor as the noun "asphyxiation" — an acute
    # report is written "the worker was asphyxiated in the tank", which previously matched nothing
    # (\basphyxiation\b does not match "asphyxiated") and dropped to LOW purely on verb-vs-noun word
    # form (the same gap already fixed for explosion/exploded and amputation/amputated).
    ("The worker was asphyxiated in the storage tank", "critical", "gas/chemical"),
    ("Two crew asphyxiated in the confined space", "critical", "gas/chemical"),
    # The present participle "asphyxiating" must reach the same critical floor as the noun
    # "asphyxiation" / past-tense "asphyxiated" — "workers asphyxiating in the tank" is how an
    # unfolding exposure emergency is written, and previously matched nothing (\basphyxiation\b /
    # \basphyxiated\b match neither) and dropped to LOW purely on verb form (the exploded->exploding
    # / collapsed->collapsing present-participle gap class).
    ("Workers asphyxiating in the storage tank right now", "critical", "gas/chemical"),
    ("Crew asphyxiating in the confined space after the release", "critical", "gas/chemical"),
    # The lay noun "suffocation" is the plain-English synonym of the clinical "asphyxiation" (same
    # oxygen-deprivation death event) and must reach the same critical floor — "death by suffocation"
    # previously matched nothing and dropped to LOW purely on clinical-vs-lay word choice. No other
    # critical/high token appears in these cases, so they isolate on the new term.
    ("The confined-space entrant succumbed to suffocation", "critical", "gas/chemical"),
    ("Incident report cites suffocation as the cause", "critical", "gas/chemical"),
    # The bare clinical ROOT noun "asphyxia" (as in "traumatic asphyxia" / "positional asphyxia") must
    # reach the same critical floor as its derived forms asphyxiation/asphyxiated/asphyxiating —
    # \basphyxiation\b does not match the shorter "asphyxia", so it previously dropped to LOW purely on
    # word form. No other critical/high token appears in these cases, so they isolate on the new term.
    ("The medic recorded traumatic asphyxia at the scene", "critical", "gas/chemical"),
    ("Positional asphyxia suspected during the restraint", "critical", "gas/chemical"),
    ("Partial roof collapse; structural failure observed", "critical", "structural"),
    # The present participle "collapsing" must reach the same critical floor as the noun "collapse"
    # / past-tense "collapsed" — "the roof is collapsing" and "walls collapsing" are how an unfolding
    # structural emergency is written, and previously matched nothing (\bcollapse\b/\bcollapsed\b
    # match neither) and dropped to LOW purely on verb form (the exploded->exploding gap class).
    ("The roof is collapsing right now over the loading dock", "critical", "structural"),
    ("Walls collapsing in the east wing after the impact", "critical", "structural"),
    ("The mezzanine floor is actively collapsing", "critical", "structural"),
    ("Crack in the load-bearing wall is widening", "high", "structural"),
    # A "sinkhole" is an acute ground-failure emergency and must reach the same HIGH floor as its
    # slower cousin "subsidence" — "a sinkhole opened under the parking lot" previously matched
    # nothing and dropped to LOW. The plural "sinkholes" must also fire (\bsinkhole\b does not match
    # "sinkholes"), the same singular->plural tokenization gap already fixed for burn/burns.
    ("A sinkhole opened under the loading dock", "high", "structural"),
    ("Sinkhole swallowed part of the sidewalk near the entrance", "high", "structural"),
    ("Multiple sinkholes appeared across the lot overnight", "high", "structural"),
    # A trench/excavation "cave-in" is the excavation sibling of "sinkhole"/"subsidence" and must
    # reach the same HIGH ground-failure floor — the hyphenated noun previously matched nothing (it
    # contains no floored substring) and dropped to LOW. The plural "cave-ins" needs to fire too
    # (\bcave\-in\b does not match "cave-ins"), the same singular->plural gap as sinkhole/sinkholes.
    ("A cave-in buried part of the excavation on the north side", "high", "structural"),
    ("Reported a cave-in at the trench where the crew was digging", "high", "structural"),
    ("Multiple cave-ins along the pipeline trench after the rain", "high", "structural"),
    # A "rockslide" is the earth-movement sibling of "sinkhole"/"cave-in"/"subsidence" and must reach
    # the same HIGH ground-failure floor — the one-word noun previously matched nothing (it contains
    # no floored substring) and dropped to LOW. The plural "rockslides" must fire too (\brockslide\b
    # does not match "rockslides"), the same singular->plural gap as sinkhole/sinkholes.
    ("A rockslide buried the access road below the quarry face", "high", "structural"),
    ("Rockslide came down onto the rail line and blocked the tunnel mouth", "high", "structural"),
    ("Repeated rockslides closed the canyon highway overnight", "high", "structural"),
    ("Break-in overnight; forced entry through side door", "high", "security/intrusion"),
    ("Active shooter reported, armed individual on site", "critical", "security/intrusion"),
    ("Shots fired in the lobby; shooter fled the scene", "critical", "security/intrusion"),
    ("Gunshots heard in the parking garage", "critical", "security/intrusion"),
    ("Reports of gunfire near the loading dock", "critical", "security/intrusion"),
    ("A shooting occurred at the north entrance", "critical", "security/intrusion"),
    # "stab wound"/"stab wounds" is the bladed-weapon sibling of "gunshot wound" (critical via
    # "gunshot"): a directly-named violent penetrating trauma that previously matched nothing and
    # dropped to LOW. Multi-word phrase — no independent critical/high token in these cases, so they
    # isolate on the new term. Same weapon word-choice class as the firearm terms already here.
    ("The victim has a stab wound to the chest; assailant fled", "critical", "security/intrusion"),
    ("Knife attack in the break room; multiple stab wounds reported", "critical", "security/intrusion"),
    # "molotov"/"molotovs" (a thrown incendiary weapon) is the sibling of bomb-threat/gunshot/
    # stab-wound: a directly-named violent attack that previously matched nothing and dropped to LOW.
    # Neither case has an independent critical/high token (bare "molotov" covers "molotov cocktail"),
    # so they isolate on the new term. Same weapon-word class as the firearm/stab-wound terms here.
    ("A molotov cocktail was thrown through the front window", "critical", "security/intrusion"),
    ("Protesters hurled molotovs at the guard shack overnight", "critical", "security/intrusion"),
    # "pistol-whipped"/"pistol whipped"/"pistol-whipping"/"pistol whipping" (beating a victim with a
    # firearm) is the armed-assault sibling of gunshot/stab-wound/molotov: a directly-named armed
    # violent attack that previously matched nothing and dropped to LOW. Reports write it hyphenated
    # and spaced, as past participle and gerund, so each is a distinct token needing its own entry.
    # No independent critical/high token in these cases, so they isolate on the new terms; the
    # compound has no benign polysemy (bare "whipped" is not added).
    ("The night-shift guard was pistol-whipped near the turnstile", "critical", "security/intrusion"),
    ("Suspect pistol whipped the attendant and fled the booth", "critical", "security/intrusion"),
    # The knife-assault EVENT terms — "knife attack", "stabbing attack"/"stabbing spree"/"mass
    # stabbing", and the bladed sibling of "gunpoint" ("knifepoint") — are directly-named armed
    # attacks that previously matched nothing and dropped to LOW, while the firearm-event family
    # (gunshot / shots fired / active shooting) already floors critical. Multi-word phrases or the
    # unambiguous single word "knifepoint" (no benign meaning); each case isolates on the new term
    # (no other critical/high token present), and none touch the bare "stab"/"stabbing" FP guard.
    ("A stabbing attack unfolded in the third-floor cafeteria", "critical", "security/intrusion"),
    ("A mass stabbing at the food court left several people hurt", "critical", "security/intrusion"),
    ("Reports of a stabbing spree along the east corridor", "critical", "security/intrusion"),
    ("Knife attack in the main lobby; the assailant fled", "critical", "security/intrusion"),
    ("A visitor was held at knifepoint near the loading dock", "critical", "security/intrusion"),
    # The explosive-DEVICE terms — "pipe bomb"/"pipe bombs"/"car bomb"/"car bombs" — are the found/
    # planted-device sibling of "bomb threat" (security critical): \bbomb\s+threat\b matches only the
    # THREAT, so a directly-named device previously matched nothing and dropped to LOW, the same
    # threat-vs-device split as stab-wound (injury) vs stabbing-attack (event). Multi-word phrases with
    # no benign polysemy (bare "bomb" is NOT added — bath bomb / photobomb / "the movie bombed"); the
    # plurals are distinct tokens needing their own entries. Each case uses a neutral verb (found/
    # discovered/reported) so it isolates on the new term with no explosion token present.
    ("A pipe bomb was found in the mailroom", "critical", "security/intrusion"),
    ("Two pipe bombs were left at the north gate", "critical", "security/intrusion"),
    ("A car bomb was discovered under the delivery van", "critical", "security/intrusion"),
    ("Multiple car bombs were reported outside the annex", "critical", "security/intrusion"),
    # "kidnapped"/"kidnappings" are the verb and plural of "kidnapping" (security critical): the same
    # violent crime that previously matched nothing and dropped to LOW purely on word form. Distinct
    # tokens needing their own entries (\bkidnapping\b matches neither), same verb/plural class as
    # carjacking/carjacked/carjackings. Whole words with no benign polysemy.
    ("A worker was kidnapped from the loading dock overnight", "critical", "security/intrusion"),
    ("Two kidnappings reported near the visitor lot this month", "critical", "security/intrusion"),
    # "gunpoint" is the firearm-coercion sibling of "knifepoint" (critical) — the knifepoint comment
    # already names it as its sibling — but the word lived ONLY inside theft's "robbery at gunpoint",
    # so a gunpoint incident with no "robbery" token previously matched nothing and dropped to LOW.
    # Whole word, no benign meaning; these cases carry no "robbery"/"armed" token so they isolate on
    # the new security-critical entry (not on theft's "robbery at gunpoint").
    ("Staff were held at gunpoint while the vault was opened", "critical", "security/intrusion"),
    ("The suspect fled at gunpoint after ordering everyone down", "critical", "security/intrusion"),
    # "grenade"/"grenades" (a thrown explosive weapon) is the device sibling of pipe bomb / molotov /
    # car bomb (all critical): a directly-named explosive attack that previously matched nothing and
    # dropped to LOW. Whole words, no benign operational polysemy; the plural is a distinct token.
    # Neutral verb (thrown/recovered) so each isolates on the new term with no explosion token.
    ("A grenade was thrown into the ground-floor lobby", "critical", "security/intrusion"),
    ("Two grenades were recovered from the abandoned vehicle", "critical", "security/intrusion"),
    # "IED"/"IEDs" is the standard acronym for "improvised explosive device" (critical): a responder
    # writes the acronym far more often than the spelled-out phrase, yet it previously matched nothing
    # and dropped to LOW while the full phrase floored critical — the acronym-vs-phrase miss class of
    # CPR. Word-boundary-guarded (does not fire inside "studied"/"tied"); plural is a distinct token.
    ("An IED was discovered in the mailroom", "critical", "security/intrusion"),
    ("Two IEDs were found along the perimeter fence", "critical", "security/intrusion"),
    # "sniper"/"snipers" is the concealed-shooter sibling of "shooter"/"active shooter"/"gunman"
    # (critical): a directly-named active lethal threat that previously matched nothing and dropped to
    # LOW. Whole words, no benign meaning in incident text (the metaphor is the gerund "sniping", not
    # added); each case isolates on the new term with no other critical/high token, plural distinct.
    ("A sniper was reported on the parking-garage roof", "critical", "security/intrusion"),
    ("Two snipers positioned near the north gate, per the guard", "critical", "security/intrusion"),
    # "was stabbed"/"stabbed to death"/"fatally stabbed" are the passive/fatal report forms of a
    # stabbing (critical via "stab wound"/"stabbing attack") that previously matched nothing and
    # dropped to LOW. Same word-form class as amputated/decapitated. Qualified phrases, NOT bare
    # "stabbed" (that stays low — see the "stabbed at his lunch" FP guard). Each case isolates on the
    # new term (no other critical/high token: "parking lot"/"loading dock" carry none).
    ("A worker was stabbed in the parking lot", "critical", "security/intrusion"),
    ("The night attendant was stabbed near the loading dock and fled", "critical", "security/intrusion"),
    ("A visitor was stabbed to death outside the east entrance", "critical", "security/intrusion"),
    # The fatal-SHOOTING victim-outcome phrasings are the firearm twins of the stabbing forms just
    # above — "fatally shot"/"shot to death"/"shot and killed" — the way a fatal shooting is written
    # up, yet only the stabbing side floored critical; the firearm EVENT/OFFENDER family
    # (gunshot/shooter/shots fired) has NO token inside "the victim was fatally shot", so it dropped
    # to LOW. "opened fire" is the twin of "shots fired": it previously floored only off the
    # coincidental "fire" token (wrong category — fire/smoke), so its own security entry fixes the
    # RATIONALE. Qualified multi-word phrases, NOT bare "shot" (see the FP guard below). Isolation:
    # "fatally shot"/"shot and killed" carry no other floored token; "shot to death" also fires
    # injury/medical via "death" and "opened fire" also fires fire/smoke via "fire", so for those two
    # the security/intrusion category assertion (only the new entry supplies it) guards the addition.
    ("The victim was fatally shot in the parking lot", "critical", "security/intrusion"),
    ("A man was shot to death outside the loading dock", "critical", "security/intrusion"),
    ("A worker was shot and killed near the north gate and the suspect fled", "critical", "security/intrusion"),
    ("The suspect opened fire in the third-floor lobby", "critical", "security/intrusion"),
    # "acid attack"/"acid attacks" (a corrosive-substance assault) is the violent-assault sibling of
    # "knife attack"/"stabbing attack" (security critical): a directly-named attack that previously
    # matched nothing and dropped to LOW. Only the qualified two-word phrase is added — bare "acid"
    # (acid wash / acid rain / lactic acid / "acid test") stays LOW (see the FP guard). The plural is a
    # distinct token needing its own entry. Each case isolates on the new term (no other floored token).
    ("An acid attack on the technician at the loading dock", "critical", "security/intrusion"),
    ("The victim of an acid attack was treated at the gate", "critical", "security/intrusion"),
    ("Two acid attacks reported near the north entrance this month", "critical", "security/intrusion"),
    # "firebomb"/"firebombs"/"firebombed"/"firebombing" (an incendiary weapon/attack) is the direct
    # synonym of "molotov" (security critical): a named incendiary assault that previously matched
    # nothing and dropped to LOW — whole-word \bfire\b does NOT fire inside "firebomb", so it was a
    # true LOW, not a mis-attributed fire/smoke hit. The verb, plural, and gerund are distinct tokens
    # needing their own entries (\bfirebomb\b matches none of them), the same tokenization class as
    # molotov/molotovs. Whole words with no benign polysemy; each case isolates on the new term.
    ("The office was firebombed overnight and the suspect fled", "critical", "security/intrusion"),
    ("A firebomb was thrown through the front window", "critical", "security/intrusion"),
    ("Firebombs were hurled at the guard shack", "critical", "security/intrusion"),
    ("A firebombing at the north depot early this morning", "critical", "security/intrusion"),
    ("Theft of equipment; inventory stolen from the dock", "high", "theft"),
    # "carjacking"/"carjacked"/"carjackings" (taking a vehicle by force) is the violent-theft sibling
    # of "armed robbery" (theft critical): a directly-named violent robbery that previously matched
    # nothing and dropped to LOW. The verb "carjacked" and plural "carjackings" are distinct tokens
    # needing their own entries (\bcarjacking\b matches neither), the same verb/plural tokenization
    # class already fixed for molotov/molotovs and burn/burns. Whole words with no benign polysemy.
    ("A carjacking occurred at the north entrance overnight", "critical", "theft"),
    ("Employee was carjacked at gunpoint in the parking garage", "critical", "theft"),
    ("Two carjackings reported in the visitor lot this month", "critical", "theft"),
    ("Site-wide outage; all systems down", "critical", "outage"),
    ("Power outage; the server is down", "high", "outage"),
    ("Tornado warning; high winds and a fallen tree", "critical", "weather"),
    # Plural weather catastrophes must reach the same CRITICAL floor as their singular — the
    # whole-word matcher scores "tornado"/"hurricane"/"earthquake" critical but the plural spelling
    # ("tornadoes"/"tornados"/"hurricanes"/"earthquakes") is a distinct token that previously matched
    # nothing and dropped to LOW. Same singular→plural tokenization class already fixed for
    # burns/injuries/fractures/explosions. Plurals are whole words with no benign polysemy.
    ("Multiple tornadoes touched down near the plant", "critical", "weather"),
    ("Two tornados reported across the county", "critical", "weather"),
    ("Several hurricanes are forecast to make landfall this week", "critical", "weather"),
    ("A series of earthquakes rattled the region overnight", "critical", "weather"),
    # "typhoon"/"typhoons" (the Northwest-Pacific regional name for a hurricane) must reach the same
    # CRITICAL floor as "hurricane" — the identical event scored critical-or-LOW purely on which
    # regional word the reporter used, the same class as the en-GB spelling and lay-synonym fixes.
    # Whole words with no benign polysemy; plural is a distinct token needing its own entry.
    ("A typhoon is forecast to make landfall near the coastal plant", "critical", "weather"),
    ("Two typhoons battered the offshore facility this season", "critical", "weather"),
    # "tropical cyclone"/"tropical cyclones" is the formal WMO/NWS umbrella for the same storm regionally
    # named "hurricane"/"typhoon" (both already CRITICAL) — Cyclone Idai/Nargis are directly-named
    # tropical-cyclone catastrophes, yet the phrase previously matched nothing and dropped to LOW. The
    # QUALIFIER "tropical" disambiguates the polysemous bare "cyclone" (fence/separator, excluded), so the
    # qualified multi-word phrase floors CRITICAL with zero collision. The plural is a distinct token
    # (\btropical\s+cyclone\b won't match "tropical cyclones"), needing its own entry. Neither sentence
    # carries any other floored token (no bare "cyclone"/"storm"), so removing the entries regresses each
    # case to LOW and fails the CRITICAL assertion (isolation).
    ("A tropical cyclone is bearing down on the coastal facility", "critical", "weather"),
    ("Two tropical cyclones intensified over the gulf this week", "critical", "weather"),
    # The plural spellings of the remaining weather-critical singulars must reach the same CRITICAL
    # floor — "wildfire"/"flash flood"/"tsunami" scored critical but "wildfires"/"flash floods"/
    # "tsunamis" are distinct tokens that previously matched nothing and dropped to LOW, the same
    # singular→plural gap already closed for tornado(es)/hurricane(s)/typhoon(s)/earthquake(s).
    ("Three wildfires are threatening the north perimeter", "critical", "weather"),
    ("Flash floods reported across the county overnight", "critical", "weather"),
    ("Tsunamis following the offshore quake are inbound", "critical", "weather"),
    # The volcanic members of the natural-catastrophe family must reach the same CRITICAL floor as
    # earthquake/tsunami — "volcanic eruption" (the event), "pyroclastic flow" (its lethal mechanism)
    # and "lava flow" (the advancing molten hazard) previously matched nothing and dropped to LOW, the
    # same whole-catastrophe absent-term gap as typhoon beside hurricane. Each is a MULTI-WORD phrase
    # with zero benign meaning; each case isolates on the new phrase (no other floored token), so
    # removing it regresses the case to LOW.
    ("A volcanic eruption forced evacuation of the coastal site", "critical", "weather"),
    ("A pyroclastic flow swept down toward the perimeter fence", "critical", "weather"),
    ("The lava flow advanced across the access road overnight", "critical", "weather"),
    # "storm surge" is the lethal mechanism of a hurricane — the deadliest coastal hazard — and must
    # reach the same CRITICAL floor as the hurricane it accompanies. It previously scored only HIGH:
    # the bare word "storm" floors at weather HIGH, so the phrase matched "storm" and under-floored one
    # level. The only floored tokens here are the new "storm surge" (CRITICAL) and "storm" (HIGH), so
    # removing the phrase regresses this case to HIGH and fails the CRITICAL assertion (isolation).
    ("A storm surge is overtopping the seawall at the coastal site", "critical", "weather"),
    # "derecho"/"derechos" name a widespread straight-line windstorm (hurricane-force gusts over a
    # 240+ mile swath) — a directly-named weather catastrophe on the same footing as tornado/hurricane/
    # typhoon, yet it previously matched nothing and dropped to LOW. The irregular loanword plural means
    # \bderecho\b does not match "derechos", so each is a distinct entry. Neither sentence carries any
    # other floored token (no "storm"/"winds"), so removing the entries regresses each case to LOW and
    # fails the CRITICAL assertion (isolation).
    ("A derecho is forecast to reach the plant this evening", "critical", "weather"),
    ("Two derechos battered the region earlier this summer", "critical", "weather"),
    # "lahar"/"lahars" name a volcanic mudflow/debris flow (the 1985 Nevado del Ruiz lahar killed
    # ~23,000) — a directly-named volcanic catastrophe on the same footing as the already-critical
    # "volcanic eruption"/"pyroclastic flow"/"lava flow", yet it previously matched nothing and dropped
    # to LOW. The loanword plural means \blahar\b does not match "lahars", so each is a distinct entry.
    # Neither sentence carries any other floored token ("volcano"/"valley" are not signals), so removing
    # the entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A lahar is descending the volcano toward the plant", "critical", "weather"),
    ("Two lahars swept through the valley last week", "critical", "weather"),
    # "temblor"/"temblors" is the English-adopted loanword synonym for "earthquake" (Merriam-Webster:
    # "temblor: earthquake") — a report is as likely to write "temblor" as "earthquake", yet it
    # previously matched nothing and dropped to LOW while its exact sibling "earthquake" floors CRITICAL.
    # The loanword plural means \btemblor\b does not match "temblors", so each is a distinct entry.
    # Neither sentence carries any other floored token (no "earthquake"/"quake"/"storm"), so removing the
    # entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A strong temblor struck the facility overnight", "critical", "weather"),
    ("Two temblors rattled the plant this week", "critical", "weather"),
    # "megaquake"/"megaquakes" name a great earthquake (magnitude ~8+) — a directly-named seismic
    # catastrophe (Cascadia/Nankai megaquake) strictly worse than the already-critical "earthquake"/
    # "temblor", yet it previously matched nothing and dropped to LOW. The compound plural means
    # \bmegaquake\b does not match "megaquakes", so each is a distinct entry. Neither sentence carries any
    # other floored token (no bare "earthquake"/"quake"/"storm"), so removing the entries regresses each
    # case to LOW and fails the CRITICAL assertion (isolation).
    ("A megaquake could level the coastal plant", "critical", "weather"),
    ("Two megaquakes struck the fault zone this decade", "critical", "weather"),
    # "megatsunami"/"megatsunamis" name a great tsunami (Lituya Bay 1958, the Cumbre Vieja flank-collapse
    # scenario) — the tsunami analogue of the already-critical "megaquake", strictly worse than the
    # already-critical "tsunami", yet it previously matched nothing and dropped to LOW because \btsunami\b
    # does not match inside the compound "megatsunami". The compound plural likewise means \bmegatsunami\b
    # does not match "megatsunamis", so each is a distinct entry. Neither sentence carries any other floored
    # token (no bare "tsunami"/"wave"/"storm"), so removing the entries regresses each case to LOW and fails
    # the CRITICAL assertion (isolation).
    ("A megatsunami could inundate the coastal plant", "critical", "weather"),
    ("Two megatsunamis followed the offshore flank collapse", "critical", "weather"),
    # "superstorm"/"superstorms" name an exceptionally destructive storm system (Superstorm Sandy 2012, the
    # 1993 "Storm of the Century") — a directly-named storm catastrophe strictly worse than the ordinary
    # "storm" (which floors only HIGH), yet it previously matched nothing and dropped to LOW because
    # \bstorm\b does not match inside the compound "superstorm". The compound plural likewise means
    # \bsuperstorm\b does not match "superstorms", so each is a distinct entry. Neither sentence carries any
    # other floored token (no bare "storm"/"flood"/"hurricane"), so removing the entries regresses each case
    # to LOW and fails the CRITICAL assertion (isolation).
    ("A superstorm was bearing down on the coastal plant", "critical", "weather"),
    ("Two superstorms tracked up the seaboard this season", "critical", "weather"),
    # The plural "hostages" must reach the same CRITICAL floor as the singular "hostage" — an active
    # abduction crisis is usually reported in the plural ("took hostages"), a distinct token that
    # \bhostage\b does not match. No other critical/high token is present in this sentence (the only
    # floored word is the new "hostages"), so removing the entry regresses this case to LOW and fails
    # the CRITICAL assertion (isolation). Kept "gunman"-free (the suspect) so it isolates on "hostages".
    ("The suspect took hostages inside the control room", "critical", "security/intrusion"),
    ("Multiple hostages are being held in the vault", "critical", "security/intrusion"),
    # "gunman"/"gunmen" name the armed offender exactly as a report writes it — the sibling of the
    # already-critical "shooter"/"active shooter". The irregular plural means \bgunman\b does not match
    # "gunmen", so each is a distinct entry. Neither sentence carries any other floored token (a bare
    # "gunman opened fire" line would falsely floor off "fire", so these avoid it), so removing the
    # entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A lone gunman barricaded himself on the third floor", "critical", "security/intrusion"),
    ("Two gunmen entered the facility through the loading dock", "critical", "security/intrusion"),
    # Verb-order lightning reports must reach the same HIGH floor as the noun "lightning strike" —
    # "lightning struck the ..." / "struck by lightning" is how a person actually reports it, and
    # previously matched nothing and dropped to LOW purely on verb-vs-noun word order.
    ("Lightning struck the rooftop antenna array", "high", "weather"),
    ("A worker was struck by lightning in the north lot", "high", "weather"),
]


def test_each_category_hits_expected_floor():
    for text, floor, category in CASES:
        sev, reasons = risk.rule_layer(text)
        assert _atleast(sev, floor), f"{text!r} -> {sev}, expected >= {floor}"
        joined = " ".join(reasons)
        assert category in joined, f"{text!r} missing category {category} in {joined!r}"


def test_rationale_lists_matched_terms():
    sev, reasons = risk.rule_layer("Gas leak with toxic fumes near the boiler")
    assert sev == "critical"
    assert any("matched:" in r for r in reasons)


# Benign phrases that embed a taxonomy keyword inside a larger word. Substring matching wrongly
# scored these CRITICAL ("armed" in "unarmed", "fire" in "firearm"); whole-word matching must not.
NO_FALSE_POSITIVE = [
    ("Unarmed guard completed a routine patrol; all clear.", "armed"),
    ("Employee cleaned the firearm display case in the lobby.", "fire"),
    # The firearm/shooting terms added to security/intrusion must not fire from inside benign
    # words: "shooting" in "troubleshooting" (extremely common in a facilities/IT incident log),
    # "shooter" in "troubleshooter"/"sharpshooter". Word boundaries (\b) must hold the line.
    ("Troubleshooting the printer took most of the morning.", "shooting"),
    ("The troubleshooter reset the panel; nothing else to note.", "shooter"),
    # The new gas-odor phrasings must stay whole-word: a benign mention of the word "gas" with no
    # odor/leak (a filled gas tank, a gas station errand) must NOT fire the gas/chemical floor.
    ("Refueled the generator; the gas tank is now full.", "gas smell"),
    ("Drove to the gas station for supplies; nothing to report.", "gas odor"),
    # The verb-order forms are adjacent-word phrases, so a bare "gas" errand with no smell report
    # (gas prices, a gas can) must NOT fire — "smell" and "gas" are not adjacent here.
    ("Gas prices went up again; refilled the gas can.", "smell gas"),
    # The new acute-medical terms must stay whole-word: a figurative "heart" mention (heartfelt
    # thanks, "at the heart of the issue") must NOT fire the injury/medical critical floor.
    ("A heartfelt thank-you note was left; nothing else to report.", "heart attack"),
    # "flames" is whole-word: the singular "flame" inside "flame-retardant" (and "inflames") must
    # NOT fire the fire/smoke critical floor — only the plural incident-word "flames" does.
    ("Inspected the flame-retardant coating on the ducts; nothing to report.", "flames"),
    # The respiratory-distress phrases are multi-word adjacency phrases: a benign lone "breath"
    # or "short of ..." with no distress must NOT fire the injury/medical HIGH floor.
    ("Team took a breath before the next task; short of staff this week.", "trouble breathing"),
    # The unconsciousness synonyms are multi-word adjacency phrases / the whole-word verb "fainted":
    # a benign "passed out the agenda", an "unresponsive" server, or the adjective "faint" must NOT
    # fire the injury/medical floor (those polysemous forms were deliberately left out).
    ("She passed out the meeting agenda; the server was unresponsive so we rebooted it.",
     "loss of consciousness"),
    ("A faint smell of coffee lingered; the wifi signal was faint in the back office.", "fainted"),
    # "CPR" is whole-word: a benign token that merely embeds the letters (a part number, a code
    # like "CPRX-100") must NOT fire the injury/medical critical floor — only the standalone acronym.
    ("Ordered replacement part CPRX-100 for the HVAC unit; nothing else to report.", "cpr"),
    # "electric shock" is a multi-word adjacency phrase, so the bare polysemous noun "shock" must NOT
    # fire the electrical/power floor: culture shock, a shock absorber, "the news was a shock".
    ("New hires felt some culture shock; the shock absorber was replaced and the news was a shock.",
     "electric shock"),
    # The cardiac-arrest phrasings are multi-word adjacency ("no pulse"/"no heartbeat") or the whole
    # clinical word "pulseless": the bare polysemous noun "pulse" must NOT fire the injury/medical
    # floor — a pulse oximeter reading, "the pulse of the organization", an electrical pulse.
    ("Checked the pulse oximeter; the pulse of the team is good and we logged a clean signal pulse.",
     "no pulse"),
    # "stab wound(s)" is a multi-word adjacency phrase, so the bare polysemous "stab"/"stabbing"/
    # "stabbed" must NOT fire the security/intrusion floor: a "stabbing pain" (benign medical),
    # "took a stab at it" / "stabbed at the food" (idiom). Only the whole phrase "stab wound" does.
    ("He reported a stabbing pain in his side; took a stab at fixing it and stabbed at his lunch.",
     "stab wound"),
    # The fatal-shooting phrases are qualified multi-word adjacency, so the bare polysemous "shot" must
    # NOT fire the security floor: "gave it a shot", a photo "shot on location", a "flu shot". Only the
    # whole fatal phrases ("fatally shot"/"shot to death"/"shot and killed") do.
    ("She gave it a shot at the interview; the photo was shot on location and he booked his flu shot.",
     "fatally shot"),
    # The excluded neighbors of the fatal-shooting phrases must NOT fire: "shot dead" was left out for
    # the benign "shot dead center" (archery/aim), and "gunned down" for the driving sense "gunned down
    # the highway" — proving only the zero-collision phrases were added, not their over-firing kin.
    ("The arrow landed shot dead center on the range as they gunned down the highway to the meeting.",
     "shot dead"),
    # "pistol-whipped"/"pistol whipping" is a two-word/hyphenated compound, so the bare "whipped"/
    # "whipping" must NOT fire the security/intrusion floor: whipped cream, "whipped the team into
    # shape", a whipping wind. Only the whole "pistol" compound does.
    ("The chef whipped the cream while the whipping wind whipped the team into finishing early.",
     "pistol-whipped"),
    # The apnea phrasings are multi-word adjacency phrases: a benign lone "breathing"/"breath" with
    # no apnea (a breather break, "breathing room" in the schedule) must NOT fire the critical floor.
    ("Team took a breather; there's finally some breathing room in the schedule.", "stopped breathing"),
    # The polysemous vascular-event synonyms were DELIBERATELY left out when "aneurysm"/"embolism"
    # were added: "stroke" (a swim stroke / stroke of luck) and "seizure" (asset seizure) carry
    # benign meanings and must NOT fire the injury/medical critical floor — proving we added only the
    # zero-collision clinical words, not their over-firing neighbors.
    ("Swimmer logged a strong stroke; the asset seizure paperwork was filed with legal.", "stroke"),
    # The lightning verb-forms are multi-word adjacency phrases: a bare "struck"/"strike" with no
    # "lightning" adjacent (a labor strike, striking a deal, struck out) must NOT fire the weather
    # floor — only the whole phrase does.
    ("The workers struck a deal; the union may strike next week and he struck out at the plate.",
     "lightning struck"),
    # The heat-emergency terms are multi-word adjacency ("heat stroke"/"heat exhaustion") or whole
    # clinical words ("heatstroke"/"hyperthermia"): a bare "heat" mention with no medical event — a
    # heat wave, a heat exchanger, "turn up the heat" — must NOT fire the injury/medical floor.
    ("A heat wave rolled through; the heat exchanger was serviced and we turned up the heat.",
     "heat stroke"),
    # Only the acute participle "amputated" was added, NOT the chronic descriptor "amputee": a
    # benign "amputee support group" / "amputee parking" mention is not an acute emergency and must
    # NOT fire the injury/medical critical floor (\bamputated\b does not match "amputee").
    ("Posted a flyer for the amputee support group; reserved an amputee parking spot.", "amputated"),
    # "hemorrhage"/"hemorrhaging" are whole words: the benign prefix-sharer "hemorrhoid"/
    # "hemorrhoids" — a non-emergency condition — must NOT fire the injury/medical floor
    # (\bhemorrhage\b / \bhemorrhaging\b do not match "hemorrhoid").
    ("Employee asked about hemorrhoid treatment and hemorrhoids relief, then took a break.",
     "hemorrhage"),
    # The en-GB spelling "haemorrhage"/"haemorrhaging" is likewise a whole word: the benign British
    # prefix-sharer "haemorrhoid"/"haemorrhoids" must NOT fire the injury/medical floor
    # (\bhaemorrhage\b / \bhaemorrhaging\b do not match "haemorrhoid").
    ("Employee asked about haemorrhoid treatment and haemorrhoids relief, then took a break.",
     "haemorrhage"),
    # "broken bones" is a multi-word adjacency phrase, deliberately NOT the bare polysemous "broken":
    # a broken printer / broken window is routine maintenance, not an injury, and must NOT fire the
    # injury/medical floor (only the "broken bone(s)" phrasing does).
    ("The broken printer and a broken window were logged in the maintenance queue.",
     "broken bones"),
    # "sinkhole" is a whole word: the common noun "sink" (a kitchen sink) and the verb "sink" (to
    # sink a budget) must NOT fire the structural floor — \bsinkhole\b does not match "sink".
    ("The kitchen sink is clogged and we may sink the extra budget into repairs.", "sinkhole"),
    # Only the HYPHENATED noun "cave-in" was added, NOT the spaced negotiation idiom "cave in":
    # "cave in to demands/pressure" is a common figurative usage and must NOT fire the structural
    # HIGH floor — \bcave\-in\b requires a literal hyphen, so the spaced idiom cannot match it.
    ("Management chose to cave in to the union's demands during the meeting.", "cave-in"),
    # Only the one-word "rockslide"/"rockslides" was added, NOT the polysemous earth-disaster siblings
    # "landslide" (a "landslide victory"), "mudslide" (the cocktail), or "avalanche" (an "avalanche of
    # tickets"): those common figurative usages must NOT fire the structural HIGH floor — none of them
    # is a match for \brockslide\b, proving we added only the zero-collision term, not its neighbors.
    ("The mayor won in a landslide, the bar served a mudslide, and support saw an avalanche of tickets.",
     "rockslide"),
    # Only the literal noun "suffocation" was added, NOT the metaphor-heavy verb/adjective forms:
    # "suffocated by the workload", "suffocating heat", and "the suffocating bureaucracy" are common
    # figurative usages and must NOT fire the gas/chemical critical floor (\bsuffocation\b matches
    # none of them) — proving we added only the zero-collision noun, not its over-firing neighbors.
    ("Staff felt suffocated by the workload amid the suffocating heat and suffocating bureaucracy.",
     "suffocation"),
    # Only the literal noun "strangulation" was added, NOT the metaphor-heavy participle/gerund forms:
    # "a strangled cry", "the merger strangled competition", and "strangling the budget" are common
    # figurative usages and must NOT fire the injury/medical critical floor (\bstrangulation\b matches
    # none of them) — the exact tolerance boundary already drawn for suffocation vs suffocated.
    ("A strangled cry was heard as the merger strangled competition and kept strangling the budget.",
     "strangulation"),
    # Only the lethal qualified phrase "tension pneumothorax" was added, NOT the bare "pneumothorax":
    # a small spontaneous/simple pneumothorax can be stable and merely monitored, so the bare token must
    # NOT fire the injury/medical critical floor (\btension pneumothorax\b does not match it) — the exact
    # bare-vs-qualified boundary already drawn for tamponade (excluded) vs cardiac/pericardial tamponade.
    ("A small spontaneous pneumothorax was monitored overnight and resolved on its own.",
     "tension pneumothorax"),
    # Only the unambiguous tropical-cyclone terms "typhoon" and the QUALIFIED "tropical cyclone" were
    # added, NOT the polysemous bare "cyclone": a "cyclone fence" (chain-link fencing) and a "cyclone
    # separator" (industrial dust collector) are routine facilities/equipment terms and must NOT fire the
    # weather critical floor — \btropical\s+cyclone\b cannot match either, proving the qualified phrase
    # closes the synonym gap without reopening its over-firing bare root.
    ("The cyclone fence along the perimeter and the cyclone separator on line 3 both need service.",
     "cyclone"),
    # "arson" is whole-word: the benign noun "parson" (a clergyman) embeds the letters a-r-s-o-n but
    # has no word boundary before them, so \barson\b must NOT fire the fire/smoke critical floor —
    # the exact armed/unarmed, fire/firearm guard applied to the new intentional-fire term.
    ("The parson led the memorial service in the chapel; nothing else to report.", "arson"),
    # The whole crime-word "carjacking"/"carjacked" was added, NOT the bare token "car": a company
    # car, a car park, and a parked car are routine facilities mentions and must NOT fire the theft
    # critical floor — proving we added only the zero-collision crime word, not its over-firing root.
    ("The company car was serviced and the car park resurfaced; a parked car was moved.", "carjacking"),
    # Only the unambiguous crime word "kidnapped" was added, NOT the polysemous legal synonym
    # "abduction"/"abducted": in an injury/rehab context "abduction" is the anatomical range-of-motion
    # term ("limited shoulder abduction", "hip abduction exercises") a PT/ergonomics note routinely
    # uses, so it must NOT fire the security critical floor — proving we added only the zero-collision
    # crime word, not its over-firing medical homograph.
    ("PT noted limited shoulder abduction; hip abduction exercises were prescribed.", "kidnapped"),
    # The electrocution verb family is whole-word: benign "electro-" neighbors — an
    # "electrocardiogram" (ECG), an "electrode" on a monitor, an "electrolyte" panel — share the
    # prefix but have no word boundary before "-cute", so \belectrocuted\b / \belectrocute\b /
    # \belectrocutes\b / \belectrocuting\b must NOT fire the electrical/power critical floor.
    ("The clinic scheduled an electrocardiogram, replaced a monitor electrode, and ran an electrolyte panel.",
     "electrocuted"),
    # Only the whole-phrase "cardiac tamponade"/"pericardial tamponade" (the emergency) was added, NOT
    # the bare word "tamponade": inside medicine "tamponade" is polysemous — a therapeutic maneuver to
    # stop bleeding (balloon/uterine/nasal tamponade) — so a treatment note using it must NOT fire the
    # injury/medical critical floor.
    ("Nursing note: balloon tamponade was placed and uterine tamponade held during the procedure.",
     "cardiac tamponade"),
    # Only the distinct clinical word "asystole" was added — NOT the routine cardiac-cycle words
    # "systole"/"diastole"/"systolic" that appear in every normal blood-pressure note. \basystole\b is
    # a separate whole word and does NOT match them, so a routine vitals note must NOT fire the
    # injury/medical critical floor — proving we added only the lethal-rhythm term, not its benign
    # look-alikes.
    ("Routine vitals charted: systole 120, diastole 80, systolic trend stable.",
     "asystole"),
    # Only the QUALIFIED two-word phrases "cardiac/myocardial/ventricular rupture" were added — NOT
    # the massively polysemous bare word "rupture" (a burst pipe, a herniated disc, a torn membrane).
    # A facilities note about a water-main rupture must NOT fire the injury/medical critical floor from
    # these cardiac phrases — proving the phrase matcher added only the heart-wall catastrophe.
    ("Warehouse note: a seal rupture on the compressor; cardiac rehab class and ventricular assist device demo both ran fine.",
     "cardiac rupture"),
    # Only the QUALIFIED brain-adjacency phrases "brain bleed"/"brain bleeds"/"bleed on the brain"
    # were added — NOT the polysemous bare "bleed" ("bleed the brakes", a "bleed valve", "colors
    # bleed", "bleeding-edge"). A maintenance note with "bleed" and "brain" as non-adjacent words must
    # NOT fire the injury/medical HIGH floor — proving the phrase matcher added only the cranial-bleed
    # synonyms, not the bare word.
    ("Engineers ran a bleed-the-brakes check on the forklift; the brain-teaser kiosk and a bleed valve on the tank both passed.",
     "brain bleed"),
    # Only the QUALIFIED two-word phrases "cardiogenic shock" / "hypovolemic shock" were added —
    # NOT the polysemous bare word "shock". A non-medical "shock" note (a shock absorber, being "in
    # shock" emotionally) must NOT fire the injury/medical critical floor — proving the phrase matcher
    # added only the lethal shock states, not their benign look-alikes.
    ("Maintenance note: replaced the worn shock absorber on the forklift; driver was in shock over the cost.",
     "cardiogenic shock"),
    # Only the QUALIFIED two-word phrase "status epilepticus" was added — NOT the massively polysemous
    # bare word "status". A routine "status update/report/meeting" note must NOT fire the injury/medical
    # critical floor — proving the phrase matcher added only the neurological emergency, not its benign
    # look-alike.
    ("Ops status update: the on-call status is green and the status meeting is at noon.",
     "status epilepticus"),
    # Only the QUALIFIED two-word phrase "aortic dissection" was added — NOT the polysemous bare
    # word "dissection". A routine surgical/anatomical/figurative "dissection" note must NOT fire the
    # injury/medical critical floor — proving the phrase matcher added only the vascular emergency.
    ("Lab log: the careful surgical dissection of the tissue plane and the frog dissection both went well.",
     "aortic dissection"),
    # Only the QUALIFIED two-word phrase "cardiorespiratory arrest" was added — NOT the bare
    # adjective "cardiorespiratory". A routine cardiorespiratory monitoring/fitness note must NOT
    # fire the injury/medical critical floor — proving the phrase matcher added only the fatal arrest.
    ("Health screening: cardiorespiratory monitor attached, cardiorespiratory fitness test normal.",
     "cardiorespiratory arrest"),
    # "cyanide" and "phosgene" are whole words: the benign prefix-sharers "cyan" (cyan ink/toner) and
    # "phosphorescent"/"phosphate" must NOT fire the gas/chemical critical floor — proving \bcyanide\b /
    # \bphosgene\b matched only the toxic gases, not the color or the phosphorus compounds.
    ("Replaced the cyan ink cartridge; the phosphorescent exit sign and the phosphate rinse are fine.",
     "cyanide"),
    # The HF entries are the full two-word names, NOT bare "fluoride": a routine dental/water note
    # ("fluoride toothpaste", "water fluoridation") must NOT fire the gas/chemical critical floor —
    # proving the phrase matchers added only the toxicant, not the benign fluoride compound.
    ("Reminder: the fluoride toothpaste order and the water fluoridation report are on file.",
     "hydrofluoric acid"),
    # The explosive-device phrases ("pipe bomb"/"car bomb") are multi-word adjacency phrases, so the
    # bare polysemous "bomb" must NOT fire the security/intrusion critical floor: "bath bomb",
    # "photobomb", "the movie bombed" all embed the token benignly and none is adjacent to pipe/car.
    ("The gift shop restocked bath bombs, someone photobombed the banner, and the film bombed.",
     "pipe bomb"),
    # The O2-deficiency entries are the full phrase "oxygen deficient/-deficient atmosphere", NOT bare
    # "oxygen": a routine benign oxygen note (a refilled oxygen tank, supplemental oxygen, a liquid-oxygen
    # delivery) must NOT fire the gas/chemical critical floor — proving the phrase matchers added only the
    # confined-space hazard, not the benign gas mention.
    ("The oxygen tank was refilled and supplemental oxygen is stocked for the liquid oxygen delivery.",
     "oxygen deficient atmosphere"),
    # "decompression sickness" is the full two-word phrase, NOT bare "decompression": a routine benign
    # decompression note — archive decompression, decompression of a pressure vessel per procedure, or
    # the emergency "needle/chest decompression" done FOR a tension pneumothorax — must NOT fire the
    # injury/medical HIGH floor on its own (\bdecompression\s+sickness\b matches none of them), proving
    # the phrase matcher added only the diving injury, not the polysemous root.
    ("Ran archive decompression and completed the pressure-vessel decompression per procedure.",
     "decompression sickness"),
    # "crush syndrome" is the full two-word phrase, NOT the bare polysemous "crush"/"crushed": a benign
    # figurative use — crushing a sprint goal, crushed gravel/aggregate, a merger crushing a rival — must
    # NOT fire the injury/medical HIGH floor (\bcrush\s+syndrome\b matches none of them), proving the
    # phrase matcher added only the clinical trauma condition, not the heavily-figurative root.
    ("The team crushed the sprint goal, crushed gravel for the walkway, and the merger crushed the rival.",
     "crush syndrome"),
    # The volcanic-catastrophe phrases are MULTI-WORD, NOT the bare polysemous roots: a figurative
    # "erupted" and the geographic mention of a dormant "volcano" (neither an incident) must NOT fire
    # the weather CRITICAL floor (the phrase matchers match none of them), proving only the whole
    # zero-benign phrases were added.
    ("The team erupted in applause after the demo; the site sits miles from a dormant volcano.",
     "volcanic eruption"),
    # "storm surge" is the adjacent two-word phrase, NOT the polysemous half "surge": a benign traffic
    # surge (or power/demand/adrenaline surge) with no storm-surge phrasing must NOT fire the weather
    # CRITICAL floor. Kept storm-free so the whole case stays LOW — proving only the whole phrase fires.
    ("A traffic surge overwhelmed the checkout page during the sale; nothing else to report.",
     "storm surge"),
    # "gunman"/"gunmen" match on whole-word boundaries only: the benign "gunmetal" color must NOT fire
    # the security CRITICAL floor, proving the word boundary keeps the offender terms from firing inside
    # a larger unrelated word.
    ("The gunmetal gray cabinet was installed in the server room; nothing else to report.",
     "gunman"),
    # "grain engulfment" is the full two-word phrase, NOT bare "engulfment": the benign biology sense
    # (macrophage engulfment of apoptotic cells / engulfment of a pathogen) must NOT fire the gas/chemical
    # critical floor — proving the phrase matcher added only the confined-space agricultural hazard, not
    # the polysemous root.
    ("The lab documented macrophage engulfment of apoptotic cells during the pathogen study.",
     "grain engulfment"),
    # "agonal" matches on whole-word boundaries only: the routine geometry/structural words that merely
    # embed the substring "agonal" — a diagonal brace, a hexagonal bolt, an octagonal duct — have no word
    # boundary before the "a", so \bagonal\b must NOT fire the injury/medical critical floor, proving the
    # terminal-respiration term cannot fire from inside a larger unrelated word.
    ("The diagonal brace, hexagonal bolt, and octagonal duct were inspected; nothing else to report.",
     "agonal"),
]


def test_bare_herniation_stays_low():
    # The QUALIFIED brain-herniation phrases (uncal/brain/transtentorial/tonsillar/cerebral) escalate
    # to critical, but the bare "herniation"/"hernia" was DELIBERATELY excluded — a disc herniation or
    # an inguinal/hiatal hernia is routine, NOT a critical emergency. \bherniation\b / \bhernia\b are
    # not taxonomy signals, so these benign cases must fall through to the LOW default; a future
    # careless add of the bare noun would fire them critical and this catches it.
    for text in ("Patient has a disc herniation at L5",
                 "Inguinal hernia repair scheduled next week",
                 "Hiatal hernia found on endoscopy"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare hernia is not critical)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_intracranial_hemorrhage_stays_high_not_critical():
    # The QUALIFIED "subarachnoid hemorrhage" escalates to critical (ruptured-aneurysm event), but
    # the broader umbrella "intracranial hemorrhage" was DELIBERATELY left at the HIGH bleeding floor
    # — it can name a slow chronic subdural, not always a hyperacute emergency. Assert exact HIGH so a
    # future careless escalation of the umbrella term is caught. Both spellings, isolated on the phrase.
    for text in ("Intracranial hemorrhage noted on the follow-up scan",
                 "Chronic subdural intracranial haemorrhage under observation"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "high", f"{text!r} -> {sev}, expected exactly high (must not escalate to critical)"


def test_qualified_hemorrhage_escalates_bare_and_metaphor_do_not():
    # The QUALIFIED "massive/catastrophic/uncontrolled hemorrhage" phrases escalate to critical
    # (life-threatening exsanguinating bleed), while the BARE clinical term stays at the conservative
    # HIGH bleeding floor and the business metaphor "hemorrhaged" must not fire at all.
    for text in ("Massive hemorrhage, transfusion protocol active",
                 "Uncontrolled haemorrhage before EMS arrived"):
        sev, _ = risk.rule_layer(text)
        assert sev == "critical", f"{text!r} -> {sev}, expected critical (qualified hemorrhage)"
    # Bare, unqualified hemorrhage holds at exactly HIGH (not escalated by the new phrases).
    sev, _ = risk.rule_layer("Hemorrhage noted; responders en route")
    assert sev == "high", f"bare hemorrhage -> {sev}, expected exactly high"
    # The "hemorrhaged"/"bleeding cash" business metaphors carry no life-threat and must stay LOW —
    # \bhemorrhage\b cannot fire from the participle "hemorrhaged", and "bleeding" is deliberately not
    # escalated by a "massive"/"catastrophic" qualifier.
    sev, reasons = risk.rule_layer("The massive turnout hemorrhaged our budget forecast this quarter")
    assert sev == "low", f"budget metaphor -> {sev}, expected low"
    assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_acid_stays_low():
    # The QUALIFIED "acid attack"/"acid attacks" phrase floors security/intrusion critical (a
    # corrosive-substance assault), but the bare token "acid" was DELIBERATELY excluded — an acid
    # wash, acid rain, lactic acid buildup, and the figurative "acid test" are all routine/benign.
    # \bacid attack\b cannot fire from any of them, so these must fall through to the LOW default;
    # a future careless add of the bare noun would fire them critical and this catches it.
    for text in ("The technician used an acid wash on the condenser coils",
                 "Acid rain damaged the rooftop panels overnight",
                 "Lactic acid buildup flagged in the fermentation tank",
                 "The rollout was the acid test for the new access policy"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare acid is not critical)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_firebomb_whole_word_guard():
    # The firebomb family floors security/intrusion critical only as a WHOLE word — the matcher must
    # not fire from a larger benign word. "firebombproof" (a cladding descriptor) embeds "firebomb"
    # but is not an attack, and "fireside"/"you're fired" are unrelated "fire*" words. All must stay
    # off the firebomb critical floor; a future switch to substring matching would regress this.
    for text in ("Firebombproof cladding was installed on the exterior wall",
                 "The team held a fireside planning meeting",
                 "You are fired, effective immediately"):
        sev, reasons = risk.rule_layer(text)
        hit_terms = " ".join(reasons).lower()
        assert "firebomb" not in hit_terms, f"{text!r} wrongly matched a firebomb token: {reasons}"


def test_embedded_substring_does_not_fire_floor():
    for text, embedded_kw in NO_FALSE_POSITIVE:
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev} (false positive from {embedded_kw!r})"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_higher_floor_wins_across_categories():
    # injury(high) + fire(critical) in one report -> critical floor.
    sev, reasons = risk.rule_layer("Fire broke out and one worker was injured")
    assert sev == "critical"
    # most-severe reason should be listed first
    assert "critical" in reasons[0]
