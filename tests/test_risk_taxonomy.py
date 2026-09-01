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
    # "cerebral infarction" is the direct clinical name for an ischemic stroke — the neuro TWIN of
    # "myocardial infarction" (critical) — a radiology/EMS report writes it this way, yet it
    # previously matched nothing and dropped to LOW. Both cases isolate on the term (no other
    # critical/high token fires), so removing it regresses them to LOW.
    ("CT confirmed an acute cerebral infarction", "critical", "injury/medical"),
    ("Imaging showed a large cerebral infarction in the left hemisphere", "critical", "injury/medical"),
    # "ventricular fibrillation" is the lethal shockable rhythm of a pulseless cardiac arrest (critical)
    # — an AED/monitor report writes it this way, yet it previously matched nothing and dropped to LOW.
    # Both cases isolate on the term (no other critical/high token fires).
    ("Patient found in ventricular fibrillation; AED advised a shock", "critical", "injury/medical"),
    ("Confirmed ventricular fibrillation on the cardiac monitor", "critical", "injury/medical"),
    # "asystolic" is the adjective clinical twin of "asystole" (critical) — the word a monitor/EMS
    # report writes for a patient in asystole — yet it is not a substring of the base term and
    # previously dropped to LOW. Both cases isolate on the term (no other critical/high token fires;
    # "pulseless"/"CPR" deliberately omitted so the case proves "asystolic" alone floors critical).
    ("Patient was asystolic on the monitor when the crew arrived", "critical", "injury/medical"),
    ("Found asystolic and cold in the stairwell", "critical", "injury/medical"),
    # "cardiac standstill" is the point-of-care-echo / bedside-ultrasound synonym for asystole
    # (critical) that previously matched nothing and dropped to LOW. Both cases isolate on the term;
    # the bare word "standstill" (traffic/talks at a standstill) is NOT floored — the two-word phrase.
    ("POCUS showed cardiac standstill during the code", "critical", "injury/medical"),
    ("Bedside ultrasound confirmed cardiac standstill", "critical", "injury/medical"),
    # "airway obstruction" / "obstructed airway" is the direct clinical MECHANISM of the already-
    # critical "not breathing" / "asphyxiation" / "suffocation" (a blocked airway = no air moves),
    # yet both previously matched nothing and dropped to LOW. Each case isolates on the term (no other
    # critical/high token fires); "airway obstruction" also covers its qualified variants as a
    # substring (complete/upper/foreign-body), while the reversed "obstructed airway" is added
    # separately. The bare words "airway"/"obstruction" are NOT floored (see the FP guard below).
    ("Crew cleared a complete airway obstruction from the collapsed worker", "critical", "injury/medical"),
    ("Foreign body airway obstruction; back blows given on scene", "critical", "injury/medical"),
    ("Patient found with an obstructed airway and cyanotic", "critical", "injury/medical"),
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
    # "massive hemothorax" / "tension hemothorax" are the BLOOD twin of tension pneumothorax — blood
    # (not air) collapsing the lung, an ATLS immediately-life-threatening chest injury needing emergent
    # thoracostomy. Both qualified phrases (and the British "haemothorax" spelling) previously matched
    # nothing and dropped to LOW. Each case isolates on the phrase (no other floored token), so removing
    # the hemothorax terms regresses to LOW.
    ("Massive hemothorax on the left; chest tube drained 1.8 L, transfusion begun", "critical", "injury/medical"),
    ("Developed a tension hemothorax after the penetrating chest trauma", "critical", "injury/medical"),
    ("Massive haemothorax confirmed on the trauma scan", "critical", "injury/medical"),
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
    # "ice jam"/"ice jams" is the directly-named NWS river-ice flood hazard — broken ice damming a
    # river and backing water over its banks (NWS issues "Ice Jam Flood" warnings). Named on its own
    # it reached NO floored token and dropped to LOW: no "flood"/"flooding" substring (critical can't
    # fire), \bice storm\b does not match "ice jam", and neither "ice" nor "jam" is floored alone —
    # same whole-hazard absent-term miss as storm surge / levee failure. Floored at HIGH (if the report
    # says it IS flooding, the bare "flood"/"flooding" token independently escalates to critical). The
    # plural is a distinct token (\bice\s+jam\b does not match "ice jams"). Each sentence is kept free
    # of any other floored token (no "flood"/"burst"/"leak"/injury word), so removing the entries
    # regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("An ice jam on the river is backing water toward the intake", "high", "water/flood"),
    ("Two ice jams formed upstream of the pump house overnight", "high", "water/flood"),
    # "seiche"/"seiches" is the directly-named NOAA/NWS standing-wave flood hazard — a wind/pressure-
    # driven oscillation of an enclosed body of water that surges up one shore (the 1954 Lake Michigan
    # seiche swept people off Chicago piers). Named on its own it reached NO floored token and dropped
    # to LOW: no "flood"/"flooding"/"surge" substring (critical can't fire) and no other floored token
    # matches — same whole-hazard absent-term miss as storm surge / ice jam. Floored at HIGH (if the
    # report says it IS flooding, the bare "flood"/"flooding" token independently escalates to critical).
    # The plural is a distinct token (\bseiche\b does not match "seiches"). Each sentence is kept free of
    # any other floored token (no "flood"/"surge"/"leak"/injury word), so removing the entries regresses
    # each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A seiche on the lake pushed water over the intake berm", "high", "water/flood"),
    ("Repeated seiches slammed the harbor wall through the night", "high", "water/flood"),
    # "freshet"/"freshets" is the directly-named NWS/USGS river-flood hazard — the sudden rise and
    # overflow of a stream/river from heavy rain or spring snowmelt (NWS warns of the annual "spring
    # freshet"). Named on its own it reached NO floored token and dropped to LOW: no
    # "flood"/"flooding"/"surge" substring (critical can't fire) and no other floored token matches —
    # same whole-hazard absent-term miss as storm surge / ice jam / seiche. Floored at HIGH (if the
    # report says it IS flooding, the bare "flood"/"flooding" token independently escalates to critical).
    # The plural is a distinct token (\bfreshet\b does not match "freshets"). Each sentence is kept free
    # of any other floored token (no "flood"/"surge"/"leak"/injury word), so removing the entries
    # regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A freshet on the river overtopped the intake screens", "high", "water/flood"),
    ("Successive freshets swelled the canal past its banks", "high", "water/flood"),
    # "king tide"/"king tides" is the directly-named NOAA coastal-flood hazard — the year's highest
    # predicted tides that push seawater over low-lying shoreline infrastructure ("sunny-day"/nuisance
    # coastal inundation). Named on its own it reached NO floored token and dropped to LOW: no
    # "flood"/"flooding"/"surge" substring (critical can't fire), \bstorm\s+tide\b does not match
    # "king tide", and neither "king" nor bare "tide" is floored — same whole-hazard absent-term miss
    # as storm surge / ice jam / seiche / freshet. Floored at HIGH (if the report says it IS flooding,
    # the bare "flood"/"flooding" token independently escalates to critical). The plural is a distinct
    # token (\bking\s+tide\b does not match "king tides"). Each sentence is kept free of any other
    # floored token (no "flood"/"surge"/"leak"/injury word), so removing the entries regresses each
    # case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A king tide is pushing seawater over the intake berm", "high", "water/flood"),
    ("Successive king tides swelled the harbor past its wall", "high", "water/flood"),
    # "sneaker wave"/"sneaker waves" is the directly-named NWS/NOAA Pacific-coast life-threat hazard —
    # a sudden oversized surge that rushes far up a beach or over rocks without warning and sweeps
    # people off the shore (the NWS issues dedicated "Sneaker Wave" advisories for the OR/WA/CA coast).
    # Named on its own it reached NO floored token and dropped to LOW: no "flood"/"flooding"/"surge"
    # substring (critical can't fire) and not a substring of any weather/water token — same whole-hazard
    # absent-term miss as storm surge / ice jam / seiche / freshet / king tide. Floored at HIGH (if the
    # report says it IS flooding or sweeps a worker to their death, the bare "flood"/"flooding" or the
    # fatality/injury tokens independently escalate to critical). The plural is a distinct token
    # (\bsneaker\s+wave\b does not match "sneaker waves"). Each sentence is kept free of any other
    # floored token (no "flood"/"surge"/injury/fatality word), so removing the entries regresses each
    # case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A sneaker wave swept a crew member off the jetty", "high", "water/flood"),
    ("Repeated sneaker waves battered the shoreline work party", "high", "water/flood"),
    # "rip current"/"rip currents" is the directly-named NWS surf-zone life-threat hazard — a narrow,
    # powerful channel of water flowing swiftly away from shore that drags swimmers and shoreline crews
    # out to sea (the NWS issues dedicated "Rip Current Statement" products; it drowns more U.S.
    # beachgoers than any other surf danger). Named on its own it reached NO floored token and dropped
    # to LOW: no "flood"/"flooding"/"surge" substring (critical can't fire) and not a substring of any
    # weather/water token — same whole-hazard absent-term miss as storm surge / ice jam / seiche /
    # freshet / king tide / sneaker wave. Floored at HIGH (if the report says it IS flooding or the
    # fatality/injury tokens fire, those independently escalate to critical). The plural is a distinct
    # token (\brip\s+current\b does not match "rip currents"), and the bare polysemous "current" is
    # deliberately NOT floored. Each sentence is kept free of any other floored token (no
    # "flood"/"surge"/injury/fatality word), so removing the entries regresses each case to LOW and
    # fails the HIGH assertion (isolation; fault-injected).
    ("A rip current dragged a swimmer off the outfall apron", "high", "water/flood"),
    ("Rip currents are pulling debris away from the intake", "high", "water/flood"),
    # floodwater/floodwaters (an active inundation) is a distinct whole-word token that \bflood\b
    # does not match; the SAME singular/compound tokenization gap class as burn/burns and the weather
    # plurals. Both cases isolate on the new terms — no independent critical token fires.
    ("Floodwaters rose to the second floor of the plant", "critical", "water/flood"),
    ("Rising floodwater poured through the loading dock doors", "critical", "water/flood"),
    # "levee failure"/"levee breach" (+ plurals) is the engineering twin of the already-critical
    # "dam failure": a flood-control-embankment collapse that releases an uncontrolled inundation.
    # \bdam\s+failure\b can't match "levee failure", and the singular "levee breach" otherwise hits
    # only the security/intrusion "breach" token (HIGH, wrong category) — an active under-floor.
    # Each plural is a distinct token. No other floored critical token fires in these sentences, so
    # they isolate on the new terms (the "breach" sentence must land water/flood critical, NOT
    # security HIGH, proving the correct-category floor wins).
    ("A levee failure released the reservoir onto the substation", "critical", "water/flood"),
    ("Multiple levee failures inundated the district overnight", "critical", "water/flood"),
    ("The upstream levee breach put water over the switchyard", "critical", "water/flood"),
    ("Two levee breaches were reported along the river wall", "critical", "water/flood"),
    # "dam break"/"dam breach"/"dam burst" (+ plurals) are the plain-English/witness twins of the
    # already-critical noun "dam failure" — a reservoir-releasing structural give-way. \bdam\s+failure\b
    # can't match "dam break"/"dam burst" (dropped LOW), and the singular "dam breach" otherwise hits only
    # the security/intrusion "breach" token (HIGH, wrong category) — the same active under-floor fixed for
    # levee breach. Each plural is a distinct token. No other floored critical token fires in these
    # sentences, so they isolate on the new terms (the "breach" sentence must land water/flood critical,
    # NOT security HIGH, proving the correct-category floor wins).
    ("A dam break upstream sent water through the plant", "critical", "water/flood"),
    ("Successive dam breaks drained the reservoir onto the town", "critical", "water/flood"),
    ("The dam breach released the reservoir onto the substation", "critical", "water/flood"),
    ("Two dam breaches were reported along the spillway", "critical", "water/flood"),
    ("A dam burst overnight and swept away the access road", "critical", "water/flood"),
    # "dam bursts" (plural) is a distinct token \bdam\s+burst\b can't match — the singular floored but
    # the plural dropped LOW; the same singular->plural miss just completed for the levee cluster.
    ("Two dam bursts were reported downstream after the quake", "critical", "water/flood"),
    # "levee break"/"levee breaks" is the plain-English/witness twin of the already-critical
    # "levee failure"/"levee breach" (the Katrina phrasing). \blevee\s+failure\b / \blevee\s+breach\b
    # can't match "levee break", and neither "levee" nor "break" floors on its own, so it dropped LOW.
    # Each plural is a distinct token. No other floored critical token fires in these sentences, so
    # they isolate on the new terms.
    ("A levee break inundated the plant floor", "critical", "water/flood"),
    ("Multiple levee breaks opened along the river wall", "critical", "water/flood"),
    # "dike"/"dyke" failure/breach/break (+ plurals) is the British/Dutch synonym of the levee
    # cluster, deferred when levee landed (bare word is polysemous — geology/slur) and now added as
    # the qualified TWO-WORD phrases only, both spellings. Neither \blevee...\b nor a bare token
    # fires, so each previously dropped LOW; the singular "dike breach"/"dyke breach" otherwise hits
    # only the security "breach" token, which the new water/flood critical floor must outrank. Each
    # plural is a distinct token. No other floored critical token appears in these sentences.
    ("A dike failure released the reservoir onto the substation", "critical", "water/flood"),
    ("Repeated dike failures overwhelmed the pumping station overnight", "critical", "water/flood"),
    ("The upstream dike breach put water over the switchyard", "critical", "water/flood"),
    ("Two dike breaches were reported along the sea wall", "critical", "water/flood"),
    ("A dike break let water into the turbine hall", "critical", "water/flood"),
    ("Multiple dike breaks opened along the embankment", "critical", "water/flood"),
    ("A dyke failure drained the reservoir toward the town", "critical", "water/flood"),
    ("Successive dyke failures inundated the low-lying district", "critical", "water/flood"),
    ("The coastal dyke breach exposed the site to the sea", "critical", "water/flood"),
    ("Two dyke breaches were logged after the surge subsided", "critical", "water/flood"),
    ("A dyke break sent water through the compound", "critical", "water/flood"),
    ("Several dyke breaks appeared along the river defence", "critical", "water/flood"),
    # "floodwall" failure/breach/break (+ plurals) is the concrete/steel flood-control structure whose
    # give-way IS the canonical Katrina catastrophe (17th Street / London Avenue Canal floodwall
    # failures). It is the CLOSED compound: \bflood\b cannot fire inside "floodwall" (no boundary after
    # "flood"), and \blevee...\b/\bdam...\b don't match, so each previously dropped LOW; the singular
    # "floodwall breach" otherwise hits only the security "breach" token, which the water/flood critical
    # floor must outrank. Each plural is a distinct token. No bare "flood"/"flooded" appears in these
    # sentences, so only the "floodwall" token can reach the critical floor.
    ("A floodwall failure released the reservoir onto the substation", "critical", "water/flood"),
    ("Repeated floodwall failures overwhelmed the pumping station overnight", "critical", "water/flood"),
    ("The upstream floodwall breach put water over the switchyard", "critical", "water/flood"),
    ("Two floodwall breaches were reported along the canal", "critical", "water/flood"),
    ("A floodwall break let water into the turbine hall", "critical", "water/flood"),
    ("Multiple floodwall breaks opened along the canal wall", "critical", "water/flood"),
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
    # "debris flow"/"mudflow" (+ plurals) are the water-saturated flowing members of the rockslide
    # earth-movement family and must reach the same HIGH ground-failure floor — they name the exact
    # hazard the polysemous "mudslide"/"landslide" were excluded for, but with zero benign meaning.
    # Each previously matched nothing (no floored substring; \blava\s+flow\b can't match "debris flow")
    # and dropped to LOW. The plurals must fire too (\bmudflow\b does not match "mudflows",
    # \bdebris\s+flow\b does not match "debris flows"), the same singular->plural gap as rockslide.
    ("A debris flow swept across the access road below the ridge", "high", "structural"),
    ("Two debris flows buried the pipeline right-of-way overnight", "high", "structural"),
    ("A mudflow came down the slope and blocked the north gate", "high", "structural"),
    ("Repeated mudflows covered the lower yard this week", "high", "structural"),
    # A "rockfall" is the free-fall member of the same earth-movement family as "rockslide" and must
    # reach the same HIGH ground-failure floor — the one-word noun previously matched nothing (bare
    # "rock"/"fall" are unfloored and \brockslide\b can't match it) and dropped to LOW. The plural
    # "rockfalls" must fire too (\brockfall\b does not match "rockfalls"), the same singular->plural
    # gap as rockslide/rockslides. Neither sentence carries another floored token, so removing the
    # entries regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A rockfall struck the haul road below the highwall this morning", "high", "structural"),
    ("Repeated rockfalls closed the canyon rail line overnight", "high", "structural"),
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
    # "suicide bomber"/"suicide bombing"/"suicide bomb" (a person-borne explosive attack) is the direct
    # sibling of the already-critical explosive-device cluster "pipe bomb"/"car bomb"/"bomb threat": a
    # named person-borne bombing that previously matched nothing and dropped to LOW (bare "bomb" is
    # deliberately excluded for its benign collisions, so no other token floored it). Each surface form
    # (bomber/bombers/bombing/bombings/bomb/bombs) is a distinct token needing its own entry
    # (\bsuicide\s+bomber\b matches none of the others), the same verb/plural tokenization class as
    # firebomb/firebombs/firebombing. Whole multi-word phrases with zero benign polysemy; each case
    # isolates on the new term (no other floored token), and the bare "suicide" FP guard stays LOW.
    ("A suicide bomber approached the north gate this morning", "critical", "security/intrusion"),
    ("Two suicide bombers were seen near the visitor lobby", "critical", "security/intrusion"),
    ("A suicide bombing at the market entrance", "critical", "security/intrusion"),
    ("Reports of multiple suicide bombings across the district", "critical", "security/intrusion"),
    ("A suicide bomb was left under the bench", "critical", "security/intrusion"),
    ("Suicide bombs recovered from the parked van", "critical", "security/intrusion"),
    # "car bombing"/"car bombings" (the gerund/event name of the already-critical device "car bomb"/"car
    # bombs") is the vehicle-borne sibling of the "suicide bombing"/"suicide bombings" gerund: a named
    # vehicle-borne explosive attack that previously matched nothing and dropped to LOW (bare "bomb" is
    # deliberately excluded, so the noun floors but the gerund carried no other floored token). The gerund
    # and its plural are distinct tokens needing their own entries (\bcar\s+bomb\b matches neither
    # "car bombing" nor "car bombings" — the trailing "ing"/"ings" breaks the boundary), the same
    # verb/gerund/plural tokenization class as suicide bomb/bombing/bombings and firebomb/firebombing.
    # Whole multi-word phrases with zero benign polysemy (the cocktail is the noun "Irish Car Bomb", never
    # "car bombing"); each case isolates on the new gerund with no other floored token.
    ("A car bombing killed six outside the gate", "critical", "security/intrusion"),
    ("A series of car bombings struck the district overnight", "critical", "security/intrusion"),
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
    # "cyclonic storm"/"cyclonic storms" is the North Indian Ocean (IMD) regional name for the SAME tropical cyclone
    # that is "hurricane"/"typhoon"/"medicane" elsewhere (Amphan 2020 = a super cyclonic storm; Bhola 1970 killed
    # ~500k). WORSE than a plain absent-term miss, it ACTIVELY UNDER-FLOORED to HIGH — bare "storm" sits at the weather
    # HIGH floor, so "cyclonic storm" hit \bstorm\b and floored one level LOW of the hurricane it IS (the storm-surge
    # under-floor class). The single "cyclonic storm" token also covers the qualified escalation names (super/severe/
    # very severe cyclonic storm carry it as a substring); \bcyclonic\s+storm\b won't match the trailing "s" so the
    # plural is a distinct entry. The two-word phrase has ZERO benign meaning (industrial senses are "cyclonic
    # separator"/"cyclonic vacuum", never "cyclonic storm"). Each sentence below carries a bare "storm"/qualifier but
    # NO other CRITICAL token, so removing both entries drops each to the "storm" HIGH floor and fails the CRITICAL
    # assertion (isolation + active-under-floor; confirmed by fault injection).
    ("A super cyclonic storm is bearing down on the coastal facility", "critical", "weather"),
    ("Two cyclonic storms formed over the bay of bengal this season", "critical", "weather"),
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
    # "storm tide"/"storm tides" name the NWS total-water-level twin of storm surge (surge + tide) — the
    # actual coastal-inundation depth and historically the deadliest coastal hazard — and must reach the
    # same CRITICAL floor as the "storm surge" it accompanies. It previously scored only HIGH: it contains
    # the bare word "storm" (weather HIGH floor), so the phrase matched "storm" and under-floored one level.
    # \bstorm\s+tide\b does not match "storm tides", so each is a distinct entry. The only floored tokens in
    # each sentence are the new "storm tide"/"storm tides" (CRITICAL) and "storm" (HIGH), so removing the
    # phrase regresses each case to HIGH and fails the CRITICAL assertion (isolation).
    ("A record storm tide is overtopping the levee at the coastal site", "critical", "weather"),
    ("Two destructive storm tides reached the low-lying plant this season", "critical", "weather"),
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
    # "bushfire"/"bushfires" is the Australian/NZ and international-Commonwealth regional name for the exact
    # same event already CRITICAL as "wildfire"/"wildfires" — the 2019-20 Australian "Black Summer" bushfires
    # were one of the worst natural disasters in the country's history — yet it previously matched nothing and
    # dropped to LOW, the same regional-synonym-of-a-critical-term gap as typhoon beside hurricane. The plural
    # means \bbushfire\b does not match "bushfires", so each is a distinct entry. Neither sentence carries any
    # other floored token (no "wildfire"/"fire"/"blaze"/"storm"), so removing the entries regresses each case to
    # LOW and fails the CRITICAL assertion (isolation).
    ("A bushfire tore through the north perimeter overnight", "critical", "weather"),
    ("Three bushfires are threatening the ridge above the plant", "critical", "weather"),
    # "megafire"/"megafires" name a great wildfire (the >100,000-acre class NIFC/USFS and the press use for the
    # worst events — the 2020 August Complex "gigafire" topped a million acres), strictly worse than the
    # already-critical "wildfire" and the wildfire analogue of the already-critical "megaquake"/"megatsunami".
    # Bare "megafire" previously matched nothing and the plural "megafires" dropped to LOW: \bwildfire\b does not
    # fire inside the compound and \bmegafire\b does not match "megafires", so each is a distinct entry. Neither
    # sentence carries any other floored token (no "wildfire"/"fire"/"blaze"/"burned"/"storm"), so removing the
    # entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A megafire has jumped the containment line north of the plant", "critical", "weather"),
    ("Two megafires merged into a single front above the ridge", "critical", "weather"),
    # "gigafire"/"gigafires" name the >1,000,000-acre wildfire — the press/agency term one order above the
    # already-critical "megafire" (the 2020 August Complex "gigafire" topped a million acres), the strictly-worse
    # "giga-" step past "mega-". Bare "gigafire" matched nothing (\bwildfire\b and \bmegafire\b do not fire inside
    # the compound) and the plural means \bgigafire\b does not match "gigafires", so each is a distinct entry.
    # Neither sentence carries any other floored token (no "wildfire"/"fire"/"blaze"/"burned"/"storm"), so removing
    # the entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A gigafire has overrun the valley west of the plant", "critical", "weather"),
    ("Two gigafires merged into a single million-acre front", "critical", "weather"),
    # "pyroclastic surge"/"pyroclastic surges" name the USGS-distinguished dilute, fast-moving volcanic
    # density current — the more diffuse and historically MORE lethal sibling of the already-critical
    # "pyroclastic flow" (the 1902 Mont Pelee surge killed ~28,000 at St. Pierre). \bpyroclastic\s+flow\b
    # does not match "pyroclastic surge" (different second word), so it previously matched nothing and
    # dropped to LOW; the plural "pyroclastic surges" is a distinct token (\bpyroclastic\s+surge\b does not
    # match it). Neither sentence carries any other floored token (bare "surge" is deliberately unfloored),
    # so removing the entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A pyroclastic surge swept down the flank toward the plant", "critical", "weather"),
    ("Successive pyroclastic surges buried the village below the vent", "critical", "weather"),
    # "pyroclastic density current"/"pyroclastic density currents" is the USGS/volcanology STANDARD umbrella
    # term (PDC) for exactly the two phenomena already floored critical — the dense "pyroclastic flow" and
    # the dilute "pyroclastic surge" — and is the PREFERRED technical name in formal hazard assessments. Neither
    # \bpyroclastic\s+flow\b nor \bpyroclastic\s+surge\b matches "pyroclastic density current" (different second
    # word), so it previously matched nothing and dropped to LOW; the plural is a distinct token
    # (\bpyroclastic\s+density\s+current\b does not match "pyroclastic density currents"). Neither sentence carries
    # any other floored token (the bare oceanographic "density current" is deliberately unfloored), so removing the
    # entries regresses each case to LOW and fails the CRITICAL assertion (isolation).
    ("A pyroclastic density current swept down the flank and destroyed the outbuildings", "critical", "weather"),
    ("Successive pyroclastic density currents buried the coastal town below the vent", "critical", "weather"),
    # "firenado"/"firenados"/"firenadoes" is the closed-compound press name for the tornadic fire vortex thrown
    # off a large wildfire — a directly-named lethal hazard (the 2018 Carr Fire firenado was rated EF3-equivalent
    # and killed a firefighter), the fire sibling of the already-critical bare "tornado". As a single compound
    # word \btornado\b does NOT match it, and — unlike the spaced "fire whirl"/"fire tornado", which are already
    # floored by a bare "fire"/"tornado" token and so are deliberately left out — \bfire\b does NOT match the
    # compound either (no boundary after "fire"), so it previously dropped to LOW. Plurals/variants are distinct
    # tokens (\bfirenado\b does not match "firenados"/"firenadoes"). Each sentence is kept free of "wildfire",
    # bare "fire", and "tornado" so it carries no other floored token — removing the entries regresses each to
    # LOW and fails the CRITICAL assertion (isolation; confirmed by fault injection).
    ("A firenado jumped the containment line north of the ridge", "critical", "weather"),
    ("Two firenados merged over the dry riverbed east of town", "critical", "weather"),
    ("Successive firenadoes spun up along the eastern flank", "critical", "weather"),
    # "limnic eruption"/"limnic eruptions" (lake overturn) is the sudden release of a dissolved-CO2/CH4 charge
    # from a stratified volcanic crater lake — the deadliest volcanic gas event on record: the 1986 Lake Nyos
    # limnic eruption asphyxiated ~1,746 people and ~3,500 livestock in minutes, on the same footing as the
    # already-CRITICAL "pyroclastic flow"/"lahar". \bvolcanic\s+eruption\b does not match "limnic eruption"
    # (different first word), so it previously matched nothing and dropped to LOW; the plural is a distinct token
    # (\blimnic\s+eruption\b does not match "limnic eruptions"). Each sentence is kept free of any other floored
    # token (no "volcanic eruption", "gas", "asphyxiated", etc.), so removing the entries regresses each case to
    # LOW and fails the CRITICAL assertion (isolation; confirmed by fault injection).
    ("A limnic eruption from the crater lake blanketed the valley below", "critical", "weather"),
    ("Two limnic eruptions rolled off the shoreline toward the low ground", "critical", "weather"),
    # "supereruption"/"supereruptions" name a VEI-8 volcanic super-eruption (Toba, Yellowstone's Lava Creek) —
    # the caldera-forming, continent-scale magnitude step above the already-CRITICAL "volcanic eruption", exactly
    # as "megaquake" is to "earthquake". \bvolcanic\s+eruption\b does not fire inside the closed compound
    # "supereruption" (no space, different token), so it previously matched nothing and dropped to LOW; the plural
    # is a distinct token (\bsupereruption\b does not match "supereruptions"). Each sentence is kept free of any
    # other floored token (no "volcanic eruption", "ash cloud" etc.), so removing the entries regresses each case
    # to LOW and fails the CRITICAL assertion (isolation; confirmed by fault injection).
    ("A supereruption from the caldera buried the region under meters of ash", "critical", "weather"),
    ("Two supereruptions in the geologic record dwarfed every historic blast", "critical", "weather"),
    # "medicane"/"medicanes" (Mediterranean + hurricane) is the regional name for a tropical-like Mediterranean
    # cyclone, the same event as the already-critical "hurricane"/"typhoon" — Medicane Daniel (2023) drove the
    # Derna, Libya dam-collapse flood that killed ~11,000+. \bhurricane\b does not match the portmanteau "medicane",
    # so it previously matched nothing and dropped to LOW; the plural is a distinct token (\bmedicane\b does not
    # match "medicanes"). Each sentence is kept free of any other floored token (no "hurricane"/"cyclone"/"flood"),
    # so removing the entries regresses each case to LOW and fails the CRITICAL assertion (isolation; fault-injected).
    ("A medicane is tracking toward the coastal plant this evening", "critical", "weather"),
    ("Two medicanes formed over the basin earlier this autumn", "critical", "weather"),
    # "bomb cyclone"/"bomb cyclones" is the qualified name for a rapidly intensifying (bombogenesis) extratropical
    # cyclone — the Dec 2022 bomb cyclone drove the Buffalo blizzard that killed ~40+. The bare root "cyclone" is
    # deliberately excluded as polysemous (cyclone fence/separator); the "bomb" qualifier removes that ambiguity, the
    # same bare-vs-qualified discipline as "tropical cyclone". \bbomb\s+cyclone\b does not match the plural, so each
    # is a distinct entry. Each sentence is kept free of any other floored token (no "blizzard"/"storm"), so removing
    # the entries regresses each case to LOW and fails the CRITICAL assertion (isolation; confirmed by fault injection).
    ("A bomb cyclone is forecast to slam the seaboard overnight", "critical", "weather"),
    ("Two bomb cyclones spun up off the coast this winter", "critical", "weather"),
    # "bombogenesis" is the meteorological process name for the exact event "bomb cyclone" denotes (>=24 mb/24 h
    # pressure crash) — the SAME storm, so a forecast/AFD writing "the system underwent rapid bombogenesis" must
    # floor identically. It is a coined single word with zero benign meaning and no floored substring (bare
    # "bomb"/"cyclone" both excluded). It is an uncountable process noun (like "flooding"), so there is no plural
    # entry to test. The sentence carries no other floored token (no "cyclone"/"storm"/"blizzard"), so removing the
    # entry regresses it to LOW and fails the CRITICAL assertion (isolation; confirmed by fault injection).
    ("The offshore system underwent rapid bombogenesis before landfall", "critical", "weather"),
    # "haboob"/"haboobs" (Arabic loanword) is the NWS/meteorological name for an intense wall-of-dust storm — a
    # directly-named severe weather hazard, a kind of storm no less severe than the generic "storm" already floored
    # HIGH. As a single loanword it shares no substring with "storm", so \bstorm\b cannot fire inside it and it
    # previously dropped to LOW; the plural is a distinct token (\bhaboob\b does not match "haboobs"). Floored at HIGH
    # (visibility/traffic/respiratory hazard, not a guaranteed mass-casualty catastrophe). Each sentence is kept free
    # of any other floored token (no "storm"/"dust storm"/injury word), so removing the entries regresses each case to
    # LOW and fails the HIGH assertion (isolation; confirmed by fault injection).
    ("A haboob rolled across the site late this afternoon", "high", "weather"),
    ("Two haboobs were reported near the desert compound this month", "high", "weather"),
    # "sandstorm"/"sandstorms" is the generic name for the same wind-driven wall-of-sand hazard the haboob is an
    # intense subtype of — visibility to zero, deadly highway pileups, a respiratory hazard. The SPACED "dust storm"
    # already floors HIGH via \bstorm\b, but the CLOSED compound "sandstorm" has no word boundary before "storm", so
    # \bstorm\b cannot fire inside it and it previously dropped to LOW (the same closed-compound miss as superstorm);
    # the plural is a distinct token (\bsandstorm\b does not match "sandstorms"). Floored at HIGH beside haboob. Each
    # sentence is kept free of any other floored token (no "storm"/"dust storm"/injury word), so removing the entries
    # regresses each case to LOW and fails the HIGH assertion (isolation; confirmed by fault injection).
    ("A sandstorm swept across the access road at dawn", "high", "weather"),
    ("Two sandstorms reduced visibility near the compound this week", "high", "weather"),
    # "duststorm"/"duststorms" is the one-word variant spelling of "dust storm" and the twin of the sandstorm above —
    # the identical zero-visibility wind-driven dust hazard. The SPACED "dust storm" floors HIGH via \bstorm\b, but the
    # CLOSED compound "duststorm" has no word boundary before "storm", so \bstorm\b cannot fire inside it and no other
    # token is a substring — it previously dropped to LOW (the same closed-compound miss as sandstorm/snowstorm/windstorm);
    # the plural is a distinct token (\bduststorm\b does not match "duststorms"). Floored at HIGH beside sandstorm. Each
    # sentence is kept free of any other floored token (no "storm"/"dust storm"/injury word), so removing the entries
    # regresses each case to LOW and fails the HIGH assertion (isolation; confirmed by fault injection).
    ("A duststorm rolled over the north yard and dropped visibility to zero", "high", "weather"),
    ("Two duststorms grounded crews at the remote site this week", "high", "weather"),
    # "snowstorm"/"snowstorms" is the direct sibling of the already-floored "ice storm"/"blizzard" (both HIGH) — a winter
    # storm driving whiteout visibility, highway pileups, and roof-load hazard. The SPACED "ice storm" floors HIGH via
    # \bstorm\b, but the CLOSED compound "snowstorm" has no word boundary before "storm" (so \bstorm\b can't fire) and no
    # boundary after "snow" (so the MEDIUM \bsnow\b can't fire either) — it previously dropped to LOW (same closed-compound
    # miss as sandstorm/superstorm). The plural is a distinct token (\bsnowstorm\b does not match "snowstorms"). Floored at
    # HIGH beside sandstorm. Each sentence is kept free of any other floored token (no "storm"/"ice storm"/"blizzard"/injury
    # word), so removing the entries regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A snowstorm knocked out access to the north site overnight", "high", "weather"),
    ("Two snowstorms buried the compound approach this month", "high", "weather"),
    # "windstorm"/"windstorms" is the generic directly-named severe-wind event and the closed-compound sibling of
    # sandstorm/snowstorm — a wind event strong enough to down lines and tear roofing. Two HIGH signals should have
    # caught it and neither can: the generic "storm" floors HIGH via \bstorm\b, but the CLOSED compound "windstorm"
    # has no word boundary before "storm" so \bstorm\b can't fire inside it; and "high winds" is a literal two-word
    # phrase that does not match the single token "windstorm" — it previously dropped to LOW (same closed-compound
    # miss as sandstorm/snowstorm). The plural is a distinct token (\bwindstorm\b does not match "windstorms").
    # Floored at HIGH beside snowstorm. Each sentence is kept free of any other floored token (no "storm"/"high
    # winds"/"downed line"/injury word), so removing the entries regresses each case to LOW and fails the HIGH
    # assertion (isolation; fault-injected).
    ("A windstorm tore the roofing off the pump house", "high", "weather"),
    ("Two windstorms battered the tank farm this spring", "high", "weather"),
    # "downburst"/"microburst"/"macroburst" (and plurals) are the NWS/AMS-named downdraft severe-wind hazards —
    # a column of sinking air that spreads out as damaging straight-line winds — the closed-compound siblings of
    # windstorm. Every HIGH severe-wind signal that should have caught them cannot: "high winds" is a two-word
    # phrase that does not match the single token; there is no "wind" or "storm" token in "…burst" so \bwind\b/
    # \bstorm\b can't fire; and the flood \bburst\b phrase can't fire inside the closed compound (no boundary
    # before "burst") — they previously dropped to LOW (same closed-compound miss as windstorm/sandstorm). Each
    # plural is a distinct token (\bdownburst\b does not match "downbursts"). Floored at HIGH beside windstorm.
    # Each sentence is kept free of any other floored token (no "storm"/"high winds"/"downed line"/injury word),
    # so removing the entries regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A downburst flattened the equipment shelter near the intake", "high", "weather"),
    ("Two downbursts snapped poles along the perimeter road", "high", "weather"),
    ("A microburst hurled debris across the loading yard", "high", "weather"),
    ("Repeated microbursts peeled cladding off the north wall", "high", "weather"),
    ("A macroburst leveled the fence line behind the compound", "high", "weather"),
    ("Two macrobursts battered the tank farm within an hour", "high", "weather"),
    # "thunderstorm"/"thunderstorms" and "hailstorm"/"hailstorms" are the precipitation-storm closed
    # compounds — the remaining directly-named "-storm" severe-weather events. "thunderstorm" is literally a
    # storm (no less severe than the generic "storm" already HIGH) but the CLOSED compound has no word
    # boundary before "storm" so \bstorm\b can't fire; "hailstorm" is the twin of the already-HIGH "hail" but
    # \bhail\b can't fire (no boundary after "hail" in hail|storm) and \bstorm\b can't fire either — both
    # previously dropped to LOW (same closed-compound miss as windstorm/snowstorm, and hail-inside-hailstorm
    # mirrors snow-inside-snowstorm). Each plural is a distinct token (\bthunderstorm\b does not match
    # "thunderstorms"). Floored at HIGH beside macroburst. Each sentence is kept free of any other floored
    # token (no bare "storm"/"hail"/"lightning strike"/injury word), so removing the entries regresses each
    # case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A thunderstorm knocked out power to the north site", "high", "weather"),
    ("Two thunderstorms rolled over the compound this week", "high", "weather"),
    ("A hailstorm shattered the skylights above the pump bay", "high", "weather"),
    ("Two hailstorms dented the equipment yard this spring", "high", "weather"),
    # "squall"/"squalls" is the NWS/AMS-named severe-convective hazard — a sudden violent wind burst
    # (often with rain/hail/snow) and, as a "squall line", the linear storm system that spawns damaging
    # straight-line winds, large hail, and embedded tornadoes (the parent structure of a derecho). It is a
    # directly-named severe-weather event no less severe than the generic "storm" already HIGH, but as a
    # bare token absent from the list it shared no substring with "storm" (\bstorm\b can't fire) and
    # previously dropped to LOW — the same absent-term miss as haboob/typhoon, one tier below the CRITICAL
    # derecho. The plural is a distinct token (\bsquall\b does not match "squalls"). \bsquall\b already
    # fires inside the phrase "squall line"/"squall lines" (the space is a word boundary), so both entries
    # cover the squall-line system too with no separate phrase entry. Floored at HIGH beside macroburst/
    # thunderstorm. Each sentence carries no other floored token (no bare "storm"/"hail"/"high winds"/
    # injury word), so removing the entries regresses each case to LOW and fails the HIGH assertion
    # (isolation; fault-injected).
    ("A squall knocked a worker's scaffold tarp loose at the intake", "high", "weather"),
    ("Repeated squalls battered the offshore platform overnight", "high", "weather"),
    ("A squall line is bearing down on the compound this evening", "high", "weather"),
    # "rainstorm"/"rainstorms" is the last directly-named "-storm" precipitation compound: literally a
    # storm (word "storm" inside it), but the closed compound has no boundary before "storm" so \bstorm\b
    # can't fire, and \brain\b can't fire after "rain" either (rain|storm) — the same closed-compound miss
    # as snowstorm/thunderstorm, and the spaced "heavy rain" only reaches MEDIUM, so the single word
    # previously dropped LOW. Each plural is a distinct token (\brainstorm\b does not match "rainstorms").
    # Floored at HIGH beside squall/thunderstorm. Each sentence carries no other floored token (no bare
    # "storm"/"flood"/"lightning"), so removing the entry regresses each case and fails the assertion.
    ("A rainstorm swept over the north compound overnight", "high", "weather"),
    ("Back-to-back rainstorms battered the access road this week", "high", "weather"),
    # "cloudburst"/"cloudbursts" is the rain-sibling of rainstorm: a sudden violent torrential downpour.
    # The closed compound reaches no floored token (no "storm" for \bstorm\b, no "rain", and the flood
    # phrases "burst pipe"/"pipe burst"/"burst main" can't fire inside "…d|burst"), so the single word
    # previously dropped LOW while spaced "heavy rain" only reaches MEDIUM. Each plural is a distinct token
    # (\bcloudburst\b does not match "cloudbursts"). Floored HIGH beside rainstorm. Each sentence carries
    # no other floored token (no bare "storm"/"flood"/"lightning"/"burst pipe"), so removing the entry
    # regresses each case to LOW and fails the assertion (isolation).
    ("A cloudburst overwhelmed the drains at the north gate", "high", "weather"),
    ("Repeated cloudbursts washed out the access road overnight", "high", "weather"),
    # "monsoon"/"monsoons" is the bare absent-loanword severe-weather hazard (sibling of haboob/typhoon/
    # squall): a single token that shares no substring with any floored signal — no "storm" so \bstorm\b
    # can't fire, no "rain"/"wind" token, and the spaced "heavy rain" it brings only reaches MEDIUM — so it
    # previously dropped LOW. Each plural is a distinct token (\bmonsoon\b does not match "monsoons").
    # Floored HIGH beside cloudburst/squall. Each sentence carries no other floored token (no bare "storm"/
    # "flood"/"lightning"/injury word), so removing the entry regresses each case to LOW and fails the
    # assertion (isolation; fault-injected).
    ("A monsoon overwhelmed the site drainage during the evening shift", "high", "weather"),
    ("Successive monsoons battered the compound access road this season", "high", "weather"),
    # "waterspout"/"waterspouts" is a directly-named NWS marine severe-weather hazard (a tornado over
    # water). As a single closed compound it reaches no floored token — no "storm" for \bstorm\b, no
    # "rain"/"wind"/"hail", and "spout" is not floored — so it previously dropped LOW while its stronger
    # land cousin "tornado" floors critical. Each plural is a distinct token (\bwaterspout\b does not
    # match "waterspouts"). Floored HIGH beside monsoon/squall. Each sentence carries no other floored
    # token (no bare "storm"/"flood"/"lightning"/injury word), so removing the entry regresses each case
    # to LOW and fails the assertion (isolation; fault-injected).
    ("A waterspout came ashore and tore panels off the dock shelter", "high", "weather"),
    ("Two waterspouts were reported off the intake pier this morning", "high", "weather"),
    # "freezing rain" is the NWS glaze-ice hazard that produces the already-HIGH "ice storm": supercooled
    # rain that coats roads/walkways and loads lines until they fall. The spaced phrase reaches no floored
    # token — no "storm" for \bstorm\b, and bare "rain" is NOT floored (only "heavy rain" sits at MEDIUM,
    # and \bheavy\s+rain\b does not match "freezing rain") — so it previously dropped LOW, scoring below the
    # same event written "ice storm". It is a mass noun (like the HIGH "hail"/"snow"), so no plural entry is
    # needed. Floored HIGH beside ice storm. Each sentence carries no other floored token (no bare "storm"/
    # "ice storm"/"downed line"/injury word — "downed lines" plural does not match \bdowned\s+line\b), so
    # removing the entry regresses each case to LOW and fails the assertion (isolation; fault-injected).
    ("Freezing rain coated every walkway across the site overnight", "high", "weather"),
    ("Crews reported freezing rain on the access road before dawn", "high", "weather"),
    # "nor'easter"/"noreaster" (and plurals) name the severe Atlantic coastal storm — blizzard whiteouts,
    # hurricane-force winds, coastal flooding. As a closed/apostrophe compound it reaches no floored token
    # (it ends in "easter", so \bstorm\b cannot fire, and there is no "snow"/"wind"/"rain"/"blizzard"
    # substring) — so it previously dropped LOW, scoring below the same storm written "snowstorm". Both
    # spellings and each plural are distinct tokens (\bnor'easter\b does not match "nor'easters"). Floored
    # HIGH beside snowstorm/blizzard. Each sentence carries no other floored token (no bare "storm"/"flood"/
    # "wind"/injury word), so removing the entries regresses each case to LOW and fails the assertion
    # (isolation; fault-injected).
    ("A nor'easter cut power to the coastal substation overnight", "high", "weather"),
    ("Back-to-back nor'easters battered the north lot this week", "high", "weather"),
    ("The noreaster stranded crews at the loading dock through the weekend", "high", "weather"),
    # "heat wave"/"heatwave" (+ plurals) name the severe prolonged-extreme-heat event — the deadliest U.S.
    # weather hazard by average annual deaths and the mass-casualty form of the already-MEDIUM "heat
    # advisory". It previously dropped LOW (an inversion below its own advisory): \bheat\s+advisory\b does
    # not match "heat wave"/"heatwave" and no other token fires. Both the spaced and closed spellings, plus
    # each plural, are distinct tokens (\bheat\s+wave\b does not match "heat waves"; \bheatwave\b does not
    # match "heatwaves"). Floored HIGH above the MEDIUM advisory. Each sentence carries no other floored
    # token (no bare "storm"/"heat advisory"/injury word), so removing the entries regresses each case to
    # LOW and fails the assertion (isolation; fault-injected).
    ("A heat wave shut outdoor operations across the site for a week", "high", "weather"),
    ("Successive heat waves left day crews rotating onto early shifts", "high", "weather"),
    ("The heatwave pushed the cooling plant past its rated capacity", "high", "weather"),
    ("Two heatwaves in one month kept the yard on restricted duty", "high", "weather"),
    # "cold wave"/"cold snap"/"arctic blast"/"arctic outbreak" (+ plurals) name the directly-named
    # severe extreme-cold events — the symmetric cold-side sibling of the just-added "heat wave" (NWS
    # "Extreme Cold Warning"). Each previously dropped LOW: every phrase carries NO floored token —
    # \bstorm\b cannot fire (no "storm" substring), no "snow"/"wind"/"rain" token, and MEDIUM "frost" is
    # not a substring — so a cold-side report scored below the same-severity "heat wave" purely on
    # hot-vs-cold wording. Each plural is a distinct token (\bcold\s+wave\b does not match "cold waves").
    # Every sentence carries no other floored token (no bare "storm"/injury word), so removing the entries
    # regresses each case to LOW and fails the assertion (isolation; fault-injected). "arctic blast" also
    # proves it does NOT over-fire the electrical CRITICAL "arc blast" (\barc\b has no boundary in "arctic").
    ("A cold wave froze the outdoor feeders across the north yard", "high", "weather"),
    ("Successive cold waves kept crews off the elevated platforms", "high", "weather"),
    ("The cold snap burst the sprinkler mains in Building C", "high", "weather"),
    ("Two cold snaps this month cracked the loading-dock seals", "high", "weather"),
    ("An arctic blast knocked the coastal substation offline overnight", "high", "weather"),
    ("Back-to-back arctic blasts stranded the yard crews all week", "high", "weather"),
    ("An arctic outbreak pushed the heating plant past its rated load", "high", "weather"),
    ("Repeated arctic outbreaks froze the intake lines twice this winter", "high", "weather"),
    # "black ice" names the glaze-ice road/walkway hazard produced by the already-HIGH "freezing rain"/
    # "ice storm" — a thin transparent ice layer that causes deadly crashes and slip-and-fall injuries.
    # It previously dropped LOW: no "storm" substring so \bstorm\b cannot fire, \bice\s+storm\b does not
    # match "black ice", and MEDIUM "frost" is not a substring — so it scored below the same glaze-ice
    # event written "freezing rain". It is a mass noun (no plural entry, like freezing rain/hail/snow).
    # Both sentences carry no other floored token, so removing the entry regresses each case to LOW and
    # fails the HIGH assertion (isolation; fault-injected).
    ("Patchy black ice was reported across the access road before dawn", "high", "weather"),
    ("Black ice on the loading dock ramp closed the north entrance", "high", "weather"),
    # "polar vortex" (+ plurals "polar vortexes"/"polar vortices") names the extreme-cold driver the
    # media/NWS use interchangeably with the just-added "arctic outbreak"/"cold wave" (both HIGH). It
    # previously dropped LOW: the phrase carries NO floored token — no "storm" substring so \bstorm\b
    # cannot fire, no "snow"/"wind"/"rain" token, MEDIUM "frost" is not a substring, and
    # \barctic\s+(blast|outbreak)\b cannot match a different second word — so it scored below the same
    # extreme-cold event written "arctic outbreak". Each plural is a distinct token (\bpolar\s+vortex\b
    # matches neither "polar vortexes" nor "polar vortices"). Every sentence carries no other floored
    # token, so removing the entries regresses each case to LOW and fails the assertion (isolation;
    # fault-injected). The bare "vortex" stays unfloored (FP guard below) — only the qualified phrase fires.
    ("The polar vortex burst the intake mains across the north plant", "high", "weather"),
    ("Successive polar vortexes kept the yard crews off the towers", "high", "weather"),
    ("Two polar vortices this winter froze the outdoor feeders twice", "high", "weather"),
    # "freezing fog" names the glaze-ice producer in the same family as the already-HIGH "freezing rain"/
    # "ice storm"/"black ice" — supercooled fog droplets that freeze on contact and coat roads/catwalks/
    # conductors in clear ice. It previously dropped LOW: no "storm" substring so \bstorm\b cannot fire,
    # \bfreezing\s+rain\b does not match "freezing fog" (different second word), bare "fog" is not floored,
    # and MEDIUM "frost" is not a substring — so it scored below the same glaze-ice event written
    # "freezing rain"/"black ice". It is a mass noun (no plural entry, like freezing rain/black ice/hail).
    # Both sentences carry no other floored token, so removing the entry regresses each case to LOW and
    # fails the HIGH assertion (isolation; fault-injected).
    ("Dense freezing fog glazed the intake catwalk overnight", "high", "weather"),
    ("Freezing fog closed the north access road before the day shift", "high", "weather"),
    # "flash freeze"/"flash freezes" is the NWS winter event — a rapid temperature crash that freezes
    # standing water and wet roads to ice almost instantly, the meteorological PRODUCER of the already-HIGH
    # "black ice"/glaze-ice family. It previously dropped LOW: \bflash\s+flood\b (the critical water phrase)
    # does not match "flash freeze" (different second word), there is no bare "freeze" token, and the HIGH
    # glaze-ice phrases freezing rain/black ice/freezing fog are different words — so the same glaze-ice
    # event scored strictly below its own product written "black ice". Unlike the mass-noun black
    # ice/freezing rain it is a countable event, so the plural "flash freezes" gets its own entry
    # (\bflash\s+freeze\b does not match the trailing "s"). Both sentences carry no other floored token, so
    # removing the entries regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A flash freeze glazed every walkway on the access road", "high", "weather"),
    ("Two flash freezes this week iced the exterior catwalks", "high", "weather"),
    # "volcanic ash"/"volcanic ashfall" names the eruption's downwind hazard — the tephra plume that grounds
    # aircraft (the NOAA Volcanic Ash Advisory Centers exist because of jet-engine flameout), loads and collapses
    # roofs, and is a respiratory hazard (NWS Ashfall Warnings). It previously dropped LOW: the volcanic entries
    # above name the ERUPTION or its ground flows ("volcanic eruption"/"pyroclastic flow"/"lahar"/"lava flow"),
    # none is a substring of "volcanic ash"/"volcanic ashfall", and there is no bare "ash" token — so the active
    # volcanic emergency scored below its own eruption. Floored HIGH (dose-dependent; the eruption tokens
    # escalate to critical on their own). Qualified with "volcanic" on purpose: bare "ashfall"/"ash fall" is
    # domain-polysemous (incinerator/furnace/combustion residue) and stays excluded. Both entries are needed —
    # \bvolcanic\s+ash\b does not match the closed compound "volcanic ashfall". Each sentence carries no other
    # floored token, so removing the entries regresses each to LOW and fails the HIGH assertion (fault-injected).
    ("Heavy volcanic ash is falling across the plant and blanketing the intakes", "high", "weather"),
    ("Volcanic ashfall has loaded the warehouse roof and clogged the air handlers", "high", "weather"),
    # "red flag warning" names the NWS fire-weather PRODUCT — the warning that low humidity, wind, and dry fuels
    # have made conditions critical for rapid wildfire ignition/spread. It is the fire-side sibling of the cold
    # PRODUCTS "extreme cold warning"/"wind chill warning" (all HIGH). It previously dropped LOW: the critical fire
    # tokens name the FIRE itself ("wildfire"/"bushfire"/"conflagration"/"structure fire"), none is a substring of
    # "red flag warning", and there is no bare "warning"/"flag" token — so the fire-weather emergency scored below
    # the fire it forecasts written "wildfire". Floored HIGH (forecast of conditions, not an active burn; the fire
    # tokens escalate to critical on their own). Only the full three-word phrase fires — bare "red flag" is
    # domain-polysemous and stays LOW (FP guard below). The sentence carries no other floored token, so removing the
    # entry regresses it to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A red flag warning is in effect for the tank-farm district through this evening", "high", "weather"),
    # "dense fog" names the NWS Dense Fog Advisory hazard — visibility collapse producing deadly chain-reaction
    # pileups. It is an ADVISORY-tier product, so it floors MEDIUM (beside "heat advisory"/"frost"), one gradient
    # BELOW its dual-hazard sibling "freezing fog" (HIGH, adds black-ice glaze). It previously dropped LOW: bare
    # "fog" is unfloored (benign "fog of war"/"brain fog"), "freezing fog" is a different first word, and no
    # floored token is a substring of "dense fog". \bdense\s+fog\b also fires inside "dense fog advisory". The
    # sentences carry no other floored token, so removing the "dense fog" entry regresses each to LOW and fails
    # the MEDIUM assertion (isolation; fault-injected). Mass noun -> no plural entry.
    ("Dense fog dropped visibility to near zero and triggered a chain-reaction pileup on the highway", "medium", "weather"),
    ("A dense fog advisory is in effect for the plant approach roads through 9 AM", "medium", "weather"),
    # "graupel" (soft hail / snow pellets) is the directly-named frozen-precipitation TYPE — the same slick-surface
    # traction hazard as the MEDIUM "sleet" (it accumulates like ball bearings on roads/catwalks/stairs). It
    # previously dropped LOW: "graupel" is a wholly distinct word from sleet/hail/snow, so nothing floored is a
    # substring of it and it shares no substring with any floored token — the same frozen-precip traction hazard
    # scored MEDIUM-or-LOW purely on sleet-vs-graupel wording. Floored MEDIUM beside "sleet"/"snow" (a precipitation
    # type / advisory-grade nuisance), one gradient below the HIGH glaze-ice warning products freezing rain / black
    # ice / flash freeze. Mass noun -> no plural entry. The sentence carries no other floored token, so removing the
    # "graupel" entry regresses it to LOW and fails the MEDIUM assertion (isolation; fault-injected).
    ("Graupel pellets coated the loading-dock stairs and the day shift reported slipping", "medium", "weather"),
    # "wind chill advisory"/"wind chill advisories" names the NWS ADVISORY-tier dangerous-cold product — the exact
    # advisory-tier sibling of the HIGH WARNING products "wind chill warning"/"extreme cold warning" (same cold
    # hazard, one NWS gradient lower). It floors MEDIUM (beside "heat advisory"/"dense fog"), the same advisory->
    # MEDIUM / warning->HIGH gradient codified for dense fog beneath freezing fog. It previously dropped LOW:
    # \bwind\s+chill\s+warning\b cannot match "wind chill advisory" (different final word), bare "wind chill" is
    # unfloored (FP guard), and no floored token is a substring of it. UNLIKE the mass-noun dense fog/graupel,
    # "advisory" is countable so the plural "advisories" is a distinct token and gets its own entry. Each sentence
    # carries no other floored token, so removing the entry regresses each to LOW and fails the MEDIUM assertion
    # (isolation; fault-injected).
    ("A wind chill advisory is in effect for the outdoor yard crew through 10 AM", "medium", "weather"),
    ("Wind chill advisories were issued for the northern counties overnight", "medium", "weather"),
    # "cold weather advisory"/"cold weather advisories" is the CURRENT (2024 NWS hazard-simplification) name for the
    # ADVISORY-tier dangerous-cold product — the rename of "wind chill advisory" (also MEDIUM), the advisory-grade
    # sibling of the HIGH WARNING product "extreme cold warning" (same cold hazard, one NWS gradient lower). It floors
    # MEDIUM (beside "wind chill advisory"/"heat advisory"), the advisory -> MEDIUM / warning -> HIGH gradient. It
    # previously dropped LOW: "cold weather advisory" shares no substring with any floored weather token (the HIGH
    # "cold wave"/"cold snap"/"arctic blast" are not substrings, \bextreme\s+cold\s+warning\b is a different phrase,
    # the legacy "wind chill advisory" is different first words, and bare "cold" is unfloored — FP guard). So the
    # current-name product scored strictly BELOW its own legacy name (MEDIUM) — the same current-name-beneath-legacy
    # miss the heat family closed pairing excessive/extreme heat warning. UNLIKE mass-noun graupel/sleet, "advisory"
    # is countable so the plural "advisories" is a distinct token and gets its own entry. Each sentence carries no
    # other floored token, so removing the entry regresses each to LOW and fails the MEDIUM assertion (isolation;
    # fault-injected).
    ("A cold weather advisory is in effect for the outdoor yard crew this morning", "medium", "weather"),
    ("Cold weather advisories were issued for the northern counties overnight", "medium", "weather"),
    # "wind advisory"/"wind advisories" names the NWS ADVISORY-tier damaging-wind product (sustained 31-39 mph /
    # gusts 46-57 mph) — the advisory-grade sibling of the HIGH wind family "high winds"/"gale-force winds"/"high
    # wind warning", one NWS gradient lower, the wind counterpart of the MEDIUM "heat advisory"/"wind chill
    # advisory". It floors MEDIUM (beside "heat advisory"/"wind chill advisory"), the advisory -> MEDIUM /
    # warning -> HIGH gradient. It previously dropped LOW: "wind advisory" shares no substring with any floored
    # token (\bhigh\s+winds\b cannot match, \bwindstorm\b/\bstorm\b share no substring, "wind damage" is a different
    # final word, and bare "wind" is unfloored — FP guard). UNLIKE mass-noun graupel/sleet, "advisory" is countable
    # so the plural "advisories" is a distinct token and gets its own entry. Each sentence carries no other floored
    # token, so removing the entry regresses each to LOW and fails the MEDIUM assertion (isolation; fault-injected).
    ("A wind advisory is in effect for the high-profile-vehicle route past the site", "medium", "weather"),
    ("Wind advisories were issued for the northern counties through the afternoon", "medium", "weather"),
    # "winter weather advisory"/"winter weather advisories" names the NWS ADVISORY-tier winter-precip product
    # (snow/sleet/freezing-rain mix or light accumulation causing slick travel, below warning criteria) — the
    # advisory-grade sibling of the HIGH winter-storm family "winter storm warning"/"blizzard"/"ice storm"/
    # "snowstorm", one NWS gradient lower, the winter-storm counterpart of the MEDIUM "heat advisory"/"wind
    # advisory"/"wind chill advisory". It floors MEDIUM (beside "wind advisory"/"wind chill advisory"), the
    # advisory -> MEDIUM / warning -> HIGH gradient (the warning half "winter storm warning" already floors HIGH
    # via bare "storm"). It previously dropped LOW: "winter weather advisory" shares no substring with any floored
    # token (\bstorm\b/\bblizzard\b/\bice\s+storm\b/\bsnowstorm\b share no substring, the MEDIUM "snow"/"sleet"/
    # "wintry mix" are different words, and bare "winter"/"winter weather" is unfloored — FP guard). UNLIKE
    # mass-noun graupel/sleet, "advisory" is countable so the plural "advisories" is a distinct token and gets its
    # own entry. Each sentence carries no other floored token, so removing the entry regresses each to LOW and
    # fails the MEDIUM assertion (isolation; fault-injected).
    ("A winter weather advisory is in effect for the overnight yard crew", "medium", "weather"),
    ("Winter weather advisories were issued for the northern counties overnight", "medium", "weather"),
    # "wintry mix" names the NWS advisory-grade mixed-precipitation event (snow + sleet + freezing rain falling
    # together) — the winter-precip TYPE sibling of the MEDIUM "sleet"/"snow"/"graupel", one gradient below the
    # HIGH glaze-ice warning products freezing rain / black ice / ice storm. It previously dropped LOW: "wintry
    # mix" shares no substring with any floored weather token (bare "wintry"/"mix" are unfloored and benign), and
    # no floored token is a substring of it — the same advisory-tier precipitation miss class as graupel beside
    # sleet. Floored MEDIUM (advisory -> MEDIUM / warning -> HIGH gradient). Mass/collective phrase -> no plural.
    # The sentence carries no other floored token, so removing the "wintry mix" entry regresses it to LOW and
    # fails the MEDIUM assertion (isolation; fault-injected).
    ("A wintry mix is expected across the site approach roads tonight", "medium", "weather"),
    # "freezing drizzle" names the NWS advisory-grade glaze-ice precipitation — supercooled drops that freeze on
    # contact into a thin, treacherous coating. It is the light, advisory-tier cousin of the HIGH warning-grade
    # "freezing rain" (NWS: freezing drizzle -> Winter Weather Advisory, sustained freezing rain -> Ice Storm
    # Warning). It previously dropped LOW: "freezing drizzle" shares no substring with any floored weather token
    # (\bfreezing\s+rain\b cannot match the different final word; bare "drizzle" is unfloored and benign), and no
    # floored token is a substring of it — the same advisory-tier glaze miss class as wintry mix/graupel beside
    # their HIGH warning siblings. Floored MEDIUM (advisory -> MEDIUM / warning -> HIGH gradient). Mass noun -> no
    # plural. The sentence carries no other floored token, so removing the "freezing drizzle" entry regresses it
    # to LOW and fails the MEDIUM assertion (isolation; fault-injected).
    ("Freezing drizzle is glazing the catwalk handrails and stairs", "medium", "weather"),
    # "lake-effect snow"/"lake effect snow" names the banded localized heavy-snow regime — feet of snow in hours,
    # whiteout, roof-collapse loads (Buffalo Nov 2014 ~7 ft/13 dead). It is the severe named event on the footing of
    # its HIGH siblings snowstorm/blizzard, yet it previously scored only MEDIUM — an UNDER-FLOOR inversion (same
    # class as "heat wave" beneath "heat advisory"): bare \bsnow\b fires on the trailing "snow" word, so the
    # feet-of-snow event scored the SAME MEDIUM as a routine dusting, while \bstorm\b cannot fire and no HIGH winter
    # token is a substring of it. Both the hyphenated (\blake\-effect\s+snow\b) and spaced (\blake\s+effect\s+snow\b)
    # spellings are distinct tokens (the matcher treats the hyphen literally). Mass noun -> no plural entry. Removing
    # both entries regresses each case from HIGH to MEDIUM (bare "snow" still fires) and fails the HIGH assertion
    # (isolation; fault-injected). Sentences carry no other floored token so the assertion turns on the new entries.
    ("Lake-effect snow buried the north access road under three feet overnight", "high", "weather"),
    ("A lake effect snow band shut the east dock approach before the day shift", "high", "weather"),
    # "avalanche warning"/"snow avalanche" name the snow-mass-movement severe hazard — a snow/ice slope releasing
    # and racing downslope at highway speed, burying roads/rail/worksites and killing by burial/trauma. It is the
    # snow sibling of the HIGH earth-movement structural tokens rockslide/debris flow/mudflow and the winter-severe
    # peer of snowstorm/blizzard/lake-effect snow. The NWS product "avalanche warning" reached NO floored token and
    # dropped LOW; the physical "snow avalanche" scored only MEDIUM off bare \bsnow\b (the same under-floor as
    # lake-effect snow). Floored HIGH. Only the two qualified zero-polysemy phrases are floored — bare
    # "avalanche"/"avalanches" is EXCLUDED (figurative "an avalanche of emails/tickets"; guarded by
    # test_bare_avalanche_figurative_stays_low). These sentences carry no other floored token, so removing the two
    # entries regresses the warning case LOW and the snow-avalanche case to MEDIUM, failing the HIGH assertion
    # (isolation; fault-injected).
    ("An avalanche warning is in effect for the mountain pass above the ridge site", "high", "weather"),
    ("A snow avalanche buried the upper access road and swept a lineman downslope", "high", "weather"),
    # "avalanche watch"/"avalanche watches" names the WATCH-tier avalanche-center product — conditions developing/
    # possible, a step below the HIGH "avalanche warning" (imminent/occurring). It floors MEDIUM, the anticipatory
    # sibling completing the avalanche family's watch->MEDIUM / warning->HIGH pair (the same watch/advisory->MEDIUM /
    # warning->HIGH gradient as the wind/cold/heat/frost families). It previously dropped LOW: "avalanche watch" shares
    # no substring with any floored token (\bavalanche\s+warning\b/\bsnow\s+avalanche\b are different phrases, bare
    # "avalanche" is unfloored per test_bare_avalanche_figurative_stays_low, bare "watch" is not a token). "watch" is
    # countable so the plural "watches" is a distinct token and gets its own entry. Each sentence carries no other
    # floored token, so removing the entry regresses each to LOW and fails the MEDIUM assertion (isolation; fault-
    # injected).
    ("An avalanche watch is in effect for the northern ranges above the access road", "medium", "weather"),
    ("Avalanche watches remain posted for the backcountry corridor the crew must cross", "medium", "weather"),
    # "atmospheric river" (+ plural "atmospheric rivers") names the NWS/CW3E flood-driving moisture plume the
    # same way the already-HIGH "monsoon" names a rain-bearing driver — the long-duration rain/snow producer
    # behind West-Coast levee breaks and evacuations (Jan 2023 CA ARs: 20+ dead). It previously dropped LOW:
    # the phrase carries NO floored token — no "storm" word so \bstorm\b cannot fire, "river" is not "rain",
    # and nothing floored is a substring of it — so it scored below the same flood driver written "monsoon".
    # The plural is a distinct token (\batmospheric\s+river\b does not match "atmospheric rivers"). Every
    # sentence carries no other floored token, so removing the entries regresses each case to LOW and fails the
    # HIGH assertion (isolation; fault-injected).
    ("An atmospheric river stalled over the watershed above the plant", "high", "weather"),
    ("Back-to-back atmospheric rivers overtopped the intake channel", "high", "weather"),
    # "landspout"/"landspouts" is the direct land analogue of the already-HIGH "waterspout" — an NWS-named
    # non-supercell tornado. It previously dropped LOW: no "storm"/"tornado" substring and nothing floored inside
    # it, so the same land-vortex scored HIGH-or-LOW purely on waterspout-vs-landspout wording. Floored HIGH beside
    # "waterspout" (not critical "tornado" — the weaker EF0–EF1 end, same call as waterspout). Plural is a distinct
    # token. Each sentence carries no other floored token, so removing the entries regresses to LOW and fails the
    # HIGH assertion (isolation; fault-injected).
    ("A landspout touched down beside the tank farm", "high", "weather"),
    ("Two landspouts were sighted north of the switchyard", "high", "weather"),
    # "gustnado"/"gustnados"/"gustnadoes" (gust + tornado) is the NWS/storm-spotter name for the short-lived
    # gust-front ground whirlwind — a damaging-wind hazard (flips high-profile vehicles, tears roofing) that sits
    # below the critical "tornado" floor because it is not connected to the cloud base. It previously dropped LOW:
    # a coined closed compound, so \btornado\b cannot fire on "gust"+"nado" and no "storm" substring exists — the
    # same coined-compound miss as firenado, one tier down. Floored HIGH beside landspout/microburst. Each spelling
    # is a distinct token (\bgustnado\b matches neither plural). Each sentence carries no other floored token, so
    # removing the entries regresses to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A gustnado flipped an empty trailer near the loading dock", "high", "weather"),
    ("Two gustnados spun up along the outflow boundary east of the yard", "high", "weather"),
    ("Successive gustnadoes tore shingles off the maintenance shed", "high", "weather"),
    # "funnel cloud"/"funnel clouds" is the NWS-named tornado PRECURSOR — the rotating funnel-shaped condensation
    # cloud descending from a cumulonimbus base that is NOT yet touching the ground (the visible vortex the NWS
    # warns on before touchdown). It previously dropped LOW: no "tornado"/"storm" substring, "funnel" and "cloud"
    # are each unfloored, and nothing floored is a substring — so the same warning-stage vortex scored below its
    # siblings written "waterspout"/"landspout"/"gustnado". Floored HIGH beside those vortices (a funnel cloud has
    # not touched down; the instant it does the bare "tornado" token escalates it to critical, verified below).
    # Only the full two-word phrase fires — bare "funnel"/"cloud" are domain-polysemous and stay LOW (FP guard
    # below). Plural "funnel clouds" is a distinct token (\bfunnel\s+cloud\b won't match the trailing "s"). Each
    # sentence carries no other floored token, so removing the entries regresses to LOW and fails the HIGH
    # assertion (isolation; fault-injected).
    ("A funnel cloud was reported rotating just west of the tank farm", "high", "weather"),
    ("Two funnel clouds were sighted descending from the shelf cloud over the yard", "high", "weather"),
    # A funnel cloud that touches down IS a tornado — the bare "tornado" token must independently escalate this to
    # critical (confirms the HIGH floor is the warning-stage precursor, not a cap on the touchdown event).
    ("The funnel cloud touched down as a tornado and tracked across the substation", "critical", "weather"),
    # "heat dome"/"heat domes" names the stalled high-pressure driver of an extreme heat wave (the June 2021 PNW
    # heat dome killed hundreds). It previously dropped LOW: no "storm" word, "dome" not floored, nothing floored a
    # substring — so it scored below the same extreme-heat event written "heat wave". Floored HIGH beside "heat
    # wave", the same driver rationale as monsoon/atmospheric river. Plural is a distinct token. Each sentence
    # carries no other floored token, so removing the entries regresses to LOW and fails the HIGH assertion.
    ("A heat dome parked over the region for a week", "high", "weather"),
    ("Successive heat domes pushed the cooling loop past its limit", "high", "weather"),
    # "excessive heat warning"/"extreme heat warning" name the NWS warning-grade extreme-heat PRODUCT — the
    # life-threatening-heat counterpart of the MEDIUM "heat advisory" ("extreme heat warning" is the current NWS
    # product name adopted 2024; "excessive heat warning" is the legacy name still in wide use, so both spellings
    # are tokens). It previously dropped LOW: no "storm" word, the HIGH heat events ("heat wave"/"heat dome") are
    # not substrings, \bheat\s+advisory\b cannot match "...heat warning", and there is no bare "heat"/"warning"
    # token — so the WARNING-grade heat product scored below its own ADVISORY-grade sibling "heat advisory"
    # (MEDIUM), the advisory-beneath-warning inversion the cold family fixed by pairing wind chill advisory with
    # wind chill warning. Floored HIGH, the hot-side mirror of the HIGH extreme cold warning / wind chill warning.
    # Each sentence carries no other floored token, so removing the entries regresses to LOW and fails the HIGH
    # assertion.
    ("An excessive heat warning is in effect for the site all week", "high", "weather"),
    ("NWS posted an extreme heat warning for the facility grounds", "high", "weather"),
    # "high wind warning" names the NWS warning-grade damaging-wind PRODUCT (sustained >=40 mph / gusts >=58 mph) —
    # the warning-grade sibling of the MEDIUM advisory-grade "wind advisory", completing the wind family's
    # advisory -> MEDIUM / warning -> HIGH pair (the same pairing the heat family — heat advisory / extreme heat
    # warning — and the cold family — wind chill advisory / wind chill warning — already codify). It previously
    # dropped LOW: the floored token is the PLURAL "high winds", and \bhigh\s+winds\b cannot match the singular
    # "high wind" in "high wind warning" (winds vs wind), \bwindstorm\b/\bstorm\b share no substring, and there is
    # no bare "wind"/"warning" token — so the warning-grade product scored below its own event written "high winds"
    # (HIGH), the whole-hazard absent-term / NWS-product-name miss the cold/heat families fixed for extreme cold
    # warning / extreme heat warning. Floored HIGH beside "high winds". Singular product name (no plural entry, the
    # gale warning / extreme cold warning discipline). Each sentence carries no other floored token, so removing the
    # entry regresses each to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A high wind warning is in effect for the crane district this evening", "high", "weather"),
    ("The bureau upgraded the wind advisory to a high wind warning for the switchyard", "high", "weather"),
    # "freeze warning" names the NWS warning-grade lethal/damaging-cold PRODUCT (sub-freezing temps that kill crops,
    # burst exposed pipes, threaten unsheltered people) — the warning-grade sibling of the advisory-grade "frost
    # advisory" (already MEDIUM via the bare "frost" token), completing the frost/freeze advisory -> MEDIUM /
    # warning -> HIGH pair (the same pairing the cold family — wind chill advisory / wind chill warning — and heat
    # family — heat advisory / extreme heat warning — already codify). It previously dropped LOW: the phrase shares
    # no substring with any floored token — the HIGH freeze phrases "flash freeze"/"freezing rain"/"freezing fog"
    # are different words, \bstorm\b/\bblizzard\b share no substring, and the bare verb "freeze" is deliberately
    # unfloored (test_bare_freeze_stays_low) — so the warning-grade product scored BELOW its own MEDIUM advisory
    # sibling "frost advisory", the advisory-beneath-warning inversion the cold/heat/wind families fixed. Floored
    # HIGH beside the cold PRODUCTS "extreme cold warning"/"wind chill warning". Singular product name (no plural
    # entry, the gale warning / extreme cold warning / high wind warning discipline). Removing the entry regresses
    # the first (isolated) case to LOW and the upgrade case to MEDIUM (bare "frost" survives) — both fail the HIGH
    # assertion (fault-injected).
    ("A freeze warning is in effect for the outdoor pipe racks tonight", "high", "weather"),
    ("The bureau upgraded the frost advisory to a freeze warning for the tank farm", "high", "weather"),
    # "freeze watch"/"freeze watches" names the WATCH-tier freeze product — significant freezing temps POSSIBLE in
    # 24-36h, a step below the HIGH "freeze warning" (imminent/occurring). It floors MEDIUM, the anticipatory sibling
    # completing the freeze family's watch->MEDIUM / warning->HIGH pair (the same watch/advisory->MEDIUM /
    # warning->HIGH gradient as the wind/cold/heat/avalanche families). It previously dropped LOW: "freeze watch"
    # shares no substring with any floored token (\bfreeze\s+warning\b is a different final word, the HIGH
    # "flash freeze"/"freezing rain"/"freezing fog" are different words, bare "freeze" is unfloored per
    # test_bare_freeze_stays_low, bare "watch" is not a token). "watch" is countable so the plural "watches" is a
    # distinct token and gets its own entry. Each sentence carries no other floored token, so removing the entry
    # regresses each to LOW and fails the MEDIUM assertion (isolation; fault-injected).
    ("A freeze watch is in effect for the tank farm ahead of tonight's cold push", "medium", "weather"),
    ("Freeze watches remain posted for the outdoor pipe racks the crew services", "medium", "weather"),
    # "high wind watch"/"high wind watches" names the WATCH-tier high-wind product — sustained >=40 mph / gusts
    # >=58 mph POSSIBLE in 12-48h, a step below the HIGH "high wind warning" (imminent/occurring). It floors MEDIUM,
    # the anticipatory sibling completing the wind family's watch->MEDIUM / warning->HIGH ladder (wind advisory
    # MEDIUM / high wind warning HIGH already present; the watch tier between them was open). It previously dropped
    # LOW: "high wind watch" shares no substring with any floored token (\bhigh\s+wind\s+warning\b is a different
    # final word, the HIGH bare "high winds" is PLURAL and \bhigh\s+winds\b cannot match the singular "high wind"
    # inside it, "wind advisory" is a different phrase, bare "watch" is not a token). "watch" is countable so the
    # plural "watches" is a distinct token and gets its own entry. Each sentence carries no other floored token, so
    # removing the entry regresses each to LOW and fails the MEDIUM assertion (isolation; fault-injected).
    ("A high wind watch is posted for the tower crew working the exposed ridgeline tomorrow", "medium", "weather"),
    ("High wind watches remain up for the line crew across the northern service yard", "medium", "weather"),
    # "thundersnow" names the NWS convective winter phenomenon (thunderstorm precipitating as snow) — a marker of
    # intense snowfall rates plus lightning. It previously dropped LOW: a single closed compound, so \bsnow\b
    # (MEDIUM) cannot fire on "thunder"+"snow" and \bthunderstorm\b does not match it — the same closed-compound
    # miss closed for snowstorm/windstorm. Floored HIGH beside snowstorm/thunderstorm. Mass noun (no plural entry).
    # The sentence carries no other floored token, so removing the entry regresses to LOW and fails the HIGH
    # assertion (isolation; fault-injected).
    ("Thundersnow dumped four inches an hour on the yard", "high", "weather"),
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
    # "supercell"/"supercells" name the NWS/AMS parent severe thunderstorm (rotating-updraft storm that spawns
    # the most violent tornadoes, giant hail, and downbursts) — the sibling of the already-HIGH "thunderstorm"/
    # "squall". As a closed compound it matches no floored token (no "storm" for \bstorm\b, "cell" not floored,
    # \bthunderstorm\b/\bsquall\b are different words), so it previously dropped to LOW. \bsupercell\b does not
    # match "supercells", so each is a distinct entry. Neither sentence carries any other floored token, so
    # removing the entries regresses each case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A supercell developed over the tank farm this afternoon", "high", "weather"),
    ("Two supercells tracked across the county toward the switchyard", "high", "weather"),
    # "whiteout"/"whiteouts" name the zero-visibility winter condition that is the defining hazard of the already-HIGH
    # blizzard/snowstorm/lake-effect-snow siblings (wind-driven snow blinds drivers/crews into deadly highway pileups
    # and lost-worker searches) — a visibility hazard on the footing of the HIGH haboob. As a closed compound it
    # matches no floored token (no "storm" for \bstorm\b, no "snow" substring for the MEDIUM \bsnow\b, and blizzard/
    # snowstorm are different words), so it previously dropped to LOW. \bwhiteout\b does not match "whiteouts", so each
    # is a distinct entry. Neither sentence carries any other floored token, so removing the entries regresses each
    # case to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("A whiteout closed the highway and stranded the night crew", "high", "weather"),
    ("Two whiteouts grounded crews at the ridge site this week", "high", "weather"),
    # "sleet" is the winter precipitation peer of the already-MEDIUM "snow"/"frost" — ice pellets that
    # accumulate into a slick coating. It shares no substring with any floored token (\bsnow\b/\bfrost\b are
    # different words; the HIGH glaze-ice phrases freezing rain/black ice/freezing fog are different phrases),
    # so it previously dropped to LOW — an under-floor inversion below its snow/frost peers. Floored at MEDIUM
    # (not the HIGH glaze-ice tier): visible bouncing ice pellets, less treacherous than invisible glaze ice.
    # Neither sentence carries any other floored token (no "snow"/"frost"/"storm"), so removing the "sleet"
    # entry regresses each case from MEDIUM to LOW and fails the >= medium assertion (isolation; fault-injected).
    ("Sleet coated the north access road overnight", "medium", "weather"),
    ("Heavy sleet made the loading dock treacherous at shift change", "medium", "weather"),
    # "gale-force winds"/"gale force winds"/"gale warning" name the NWS severe-wind hazard (sustained 34-47 kt),
    # the wind-family peer of the already-HIGH "high winds"/"windstorm". They match no floored token
    # (\bhigh\s+winds\b needs "high", \bwindstorm\b/\bstorm\b share no substring, "gale warning" reaches nothing),
    # so they previously dropped to LOW. The matcher treats the hyphen literally, so the spaced form is a distinct
    # entry; the bare root "gale" is excluded as polysemous (a name / "a gale of laughter"). Each sentence carries
    # no other floored token (no "storm"/"high winds"/"fallen tree"/"downed line"/injury word), so removing the
    # entries regresses each case from HIGH to LOW and fails the HIGH assertion (isolation; fault-injected).
    ("Gale-force winds battered the offshore rig overnight", "high", "weather"),
    ("Gale force winds tore roofing off the north warehouse", "high", "weather"),
    ("A gale warning was issued for the harbor front", "high", "weather"),
    # "extreme cold warning"/"wind chill warning" name the NWS life-threatening-cold PRODUCTS — the warning-
    # tier hazard for exactly the dangerous cold the whole cold family floors HIGH (arctic blast/cold wave/
    # polar vortex). "Extreme Cold Warning" is the current NWS product (replaced "Wind Chill Warning" winter
    # 2024-25); "Wind Chill Warning" is the prior NWS + standing Environment Canada product (historical/
    # imported bulletins). Both previously dropped LOW: no "storm" substring, "cold"/"wind"/"chill"/"warning"
    # not floored alone, and \bcold\s+wave\b/\barctic\s+(blast|outbreak)\b/\bpolar\s+vortex\b cannot match a
    # different phrase — so each scored below the same event written "arctic blast" (the NWS-product-name miss
    # class of the already-HIGH "gale warning"/"severe storm warning"). Floored HIGH, not critical; singular
    # products only (no plural). Bare "wind chill"/"extreme cold" stay LOW (FP guard below). Each sentence
    # carries no other floored token, so removing the entries regresses each case to LOW and fails the HIGH
    # assertion (isolation; fault-injected).
    ("An Extreme Cold Warning is in effect; wind chills to -40F overnight", "high", "weather"),
    ("A Wind Chill Warning was posted for the yard crews before the night shift", "high", "weather"),
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
    # "sleet" floors weather MEDIUM, but the NWS synonym "ice pellets" was DELIBERATELY EXCLUDED because
    # ice-making equipment legitimately dispenses literal "ice pellets" (pellet/nugget ice) — this benign
    # equipment mention must stay LOW (locks in the polysemy-exclusion decision; only "sleet" fires).
    ("The ice maker dispensed ice pellets into the bin; nothing to report.", "ice pellets"),
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
    # heat exchanger, radiant heat, "turn up the heat" — must NOT fire the injury/medical floor.
    # (NB: "heat wave" is deliberately excluded here — it is now a real weather HIGH signal, not benign
    # filler — so this guard uses only bare-"heat" phrases that stay LOW.)
    ("The heat exchanger was serviced, we felt the radiant heat, and we turned up the heat.",
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
    # Only the one-word "rockfall"/"rockfalls" was added, NOT the spaced form: bare "rock" is routine
    # ("rock samples", "a loose rock") and must NOT fire the structural HIGH floor — \brockfall\b
    # requires the single compound word, so the separated word cannot match it. (A literal "rock fall"
    # containing the bare word "fall" would floor injury/medical MEDIUM via "fall", never this HIGH
    # entry, so the guard uses "rock" alone to isolate the compound-word claim.)
    ("The rock samples were catalogued in the display case near the quarry office.", "rockfall"),
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
    # Only the lethal QUALIFIED hemothorax phrases ("massive"/"tension") were added, NOT the bare
    # "hemothorax": a small or minimal hemothorax is routinely observed or drained with a single chest
    # tube, so the bare token must NOT fire the injury/medical critical floor (\bmassive hemothorax\b /
    # \btension hemothorax\b do not match it) — the exact bare-vs-qualified boundary drawn for
    # pneumothorax (excluded) vs tension pneumothorax, and hemorrhage (HIGH) vs massive hemorrhage.
    ("A small hemothorax was observed on the scan and managed conservatively with a chest tube.",
     "massive hemothorax"),
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


def test_bare_suicide_stays_low():
    # The QUALIFIED "suicide bomber/bombing/bomb" phrases escalate to critical (person-borne explosive
    # attack), but the bare word "suicide" was DELIBERATELY excluded — "suicide prevention", an
    # insurance "suicide clause", and a hockey "suicide pass" are benign. \bsuicide\b is not a taxonomy
    # signal, so these must fall through to the LOW default; a future careless add of the bare noun
    # would fire them critical and this catches it.
    for text in ("The employee assistance line covers suicide prevention",
                 "A suicide clause in the group life insurance policy",
                 "He took a suicide pass in the rec-league hockey game"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'suicide' is not critical)"
        assert "no risk taxonomy signals" in reasons[0].lower()


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


def test_bare_vortex_stays_low():
    # The QUALIFIED "polar vortex" floors weather HIGH (the extreme-cold driver), but the bare "vortex"
    # was DELIBERATELY left unfloored — it is routine engineering/meteorology vocabulary (a fire vortex
    # in this file's own firenado prose, vortex shedding, a vortex tube, a lab vortex mixer). \bvortex\b
    # is not a taxonomy signal, so these benign cases must fall through to the LOW default; a future
    # careless add of the bare noun would fire them HIGH and this catches it.
    for text in ("Vortex shedding observed on the stack at high flow",
                 "The lab vortex mixer needs recalibration",
                 "Vortex tube cooling on the test bench is within spec"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'vortex' is not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_wind_chill_and_extreme_cold_stay_low():
    # The QUALIFIED NWS products "extreme cold warning"/"wind chill warning" floor weather HIGH (the
    # life-threatening-cold hazard), but the bare "wind chill" and bare "extreme cold" were DELIBERATELY
    # left unfloored — a routine "wind chill of 25F", a wind-chill chart, and cold-storage/cryo "extreme
    # cold" are all benign. \bextreme cold warning\b / \bwind chill warning\b cannot fire from any of them,
    # so these must fall through to the LOW default; a future careless add of the bare noun would fire them
    # HIGH and this catches it (the polysemous-by-severity discipline of gale / gale warning).
    for text in ("The wind chill was a mild 25 degrees at the morning walkdown",
                 "Updated the wind chill chart posted in the break room",
                 "The extreme cold storage room held the samples at spec"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare wind chill / extreme cold not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_red_flag_stays_low():
    # The QUALIFIED NWS product "red flag warning" floors weather HIGH (critical fire-weather conditions), but
    # the bare idiom "red flag" was DELIBERATELY left unfloored — a code-review red flag, a beach/racing red
    # flag, and a diplomatic red flag are all benign. \bred flag warning\b cannot fire from any of them, so
    # these must fall through to the LOW default; a future careless add of a bare "red flag" token would fire
    # them HIGH and this catches it (the qualified-phrase discipline of volcanic ash / storm surge).
    for text in ("The auditor raised a red flag on the vendor invoice",
                 "A red flag flew at the beach lifeguard tower for high surf",
                 "The code review flagged a red flag in the retry logic"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'red flag' idiom not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_heat_stays_low():
    # The QUALIFIED NWS products "excessive heat warning"/"extreme heat warning" floor weather HIGH, but the
    # bare noun "heat" was DELIBERATELY left unfloored — a heat exchanger, "turn up the heat", and body heat are
    # all benign, and even the closed "heat" inside "preheat"/"reheat" must not fire. \bexcessive heat warning\b /
    # \bextreme heat warning\b cannot fire from any of them, so these must fall through to the LOW default; a
    # future careless add of a bare "heat" token would fire them HIGH and this catches it (the qualified-phrase
    # discipline of red flag warning / wind chill warning, where the bare token stays LOW).
    for text in ("The heat exchanger in bay 3 needs service before summer",
                 "Crews asked to turn up the heat in the loading dock",
                 "Body heat from the crowd fogged the lobby camera lens"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'heat' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_freeze_stays_low():
    # The two-word EVENT "flash freeze"/"flash freezes" floors weather HIGH (glaze-ice producer), but the
    # bare verb "freeze" was DELIBERATELY left unfloored — routine "freeze the sample", a freeze-frame, a
    # hiring freeze, and "the pipe may freeze" are all benign or already covered by other tokens. Only the
    # full "flash freeze" phrase fires; \bflash\s+freeze(s)?\b cannot match a bare "freeze", so these must
    # fall through to LOW. A future careless add of a bare "freeze" token would fire them HIGH — this
    # catches it (the phrase-only discipline of rip current, where bare "current" is not floored).
    for text in ("Freeze the water sample before shipping it to the lab",
                 "Management announced a hiring freeze for the quarter",
                 "The freeze-frame from the camera was saved to the archive"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare freeze not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_freeze_watch_needs_adjacency():
    # The QUALIFIED phrase "freeze watch"/"freeze watches" floors weather MEDIUM (WATCH-tier freeze product), but it
    # fires ONLY as the adjacent two-word phrase (\bfreeze\s+watch\b). A benign "freeze" and a benign "watch"
    # separated by other words must NOT trip it — a hiring freeze while reqs sit "on a watch" list, a freeze-frame
    # the security guard keeps watch over. Both bare halves are DELIBERATELY unfloored (freeze -> hiring/frame,
    # watch -> security/wristwatch), so absent adjacency these fall through to LOW; this catches a future careless
    # bare-token add or a loosened matcher (the adjacency discipline of avalanche watch / dense fog / red flag
    # warning).
    for text in ("Management announced a hiring freeze and put the open reqs on a watch list",
                 "The guard kept watch while the freeze-frame from the camera rendered"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (freeze watch needs adjacency)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_high_wind_watch_needs_adjacency():
    # The QUALIFIED phrase "high wind watch"/"high wind watches" floors weather MEDIUM (WATCH-tier high-wind
    # product), but it fires ONLY as the adjacent three-word phrase (\bhigh\s+wind\s+watch\b). A benign "high wind"
    # and a benign "watch" separated by other words must NOT trip it — the singular "high wind" is deliberately
    # unfloored (only the PLURAL "high winds" floors HIGH), and bare "watch" is unfloored (security watch /
    # wristwatch). Absent adjacency these fall through to LOW; this catches a future careless bare-token add or a
    # loosened matcher (the same adjacency discipline as freeze watch / avalanche watch / high wind warning).
    for text in ("He kept a high wind at his back while the guard stood watch by the gate",
                 "A high wind turbine spun steadily while the shift watch checked the gauges"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (high wind watch needs adjacency)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_fog_stays_low():
    # The QUALIFIED phrase "dense fog" floors weather MEDIUM (Dense Fog Advisory visibility hazard), but bare
    # "fog" was DELIBERATELY left unfloored — "fog of war", "brain fog", "light fog over the valley", and a
    # "foggy" recollection are all benign. \bdense\s+fog\b cannot fire from any of them, so these must fall
    # through to the LOW default; a future careless add of a bare "fog" token would fire them MEDIUM and this
    # catches it (the qualified-phrase discipline of volcanic ash / red flag warning, where the bare token stays
    # LOW). Note "freezing fog" (HIGH) is a distinct phrase and is not exercised here.
    for text in ("The general described the fog of war during the briefing",
                 "She had brain fog all morning after the double shift",
                 "A light fog drifted over the valley at dawn"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'fog' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_drizzle_stays_low():
    # The QUALIFIED phrase "freezing drizzle" floors weather MEDIUM (NWS advisory-grade glaze-ice precipitation),
    # but bare "drizzle" was DELIBERATELY left unfloored — a culinary "drizzle olive oil over the salad", a light
    # afternoon drizzle, and a chocolate drizzle on the break-room donuts are all benign. \bfreezing\s+drizzle\b
    # cannot fire from any of them, so these must fall through to the LOW default; a future careless add of a bare
    # "drizzle" token would fire them MEDIUM and this catches it (the qualified-phrase discipline of dense fog /
    # wintry mix / red flag warning, where the bare token stays LOW).
    for text in ("The caterer will drizzle olive oil over the salad greens",
                 "A light drizzle passed over the yard around noon",
                 "Someone left a chocolate drizzle on the break-room donuts"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'drizzle' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_cold_stays_low():
    # The QUALIFIED phrase "cold weather advisory" floors weather MEDIUM (the current 2024-NWS advisory-tier
    # dangerous-cold product), but bare "cold" and bare "cold weather" were DELIBERATELY left unfloored — "cold
    # weather gear", "a cold morning", "the cold shoulder", and a "cold" illness are all benign.
    # \bcold\s+weather\s+advisory\b cannot fire from any of them, so these must fall through to the LOW default; a
    # future careless add of a bare "cold"/"cold weather" token would fire them MEDIUM and this catches it (the
    # qualified-phrase discipline of wind chill advisory / dense fog / red flag warning, where the bare token stays
    # LOW).
    for text in ("The crew wore cold weather gear on the cold morning",
                 "She gave the vendor the cold shoulder after the dispute",
                 "Two staff called out with a bad head cold this week",
                 "Cold storage kept the samples at four degrees overnight"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'cold'/'cold weather' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_wind_stays_low():
    # The QUALIFIED phrases "wind advisory" (MEDIUM) and "high wind warning" (HIGH) floor the wind family, but bare
    # "wind" was DELIBERATELY left unfloored — "the wind picked up", "second wind", "a wind of change", and "wind
    # down the shift" are all benign. \bwind\s+advisory\b / \bhigh\s+wind\s+warning\b cannot fire from any of them,
    # so these must fall through to the LOW default; a future careless add of a bare "wind" token would fire them
    # MEDIUM/HIGH and this catches it (the qualified-phrase discipline of wind chill advisory / heat advisory / red
    # flag warning, where the bare token stays LOW).
    for text in ("The wind picked up a little around noon on the yard",
                 "The runner got a second wind on the final lap",
                 "Management framed it as a wind of change for the team",
                 "The crew will wind down the shift at six"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'wind' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_winter_weather_stays_low():
    # The QUALIFIED phrase "winter weather advisory" floors weather MEDIUM (the NWS advisory-tier winter-precip
    # product), but bare "winter" and bare "winter weather" were DELIBERATELY left unfloored — "the winter weather
    # on their break", "winter weather gear", "winter maintenance", and a "winter shutdown" are all benign.
    # \bwinter\s+weather\s+advisory\b cannot fire from any of them, so these must fall through to the LOW default; a
    # future careless add of a bare "winter"/"winter weather" token would fire them MEDIUM and this catches it (the
    # qualified-phrase discipline of wind advisory / cold weather advisory / heat advisory, where the bare token
    # stays LOW).
    for text in ("The crew enjoyed the winter weather on their break",
                 "The team stocked winter weather gear before the season",
                 "A winter maintenance advisory was posted for the parking deck",
                 "The plant scheduled its winter shutdown for late December"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'winter'/'winter weather' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_avalanche_figurative_stays_low():
    # The QUALIFIED phrases "avalanche warning" (NWS/avalanche-center product) and "snow avalanche" (the physical
    # event) floor weather HIGH, but the bare root "avalanche"/"avalanches" was DELIBERATELY left unfloored — the
    # figurative "an avalanche of emails / support tickets / paperwork / complaints" is routine ops language and
    # denotes no hazard. Neither \bavalanche\s+warning\b nor \bsnow\s+avalanche\b can fire from a bare "avalanche of
    # X", so these must fall through to the LOW default; a future careless add of a bare "avalanche" token would fire
    # them HIGH and this catches it (the qualified-phrase discipline of gale-force winds/gale warning, where the bare
    # polysemous "gale" stays LOW).
    # The last sentence also proves the qualified "avalanche watch" token needs ADJACENCY: a figurative
    # "avalanche" and a benign "watch" separated by other words must NOT fire \bavalanche\s+watch\b.
    for text in ("The team is buried under an avalanche of support tickets this morning",
                 "An avalanche of paperwork landed on the compliance desk after the audit",
                 "Marketing faced an avalanche of customer emails after the launch",
                 "An avalanche of tickets hit the queue while the guard was on watch overnight"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'avalanche' not floored)"
        assert "no risk taxonomy signals" in reasons[0].lower()


def test_bare_funnel_and_cloud_stay_low():
    # The QUALIFIED phrase "funnel cloud" floors weather HIGH (NWS tornado precursor), but bare "funnel" and bare
    # "cloud" were DELIBERATELY left unfloored — a funnel cake, a sales funnel, a drain funnel, cloud computing,
    # cloud cover, and a cloud storage outage are all benign. \bfunnel\s+cloud\b cannot fire from any of them, so
    # these must fall through to the LOW default; a future careless add of a bare "funnel" or "cloud" token would
    # fire them HIGH and this catches it (the qualified-phrase discipline of red flag warning / volcanic ash /
    # storm surge, where the bare token stays LOW).
    for text in ("The vendor served funnel cake at the plant open house",
                 "Marketing reviewed the sales funnel conversion metrics",
                 "The team migrated the database to the cloud last quarter",
                 "Low cloud cover drifted over the valley at dawn"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare 'funnel'/'cloud' not floored)"
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


def test_bare_airway_and_obstruction_stay_low():
    # The two-word phrases "airway obstruction"/"obstructed airway" floor injury/medical critical,
    # but the bare tokens were DELIBERATELY excluded: "airway" (a flight corridor / ventilation duct)
    # and "obstruction" ("obstruction of justice", "an obstruction on the track", a routine "bowel
    # obstruction") are not by themselves the airway emergency. The adjacency phrases cannot fire from
    # any of them, so these must fall through to LOW; a future careless add of a bare token catches here.
    for text in ("Filed a flight plan along the northern airway",
                 "Maintenance cleared an obstruction from the ventilation airway duct — reversed order, no adjacency",
                 "Charged with obstruction of justice after the audit",
                 "A fallen branch was an obstruction on the access track"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare airway/obstruction is not critical)"
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


def test_qualified_stroke_escalates_bare_and_idiom_do_not():
    # A stroke ("brain attack") is a time-critical emergency on the heart-attack tier: the QUALIFIED
    # multi-word forms a reporter actually writes must floor injury/medical critical.
    for text in ("Employee collapsed at the desk, suspected stroke, called 911",
                 "CT confirmed an ischemic stroke",
                 "Hemorrhagic stroke, transported by ambulance",
                 "Acute stroke in progress on the floor",
                 "Stroke victim found unresponsive in the break room",
                 "EMS report notes a cerebrovascular accident",
                 "CT confirmed an acute cerebral infarction"):
        sev, _ = risk.rule_layer(text)
        assert sev == "critical", f"{text!r} -> {sev}, expected critical (qualified stroke)"
    # "cerebral infarction" (the clinical twin of "myocardial infarction") floors critical, but the
    # bare short form "cerebral infarct" was DELIBERATELY excluded — an "old cerebral infarct" is
    # routinely an incidental chronic radiology finding (a severity judgment, not a clean miss), and
    # "massive stroke" was excluded because it fires inside the idiom "a massive stroke of luck".
    # Both must fall through to the LOW default.
    for text in ("Old cerebral infarct noted incidentally on the scan",
                 "That was a massive stroke of luck for the whole team"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (excluded stroke short-form/idiom)"
        assert "no risk taxonomy signals" in reasons[0].lower()
    # The bare polysemous word "stroke" was DELIBERATELY excluded — brush/swim/key strokes, a
    # "stroke of luck/genius", back/breaststroke and a two-stroke engine are all routine/benign and
    # must fall through to the LOW default; a future careless add of the bare noun would fire them.
    for text in ("The artist added a final brush stroke to the mural",
                 "It was a lucky stroke of genius by the design team",
                 "He swam the breaststroke leg of the relay",
                 "The generator uses a two-stroke engine"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (bare/idiom stroke is not critical)"
        assert "no risk taxonomy signals" in reasons[0].lower()
    # The idiom-substring forms "having a/suffered a stroke" were excluded so they cannot fire from
    # inside "having a stroke of genius" / "suffered a stroke of bad luck"; both must stay LOW.
    for text in ("That was having a stroke of genius on the rollout",
                 "The startup suffered a stroke of bad luck this quarter"):
        sev, reasons = risk.rule_layer(text)
        assert sev == "low", f"{text!r} -> {sev}, expected low (stroke idiom must not fire)"
        assert "no risk taxonomy signals" in reasons[0].lower()
