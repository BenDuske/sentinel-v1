"""Store CRUD + Markdown report export + offline AI fallbacks. No network/LLM."""
import importlib

from sentinel import models, report, risk_fallback


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "sentinel.db"))
    from sentinel import config, store
    importlib.reload(config)
    importlib.reload(store)
    return store


def test_store_crud_roundtrip(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    inc = models.new_incident("Server room flood", "burst pipe, water everywhere")
    store.save(inc)

    got = store.get(inc["id"])
    assert got is not None and got["title"] == "Server room flood"

    # update persists
    got["status"] = "resolved"
    got["severity"] = "high"
    store.save(got)
    assert store.get(inc["id"])["status"] == "resolved"

    # list returns it
    ids = [i["id"] for i in store.list_all()]
    assert inc["id"] in ids


def test_store_list_orders_newest_first(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    a = models.new_incident("first")
    a["created_at"] = 1000.0
    store.save(a)
    b = models.new_incident("second")
    b["created_at"] = 2000.0
    store.save(b)
    titles = [i["title"] for i in store.list_all()]
    assert titles.index("second") < titles.index("first")


def test_get_missing_returns_none(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    assert store.get("does-not-exist") is None


def test_store_does_not_leak_sqlite_connections(tmp_path, monkeypatch):
    """Regression guard for the connection leak: ``with sqlite3.connect(...)`` commits but never
    closes, so each save/get/list_all leaked a Connection until GC (a ResourceWarning). Assert the
    store closes its connections deterministically — no unclosed-database warning under -W error."""
    import gc
    import sqlite3
    import warnings

    store = _fresh_store(tmp_path, monkeypatch)
    inc = models.new_incident("Leak check", "burst pipe")

    # Track that every connection opened during CRUD reaches a closed state deterministically
    # (before any GC pass), rather than lingering open until finalization.
    opened = []
    real_connect = sqlite3.connect

    def _tracking_connect(*a, **k):
        c = real_connect(*a, **k)
        opened.append(c)
        return c

    monkeypatch.setattr(sqlite3, "connect", _tracking_connect)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        store.save(inc)
        store.get(inc["id"])
        store.list_all()
        gc.collect()  # would raise the ResourceWarning-as-error here if a conn were left unclosed

    assert opened, "no sqlite connections were opened — test wired up wrong"
    for c in opened:
        # A closed connection raises ProgrammingError on use; an open one would succeed (leak).
        try:
            c.execute("SELECT 1")
            assert False, "store left a sqlite connection open (leak)"
        except sqlite3.ProgrammingError:
            pass


def test_markdown_report_contains_all_sections():
    inc = models.new_incident("Warehouse fire", "fire broke out, two injured")
    inc["severity"] = "critical"
    inc["severity_rationale"] = "Rule layer → critical. AI judgment → critical."
    inc["summary"] = "A fire occurred in the warehouse; two workers were injured."
    inc["recommended_actions"] = ["Evacuate", "Call fire department", "Document damage"]
    inc["evidence"] = ["photo1.jpg"]
    md = report.to_markdown(inc)
    assert "# Incident Report — Warehouse fire" in md
    assert "CRITICAL" in md
    assert "Rule layer → critical" in md
    assert "two workers were injured" in md
    assert "1. Evacuate" in md
    assert "photo1.jpg" in md


def test_markdown_report_handles_empty_fields():
    inc = models.new_incident("Minor note")
    md = report.to_markdown(inc)
    assert "# Incident Report — Minor note" in md
    assert "_none_" in md  # no actions
    assert "_none attached_" in md  # no evidence


def test_markdown_report_handles_missing_or_null_created_at():
    """Regression guard: created_at is the one report field whose absence/null was unhandled.

    A stored incident that is valid JSON but carries created_at=null (hand-edit / partial
    migration / foreign writer — the store column has no NOT NULL constraint and save() persists
    None) made ``fromtimestamp(None)`` raise TypeError → a 500 on the customer-facing export; an
    absent key silently rendered a misleading 1969 epoch date. Both must degrade to "unknown",
    exactly like every other field's placeholder. Mutation check: restoring
    ``fromtimestamp(inc.get("created_at", 0))`` crashes on the null case and prints 1969 on the
    missing case, failing this test.
    """
    # present-but-null: must NOT raise, must read "unknown"
    null_ts = models.new_incident("Null timestamp")
    null_ts["created_at"] = None
    md = report.to_markdown(null_ts)
    assert "**Reported:** unknown" in md
    assert "1969" not in md and "1970" not in md

    # key absent entirely: must not fall back to the epoch date
    missing = models.new_incident("Missing timestamp")
    missing.pop("created_at", None)
    md2 = report.to_markdown(missing)
    assert "**Reported:** unknown" in md2
    assert "1969" not in md2 and "1970" not in md2

    # non-numeric (e.g. an ISO string a foreign writer stored) also degrades, never crashes
    stringy = models.new_incident("Stringy timestamp")
    stringy["created_at"] = "2026-07-04T12:00:00"
    assert "**Reported:** unknown" in report.to_markdown(stringy)

    # a genuine numeric timestamp still renders normally (behaviour preserved)
    ok = models.new_incident("Normal timestamp")
    ok["created_at"] = 1_700_000_000.0  # 2023-11, a real date
    ok_md = report.to_markdown(ok)
    assert "**Reported:** 2023-11" in ok_md
    assert "unknown" not in ok_md


def test_markdown_report_handles_null_actions_and_evidence():
    """Regression guard: to_markdown must survive a stored-but-null recommended_actions/evidence.

    new_incident seeds both as []; but a valid-JSON incident carrying either field as null (same
    hand-edit / partial-migration / foreign-writer path as the created_at case — no NOT NULL
    constraint) made ``enumerate(inc.get("recommended_actions", []))`` iterate None → TypeError →
    a 500 on the PRIMARY (Markdown) export, even though the optional PDF renderer already coerces
    both with ``or []``. Both export paths must be at parity. Mutation check: reverting to the
    bare ``inc.get(..., [])`` (no ``or []``) re-raises TypeError here.
    """
    inc = models.new_incident("Null actions/evidence")
    inc["recommended_actions"] = None
    inc["evidence"] = None
    md = report.to_markdown(inc)  # must not raise
    assert "_none_" in md            # actions degrade to the empty placeholder
    assert "_none attached_" in md   # evidence degrades to the empty placeholder

    # a real list still renders (behaviour preserved)
    filled = models.new_incident("Filled")
    filled["recommended_actions"] = ["Evacuate the area", "Call it in"]
    filled["evidence"] = ["photo1.jpg"]
    md2 = report.to_markdown(filled)
    assert "1. Evacuate the area" in md2 and "2. Call it in" in md2
    assert "- photo1.jpg" in md2


def test_markdown_severity_matches_pdf_coercion_for_null_or_empty():
    """Regression guard: the two customer-facing exports must AGREE on severity for the same
    incident. The PDF path coerces severity via ``str(inc.get("severity","") or "unscored")`` →
    "UNSCORED" for a null/empty severity, but to_markdown used ``str(inc.get("severity",""))`` →
    "NONE" for null (present-but-null, valid JSON, no NOT NULL constraint — the hand-edit /
    partial-migration / foreign-writer path) and a blank "## Severity:" for "". So the same stored
    incident rendered UNSCORED in the PDF but NONE / blank in the Markdown — a parity break of the
    exact class as [sentinel-report-null-lists] and [sentinel-report-created-at]. Both paths now
    fall back to "unscored". Mutation check: reverting to ``inc.get("severity","")`` renders
    "## Severity: NONE" and fails the null assertion below.
    """
    for bad in (None, ""):
        inc = models.new_incident("Sev parity")
        inc["severity"] = bad
        md = report.to_markdown(inc)
        assert "## Severity: UNSCORED" in md          # matches the PDF path's coercion
        assert "## Severity: NONE" not in md
        assert "## Severity: \n" not in md            # not left blank for the "" case

    # a real severity still renders verbatim (behaviour preserved)
    ok = models.new_incident("Real sev")
    ok["severity"] = "critical"
    assert "## Severity: CRITICAL" in report.to_markdown(ok)


def test_markdown_summary_matches_pdf_placeholder_when_empty():
    """Regression guard: the two exports must AGREE on the Summary field. The PDF path falls back
    to ``inc.get("summary") or inc.get("description","") or "—"`` so an incident with an empty
    summary AND empty description renders "—"; to_markdown used ``... or inc.get("description","")``
    with NO final placeholder, so the same incident rendered a BLANK "## Summary" section — a parity
    break of the exact class as the severity / null-lists / created_at fixes. Reachable through the
    NORMAL API (POST /api/incidents with analyze=false and an empty description leaves summary=""
    and description=""), not just a foreign writer. Both paths now fall back to "—". Mutation check:
    dropping the trailing ``or '—'`` renders a blank section and fails the assertion below.
    """
    inc = models.new_incident("Gate left open")  # analyze=false path: summary="" and description=""
    assert inc["summary"] == "" and inc["description"] == ""
    md = report.to_markdown(inc)
    summary_section = md.split("## Summary", 1)[1].split("##", 1)[0]
    assert summary_section.strip() != ""            # never blank (matches the PDF path)
    assert summary_section.strip() == "—"           # same placeholder the PDF renders

    # description present but no summary → description fills in (behaviour preserved)
    with_desc = models.new_incident("Leak", "water pooling under the AHU")
    assert "water pooling under the AHU" in report.to_markdown(with_desc)

    # real summary still renders verbatim (behaviour preserved)
    with_sum = models.new_incident("Outage")
    with_sum["summary"] = "Primary feed lost; failover held."
    assert "Primary feed lost; failover held." in report.to_markdown(with_sum)


def test_markdown_and_pdf_degrade_null_or_empty_title_status_id():
    """Regression guard: title/status/id must degrade to a placeholder like every other field.

    Every customer-facing field has a placeholder (created_at→"unknown", severity→"unscored",
    summary→"—", actions/evidence→"none"), but title/id/status used a bare ``.get(key, default)``
    whose default only fires on an ABSENT key — a present-but-empty title (reachable via the NORMAL
    API: a whitespace-only title strips to "") rendered a blank ``# Incident Report — `` heading,
    and a present-but-null title/id/status (hand-edit / partial migration / foreign writer — the
    store has no NOT NULL constraint) rendered the literal "None". Both export paths must coerce
    identically. Mutation check: reverting either renderer to ``.get('title','(untitled)')`` etc.
    renders "None"/blank and fails the assertions below.
    """
    # empty title (whitespace strips to "" via new_incident) — reachable through the public API
    empty = models.new_incident("   ")
    assert empty["title"] == ""
    md = report.to_markdown(empty)
    assert "# Incident Report — (untitled)" in md
    assert "# Incident Report — \n" not in md  # never a blank heading

    # present-but-null title/status/id all degrade, none render "None"
    nulls = models.new_incident("placeholder")
    nulls["title"] = None
    nulls["status"] = None
    nulls["id"] = None
    md2 = report.to_markdown(nulls)
    assert "# Incident Report — (untitled)" in md2
    assert "**Status:** unknown" in md2
    assert "**ID:** `—`" in md2
    assert "None" not in md2

    # the PDF path coerces the same way (build must not embed "None" for the title/status/id)
    import importlib.util
    if importlib.util.find_spec("reportlab") is not None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = f"{d}/r.pdf"
            assert report.to_pdf(nulls, out) is True  # renders without raising

    # a real title/status/id still renders verbatim (behaviour preserved)
    ok = models.new_incident("Server room flood")
    ok["status"] = "reviewing"
    ok_md = report.to_markdown(ok)
    assert "# Incident Report — Server room flood" in ok_md
    assert "**Status:** reviewing" in ok_md
    assert "(untitled)" not in ok_md and "unknown" not in ok_md


def test_fallback_summary_is_category_aware():
    inc = models.new_incident("Gas leak", "carbon monoxide alarm triggered in the basement")
    inc["severity"] = "critical"
    s = risk_fallback.fallback_summary(inc)
    assert "gas/chemical" in s
    assert "critical" in s
    assert len(s) > 40


def test_fallback_actions_are_concrete_and_category_aware():
    inc = models.new_incident("Break-in", "forced entry overnight, equipment stolen")
    acts = risk_fallback.fallback_actions(inc)
    assert 3 <= len(acts) <= 5
    joined = " ".join(acts).lower()
    # security + theft categories should both surface
    assert "law enforcement" in joined or "police" in joined


def test_fallback_categories_use_whole_word_matching():
    """Regression guard: the offline path must match categories on WHOLE words, exactly like the
    scoring layer (risk.rule_layer / risk._MATCHERS). A raw-substring implementation here fires
    "armed" inside "unarmed" and "fire" inside "firearm", so a benign all-clear report would get a
    security/intrusion or fire/smoke summary + "notify law enforcement" actions — and, worse, the
    offline output would disagree with the audited score's own rule-layer rationale.
    """
    from sentinel import risk

    # "unarmed ... all clear": substring matching sees "armed" (security/intrusion). Whole-word must not.
    benign_security = models.new_incident("Unarmed guard completed patrol", "all clear, nothing to report")
    cats = risk_fallback._matched_categories(
        f"{benign_security['title']}. {benign_security['description']}")
    assert "security/intrusion" not in cats
    acts = " ".join(risk_fallback.fallback_actions(benign_security)).lower()
    assert "law enforcement" not in acts and "credentials" not in acts

    # "firearm" must not fire the fire/smoke category via the embedded "fire".
    firearm = models.new_incident("Firearm brought to the training range", "routine, supervised")
    assert "fire/smoke" not in risk_fallback._matched_categories(
        f"{firearm['title']}. {firearm['description']}")

    # The offline categories must AGREE with the scoring layer's rule-layer categories for the same
    # text — that consistency is the whole point of grounding both on risk._MATCHERS.
    for text in ("Unarmed guard completed patrol, all clear",
                 "Firearm brought to the training range",
                 "Gas leak: carbon monoxide alarm triggered"):
        _, reasons = risk.rule_layer(text)
        score_cats = {r.split(" → ")[0] for r in reasons if "→" in r}
        assert set(risk_fallback._matched_categories(text)) == score_cats, text


def test_every_taxonomy_category_has_tailored_fallback_actions():
    """Regression guard: the offline path must stay category-aware for EVERY taxonomy
    category. If a future taxonomy category is added without matching _CATEGORY_ACTIONS,
    its offline output would silently degrade to the generic-only list — catch that here."""
    from sentinel import risk

    tax = set(risk.TAXONOMY.keys())
    act = set(risk_fallback._CATEGORY_ACTIONS.keys())
    missing = tax - act
    orphan = act - tax
    assert not missing, f"taxonomy categories with no tailored fallback actions: {sorted(missing)}"
    assert not orphan, f"_CATEGORY_ACTIONS keys with no matching taxonomy category: {sorted(orphan)}"
