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
