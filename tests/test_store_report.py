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
