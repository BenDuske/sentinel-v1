"""Real-PDF export header check + offline seed-script smoke test.

Both run with NO network/LLM: the PDF is rendered by reportlab (skipped if it isn't installed),
and the seed script is forced offline so severity comes from the grounded rule layer and the
summary/actions from the deterministic fallbacks.
"""
import importlib
import importlib.util
import os

import pytest


def test_pdf_export_starts_with_pdf_header(tmp_path, monkeypatch):
    # Skip cleanly when the optional PDF extra isn't installed — the app's 501 fallback path is
    # already covered by test_report_pdf_graceful_without_reportlab in test_flow.py.
    pytest.importorskip("reportlab")
    from sentinel import models, report

    inc = models.new_incident("Kitchen fire", "grease fire on the stovetop, heavy smoke, minor burn")
    inc["severity"] = "critical"
    inc["severity_rationale"] = "Rule layer → critical. AI judgment → high. Final = critical."
    inc["summary"] = "A grease fire ignited on the stovetop and spread to the cabinets."
    inc["recommended_actions"] = ["Evacuate", "Call the fire department", "Document the damage"]
    inc["evidence"] = ["photo1.jpg"]

    out = str(tmp_path / "report.pdf")
    assert report.to_pdf(inc, out) is True
    data = open(out, "rb").read()
    assert len(data) > 800              # a real, non-empty document
    assert data[:4] == b"%PDF"          # valid PDF magic header


def _load_seed_module():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "scripts", "seed_demo.py")
    spec = importlib.util.spec_from_file_location("seed_demo", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_demo_populates_store_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "seed.db"))
    monkeypatch.setenv("SENTINEL_ASSUME_CONSENT", "1")
    # Reload config + dependents so the per-test DB takes effect.
    from sentinel import config, store, risk_fallback, ai, risk
    for m in (config, store, risk_fallback, ai, risk):
        importlib.reload(m)
    # Force fully offline so the seed is deterministic (rule layer + fallbacks, no network).
    monkeypatch.setattr(ai, "chat", lambda *a, **k: "")
    monkeypatch.setattr(risk.ai, "llm_severity", lambda text: "")

    seed = _load_seed_module()
    created = seed.seed(reset=True)

    assert len(created) == 5
    rows = store.list_all()
    assert len(rows) == 5
    for inc in rows:
        assert inc["severity"] in ("low", "medium", "high", "critical")
        assert inc["summary"]
        assert len(inc["recommended_actions"]) >= 3
        assert inc["ai_generated"] is True
        assert "Rule layer →" in inc["severity_rationale"]
