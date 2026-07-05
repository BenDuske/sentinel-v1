"""Real-PDF export header check + offline seed-script smoke test.

Both run with NO network/LLM: the PDF is rendered by reportlab (skipped if it isn't installed),
and the seed script is forced offline so severity comes from the grounded rule layer and the
summary/actions from the deterministic fallbacks.
"""
import builtins
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
    with open(out, "rb") as fh:
        data = fh.read()
    assert len(data) > 800              # a real, non-empty document
    assert data[:4] == b"%PDF"          # valid PDF magic header


def test_pdf_export_survives_markup_in_severity(tmp_path):
    """The severity badge must _esc its value like every other text field.

    Regression for the one field that reached a reportlab Paragraph RAW: a severity carrying an
    unclosed inline tag (e.g. "x <b y" — <b> is a real reportlab bold tag) crashed doc.build with
    an uncaught ValueError → a 500 on /report.pdf, while to_markdown rendered the same string fine
    (an MD↔PDF parity break). Such a severity is reachable via a hand-edit / partial migration /
    foreign writer — the store column has no CHECK/NOT NULL constraint and save() writes it
    verbatim; the normal API only ever stores low/medium/high/critical/unscored.
    """
    pytest.importorskip("reportlab")
    from sentinel import models, report

    inc = models.new_incident("Server room water intrusion", "pipe burst overnight")
    inc["severity"] = "x <b y"          # unclosed reportlab tag → pre-fix ValueError in doc.build

    # to_markdown handles ANY severity string (plain text) — pin the parity baseline.
    md = report.to_markdown(inc)
    assert "## Severity: X <B Y" in md

    # to_pdf must now degrade gracefully (escape, don't crash) and still emit a valid PDF.
    out = str(tmp_path / "markup.pdf")
    assert report.to_pdf(inc, out) is True
    with open(out, "rb") as fh:
        assert fh.read(4) == b"%PDF"

    # Every normal severity is unaffected — _esc is a no-op on markup-free text.
    for sev in ("low", "medium", "high", "critical", "unscored"):
        inc["severity"] = sev
        assert report.to_pdf(inc, str(tmp_path / f"{sev}.pdf")) is True


def test_pdf_export_returns_false_when_reportlab_absent(tmp_path, monkeypatch):
    """The reportlab-absent branch INSIDE report.to_pdf itself (not the app's 501 wrapper).

    reportlab is now a core dep so this import never fails at runtime, leaving the graceful
    `except -> return False` path untested. Simulate an install without the optional renderer by
    making any `reportlab*` import raise, then assert to_pdf degrades to False and writes NO file
    (the caller turns that False into a clean 501 — the PDF path is never required).
    """
    from sentinel import models, report

    real_import = builtins.__import__

    def _no_reportlab(name, *args, **kwargs):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("simulated: reportlab not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_reportlab)

    inc = models.new_incident("Gas leak", "strong odor of gas in the basement")
    out = str(tmp_path / "nope.pdf")
    assert report.to_pdf(inc, out) is False
    assert not os.path.exists(out)      # nothing half-written on the failure path


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
