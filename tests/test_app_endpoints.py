"""HTTP-layer robustness for app.py — the endpoint branches the flow test doesn't exercise.

Pins that every /api/incidents/{iid} endpoint returns a clean 404 (not a 500) for a missing
incident, that the dashboard never 500s (serves the page or a minimal fallback), that the
/analyze endpoint (re)scores, that invalid status is rejected 422, and that PDF export degrades
gracefully to 501 when the renderer reports it couldn't write. LLM forced offline — no network.
"""
import importlib
import io

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def appmod(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "sentinel.db"))
    monkeypatch.setenv("SENTINEL_EVIDENCE_DIR", str(tmp_path / "evidence"))
    from sentinel import config, store, policy, ai, risk_fallback, risk, report, app as appmod
    for m in (config, store, policy, risk_fallback, ai, risk, report, appmod):
        importlib.reload(m)
    # Force LLM offline: the deterministic rule layer + fallbacks run, no Ollama/endpoint needed.
    monkeypatch.setattr(appmod.ai, "chat", lambda *a, **k: "")
    monkeypatch.setattr(appmod.risk.ai, "llm_severity", lambda text: "")
    return appmod


@pytest.fixture()
def client(appmod):
    return fastapi_testclient.TestClient(appmod.app)


def _new_unscored(client) -> str:
    """Create an incident without auto-analyze; return its id."""
    return client.post("/api/incidents",
                       json={"title": "x", "analyze": False}).json()["id"]


# ---- dashboard: serves the page, and never 500s if the file is missing --------------------

def test_dashboard_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers["content-type"].lower()


def test_dashboard_falls_back_when_index_missing(appmod, monkeypatch):
    # If web/index.html isn't shipped, the dashboard still returns a minimal 200 page, not a 500.
    monkeypatch.setattr(appmod.os.path, "exists", lambda p: False)
    r = fastapi_testclient.TestClient(appmod.app).get("/")
    assert r.status_code == 200
    assert "Sentinel v1" in r.text


# ---- /analyze: (re)scores an existing incident, 404 on a missing one ----------------------

def test_analyze_endpoint_scores_existing(client):
    iid = _new_unscored(client)
    assert client.get(f"/api/incidents/{iid}").json()["severity"] == "unscored"
    r = client.post(f"/api/incidents/{iid}/analyze")
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] != "unscored"      # grounded rule layer scored it, offline
    assert body["ai_generated"] is True
    assert body["status"] == "reviewing"


def test_analyze_missing_404(client):
    assert client.post("/api/incidents/nope/analyze").status_code == 404


# ---- patch: 404 on missing, 422 on an out-of-vocabulary status ----------------------------

def test_patch_missing_404(client):
    assert client.patch("/api/incidents/nope", json={"summary": "x"}).status_code == 404


def test_invalid_status_patch_rejected(client):
    iid = _new_unscored(client)
    r = client.patch(f"/api/incidents/{iid}", json={"status": "teleported"})
    assert r.status_code == 422


# ---- the remaining {iid} endpoints all 404 cleanly on a missing incident ------------------

def test_upload_evidence_missing_404(client):
    r = client.post("/api/incidents/nope/evidence",
                    files={"file": ("e.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 404


def test_report_md_missing_404(client):
    assert client.get("/api/incidents/nope/report.md").status_code == 404


def test_report_pdf_missing_404(client):
    assert client.get("/api/incidents/nope/report.pdf").status_code == 404


# ---- PDF export degrades to a clear 501 when the renderer can't write ----------------------

def test_report_pdf_501_when_renderer_unavailable(client, appmod, monkeypatch):
    # Deterministic regardless of whether reportlab is installed: force to_pdf to report failure
    # and assert the caller surfaces a graceful 501 with an actionable message (never a 500).
    iid = _new_unscored(client)
    monkeypatch.setattr(appmod.report, "to_pdf", lambda inc, path: False)
    r = client.get(f"/api/incidents/{iid}/report.pdf")
    assert r.status_code == 501
    assert "reportlab" in r.json()["detail"].lower()
