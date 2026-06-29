"""End-to-end HTTP flow via TestClient, LLM forced OFFLINE (deterministic, no network).

create -> (auto-analyze) -> list -> get -> edit (human-in-the-loop) -> evidence -> export.
The LLM chat() is monkeypatched to return "" so the deterministic rule layer + fallbacks run,
proving the whole flow works with no Ollama/endpoint reachable.
"""
import importlib
import io

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "sentinel.db"))
    monkeypatch.setenv("SENTINEL_EVIDENCE_DIR", str(tmp_path / "evidence"))
    from sentinel import config, store, policy, ai, risk_fallback, risk, report, app as appmod
    importlib.reload(config)
    importlib.reload(store)
    importlib.reload(policy)
    importlib.reload(risk_fallback)
    importlib.reload(ai)
    importlib.reload(risk)
    importlib.reload(report)
    importlib.reload(appmod)
    # Force LLM offline for every test using this client.
    monkeypatch.setattr(appmod.ai, "chat", lambda *a, **k: "")
    monkeypatch.setattr(appmod.risk.ai, "llm_severity", lambda text: "")
    return fastapi_testclient.TestClient(appmod.app)


def test_healthz_reports_offline_shape(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"] in ("reachable", "offline")
    assert "llm_model" in body


def test_full_flow_offline(client):
    # 1. create (auto-analyze on)
    r = client.post("/api/incidents", json={
        "title": "Server room flood",
        "description": "Burst pipe overnight, water damage to two racks; one tech slipped."})
    assert r.status_code == 200, r.text
    inc = r.json()
    iid = inc["id"]
    # grounded severity from rule layer even with NO LLM
    assert inc["severity"] in ("high", "critical")
    assert "Rule layer →" in inc["severity_rationale"]
    assert "offline" in inc["severity_rationale"].lower()
    # deterministic fallback summary + actions present
    assert inc["summary"]
    assert len(inc["recommended_actions"]) >= 3
    assert inc["ai_generated"] is True

    # 2. list
    r = client.get("/api/incidents")
    assert r.status_code == 200
    assert any(i["id"] == iid for i in r.json())

    # 3. get one
    r = client.get(f"/api/incidents/{iid}")
    assert r.status_code == 200 and r.json()["id"] == iid

    # 4. human edits (HCAI): override severity + summary + actions
    r = client.patch(f"/api/incidents/{iid}", json={
        "severity": "critical", "summary": "Edited by reviewer.",
        "recommended_actions": ["Shut off water", "Call facilities"], "status": "reviewing"})
    assert r.status_code == 200
    edited = r.json()
    assert edited["severity"] == "critical"
    assert edited["summary"] == "Edited by reviewer."

    # 5. evidence upload (stored + referenced)
    r = client.post(f"/api/incidents/{iid}/evidence",
                    files={"file": ("damage.txt", io.BytesIO(b"photo evidence"), "text/plain")})
    assert r.status_code == 200
    assert "damage.txt" in r.json()["evidence"]

    # 6. export markdown — reflects edits + evidence
    r = client.get(f"/api/incidents/{iid}/report.md")
    assert r.status_code == 200
    md = r.text
    assert "Server room flood" in md
    assert "CRITICAL" in md
    assert "Edited by reviewer." in md
    assert "damage.txt" in md


def test_create_without_analyze_leaves_unscored(client):
    r = client.post("/api/incidents",
                    json={"title": "Note", "description": "routine check", "analyze": False})
    assert r.status_code == 200
    assert r.json()["severity"] == "unscored"
    assert r.json()["ai_generated"] is False


def test_invalid_severity_patch_rejected(client):
    iid = client.post("/api/incidents", json={"title": "x", "analyze": False}).json()["id"]
    r = client.patch(f"/api/incidents/{iid}", json={"severity": "apocalyptic"})
    assert r.status_code == 422


def test_get_missing_incident_404(client):
    assert client.get("/api/incidents/nope").status_code == 404


def test_report_pdf_graceful_without_reportlab(client):
    iid = client.post("/api/incidents", json={"title": "x", "analyze": False}).json()["id"]
    r = client.get(f"/api/incidents/{iid}/report.pdf")
    # 200 if reportlab installed, 501 graceful fallback if not — both acceptable.
    assert r.status_code in (200, 501)
