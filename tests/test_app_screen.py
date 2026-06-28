"""App-level safety wiring — uses FastAPI TestClient, no live LLM/network.

The /analyze path (which would call the LLM) is not exercised here; chat() degrades to "" when
the endpoint is unreachable, but these tests deliberately only touch the create() entry point so
they need no Ollama and no network.
"""
import os
import sys

import pytest

# Skip cleanly if optional web deps aren't installed (keeps the core suite green either way).
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate the SQLite store + evidence dir per test.
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "sentinel.db"))
    monkeypatch.setenv("SENTINEL_EVIDENCE_DIR", str(tmp_path / "evidence"))
    # Reload config + dependents so the env vars take effect.
    import importlib
    from sentinel import config, store, policy, app as appmod
    importlib.reload(config)
    importlib.reload(store)
    importlib.reload(policy)
    importlib.reload(appmod)
    return fastapi_testclient.TestClient(appmod.app)


def test_create_allows_normal_incident(client):
    r = client.post("/api/incidents",
                    json={"title": "Warehouse fire", "description": "fire broke out, two injured"})
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Warehouse fire"


def test_create_hard_blocks_minor_safety(client):
    r = client.post("/api/incidents",
                    json={"title": "report", "description": "sexual content involving a 10 year old child"})
    assert r.status_code == 422
