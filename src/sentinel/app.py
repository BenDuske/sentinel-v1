# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""Sentinel v1 API + dashboard.  uvicorn sentinel.app:app --reload"""
import logging
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from pydantic import BaseModel

from . import config, store, ai, risk, report, policy
from .models import new_incident, SEVERITIES, STATUSES

app = FastAPI(title="Sentinel v1", version="0.1.0")
_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")

# Consent banner at startup — Sentinel is a web app, so the gate is a logged notice that
# points to CONSENT.md plus the in-page "I agree" acknowledgment in web/index.html.
logging.getLogger("uvicorn.error").info(policy.STARTUP_BANNER)


class IncidentIn(BaseModel):
    title: str
    description: str = ""
    analyze: bool = True  # auto-draft severity/summary/actions on create (HCAI: human edits after)


class IncidentPatch(BaseModel):
    severity: str | None = None
    summary: str | None = None
    recommended_actions: list[str] | None = None
    status: str | None = None


def _analyze(inc: dict) -> dict:
    """Grounded severity (rule layer ⨉ LLM) + summary + actions. Mutates and returns inc."""
    inc["severity"], inc["severity_rationale"] = risk.score(inc)
    inc["summary"] = ai.summarize(inc)
    inc["recommended_actions"] = ai.recommend(inc)
    inc["ai_generated"] = True
    inc["status"] = "reviewing"
    return inc


@app.get("/", response_class=HTMLResponse)
def dashboard():
    p = os.path.join(_WEB, "index.html")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("<h1>Sentinel v1</h1>")


@app.get("/healthz")
def healthz():
    """Health + LLM reachability. The app works offline; this just reports whether the LLM is live."""
    h = ai.health()
    return {
        "status": "ok",
        "llm": "reachable" if h["reachable"] else "offline",
        "llm_base": h["base"],
        "llm_model": h["model"],
        "mode": "llm-enriched" if h["reachable"] else "rule-layer (offline)",
    }


@app.post("/api/incidents")
def create(inc: IncidentIn):
    # Safety screen on user-supplied incident text. Only hard-blocks the zero-tolerance category
    # (sexualization of minors) by default — real incident reports (violence, injury, crime) are
    # the legitimate purpose and pass through. Strict adult-sexual screening is opt-in via
    # SENTINEL_STRICT_SCREEN.
    allowed, _category, message = policy.screen(f"{inc.title}\n{inc.description}")
    if not allowed:
        raise HTTPException(422, message)
    incident = new_incident(inc.title, inc.description)
    if inc.analyze:
        _analyze(incident)
    return store.save(incident)


@app.get("/api/incidents")
def list_incidents():
    return store.list_all()


@app.get("/api/incidents/{iid}")
def get_incident(iid: str):
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    return inc


@app.post("/api/incidents/{iid}/analyze")
def analyze(iid: str):
    """(Re)draft severity (grounded) + summary + actions. Human reviews/edits afterward (HCAI)."""
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    _analyze(inc)
    return store.save(inc)


@app.patch("/api/incidents/{iid}")
def edit(iid: str, patch: IncidentPatch):
    """Human-in-the-loop: correct any AI output before exporting."""
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    data = patch.model_dump(exclude_none=True)
    if "severity" in data and data["severity"] not in SEVERITIES:
        raise HTTPException(422, f"severity must be one of {SEVERITIES}")
    if "status" in data and data["status"] not in STATUSES:
        raise HTTPException(422, f"status must be one of {STATUSES}")
    inc.update(data)
    return store.save(inc)


@app.post("/api/incidents/{iid}/evidence")
async def upload_evidence(iid: str, file: UploadFile = File(...)):
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename or "evidence")
    dest = os.path.join(config.EVIDENCE_DIR, f"{iid}__{safe_name}")
    with open(dest, "wb") as f:
        f.write(await file.read())
    # `or []` (not setdefault) coerces a present-but-null evidence field: setdefault only fills an
    # ABSENT key, so a stored-but-null evidence (valid JSON, reachable via a hand-edit / partial
    # migration / foreign writer — the store column has no NOT NULL constraint) would return None
    # and make None.append(...) raise AttributeError → a 500 on this endpoint. The report export
    # coerces the same field the same way on read; keep the write path at parity.
    evidence = inc.get("evidence") or []
    evidence.append(safe_name)
    inc["evidence"] = evidence
    return store.save(inc)


@app.get("/api/incidents/{iid}/report.md", response_class=PlainTextResponse)
def report_md(iid: str):
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    return report.to_markdown(inc)


@app.get("/api/incidents/{iid}/report.pdf")
def report_pdf(iid: str):
    """Export PDF if reportlab is installed; otherwise 501 with a clear, graceful message."""
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(config.EVIDENCE_DIR, f"{iid}__report.pdf")
    if not report.to_pdf(inc, path):
        raise HTTPException(
            501, "PDF export unavailable (reportlab not installed). "
                 "Use the Markdown export, or `pip install reportlab`.")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"sentinel-incident-{iid}.pdf")
