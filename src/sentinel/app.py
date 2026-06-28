"""Sentinel v1 API + dashboard.  uvicorn sentinel.app:app --reload"""
import logging
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
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


class IncidentPatch(BaseModel):
    severity: str | None = None
    summary: str | None = None
    recommended_actions: list[str] | None = None
    status: str | None = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    p = os.path.join(_WEB, "index.html")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("<h1>Sentinel v1</h1>")


@app.post("/api/incidents")
def create(inc: IncidentIn):
    # Safety screen on user-supplied incident text. Only hard-blocks the zero-tolerance category
    # (sexualization of minors) by default — real incident reports (violence, injury, crime) are
    # the legitimate purpose and pass through. Strict adult-sexual screening is opt-in via
    # SENTINEL_STRICT_SCREEN.
    allowed, _category, message = policy.screen(f"{inc.title}\n{inc.description}")
    if not allowed:
        raise HTTPException(422, message)
    return store.save(new_incident(inc.title, inc.description))


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
    """AI drafts severity (grounded) + summary + actions. Human reviews/edits afterward (HCAI)."""
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    inc["severity"], inc["severity_rationale"] = risk.score(inc)
    inc["summary"] = ai.summarize(inc) or inc["description"]
    inc["recommended_actions"] = ai.recommend(inc)
    inc["ai_generated"] = True
    inc["status"] = "reviewing"
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
    dest = os.path.join(config.EVIDENCE_DIR, f"{iid}__{file.filename}")
    with open(dest, "wb") as f:
        f.write(await file.read())
    inc.setdefault("evidence", []).append(file.filename)
    return store.save(inc)


@app.get("/api/incidents/{iid}/report.md", response_class=PlainTextResponse)
def report_md(iid: str):
    inc = store.get(iid)
    if not inc:
        raise HTTPException(404, "not found")
    return report.to_markdown(inc)
