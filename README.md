# Sentinel v1 — Local AI Risk & Incident Intelligence System

Turn a raw incident — a note, an alert, an uploaded log — into a **classified, summarized,
action-ready, exportable** report. **Local-first**: the AI runs on your machine (no cloud
dependency required), so sensitive incident data never has to leave the building.

> Built by **Ben Duske** (student / independent developer) for hackathon competition; later
> expanded under Aetherion Technology. © 2026 Digital Real-Estate Frontier, LLC. MIT-licensed.

## The flow (MVP)
1. **Enter or upload** an incident (text, plus optional evidence files).
2. Sentinel **classifies severity** — low / medium / high / critical.
3. Sentinel writes a **clean incident summary**.
4. Sentinel **recommends next steps**.
5. **Export** a professional report (Markdown / PDF) for insurance, facilities, or security use.
6. **Dashboard** shows incident history — severity, status, timestamps.

## Two design principles that set it apart

**Grounded risk scoring (not "the AI said high").** Severity is decided by a deterministic
**rule layer** (keywords/thresholds → severity floor) *reconciled with* the LLM's judgment.
The result carries a **rationale** showing both, so the score is *defensible* — what insurers
and technical judges actually want.

**Human-Centered AI (HCAI).** The AI drafts; the human decides. Every AI output (severity,
summary, actions) is editable and shown with its reasoning before anything is exported.
Human-in-the-loop at the decision points, transparency by default — no black-box automation.

## Architecture (local-first)

```
incident in ──▶ FastAPI backend
                  ├─ risk.py    rule layer ⨉ LLM  → severity + rationale   (grounded)
                  ├─ ai.py      summary + recommended actions               (local LLM)
                  ├─ store.py   SQLite incident history
                  └─ report.py  Markdown / PDF export
                  ▲
            web dashboard (review / edit / export)         LLM = local Ollama (Aegis brain),
                                                            swappable to any OpenAI-compatible endpoint
```

The LLM defaults to your **local Ollama** (`qwen3` / the Aegis model) — fully offline. It's
OpenAI-compatible, so it can also point at a cloud endpoint if you ever want to.

## Run

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn sentinel.app:app --reload      # dashboard at http://127.0.0.1:8000
```

## Two hackathons, one build
- **HackTitan** (State Farm-sponsored) → insurance / claim reduction / incident documentation /
  home & small-business safety.
- **Dev Clash** → AI · cybersecurity · DevOps · automation · local-first incident intelligence.

Same core, two pitches. See `docs/` for each framing.

## License
MIT (see `LICENSE`). Third-party components in `THIRD_PARTY_NOTICES.md`.
