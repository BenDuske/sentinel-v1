# Sentinel v1 — Local AI Risk & Incident Intelligence System

Turn a raw incident — a note, an alert, an uploaded log — into a **classified, summarized,
action-ready, exportable** report. **Local-first**: the AI runs on your machine (no cloud
dependency required), so sensitive incident data never has to leave the building.

> Built by **Ben Duske** (student / independent developer) for hackathon competition; later
> expanded under a commercial offering. © 2026 Ben Duske. MIT-licensed.

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

## Safety, privacy & legal

This is a public release, so it ships with a real policy layer — not just a disclaimer:

- **Baseline safety policy** (`sentinel.policy`) is prepended to every LLM system prompt and
  cannot be disabled: lawful use only (U.S. federal + your state/local law), no sexual/explicit
  content, **zero tolerance** for sexualizing minors (hard-blocked at the input screen).
  Documenting real incidents — violence, injury, theft, fire, crime — is the legitimate purpose
  and is fully supported; the screen does **not** block legitimate incident reports.
- **Consent gate** — the web dashboard requires an "I agree" acknowledgment (License/Terms/AUP/
  Privacy/Disclaimer) before you can log incidents, and the server logs a notice at startup.
  See [`CONSENT.md`](CONSENT.md). For CLI/automation, `sentinel.policy.ensure_consent()` records
  acceptance locally; set `SENTINEL_ASSUME_CONSENT=1` for headless/CI.
- **Ethics onboarding** — an insurer/facilities/security org can layer its own mission, voice,
  and boundaries on top of the baseline via `ethics.yaml`
  ([`docs/ETHICS_ONBOARDING.md`](docs/ETHICS_ONBOARDING.md)); config can only tighten the
  baseline, never loosen it.
- **AI-output disclaimer** — severity, summaries, and actions are **decision-support, not an
  underwriting or safety determination** and not professional advice; a human reviews
  consequential decisions. [`docs/legal/DISCLAIMER.md`](docs/legal/DISCLAIMER.md).
- **Local-first privacy** — the author runs no servers and collects nothing; incident data lives
  in your local SQLite store and evidence in a local uploads dir; the default LLM is local
  Ollama. [`docs/legal/PRIVACY_POLICY.md`](docs/legal/PRIVACY_POLICY.md).
- Full policy set: [`docs/legal/`](docs/legal/) · vulnerabilities: [`SECURITY.md`](SECURITY.md)
  · conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Bug reports use the issue templates.

Optional env: `SENTINEL_ETHICS_FILE`, `SENTINEL_STRICT_SCREEN` (opt-in adult-sexual screening,
default off), `SENTINEL_ASSUME_CONSENT`.

> Legal docs are **DRAFT — pending attorney review**; they are templates, not legal advice.

## License
MIT (see `LICENSE`). Third-party components in `THIRD_PARTY_NOTICES.md`.
