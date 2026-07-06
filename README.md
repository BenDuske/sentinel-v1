# Sentinel v1 — Local AI Risk & Incident Intelligence System

Turn a raw incident — a note, an alert, an uploaded log — into a **classified, summarized,
action-ready, exportable** report. **Local-first**: the AI runs on your machine (no cloud
dependency required), so sensitive incident data never has to leave the building.

> Built by **Ben Duske** (student / independent developer) for hackathon competition; later
> expanded under a commercial offering. © 2026 Ben Duske. MIT-licensed.

## The flow (MVP)
1. **Enter or upload** an incident (text, plus an optional evidence file).
2. Sentinel **classifies severity** — low / medium / high / critical — via a **grounded rule
   layer reconciled with the LLM** (the higher of the two wins), with a rationale showing both.
3. Sentinel writes a **clean incident summary**.
4. Sentinel **recommends next steps**.
5. A human **reviews and edits** any AI field before exporting (human-in-the-loop).
6. **Export** a professional report — **Markdown always**, **PDF when `reportlab` is installed**
   (graceful fallback otherwise) — for insurance, facilities, or security use.
7. **Dashboard** shows incident history — severity badge, status, timestamps.

**Works fully offline.** The LLM *enriches* but is never required: with no Ollama/endpoint
reachable, the deterministic rule layer still produces a defensible severity and the summary +
recommended actions fall back to category-aware deterministic output. `GET /healthz` reports
`llm: reachable | offline`.

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

# src/ layout: tell uvicorn where the package lives with --app-dir
uvicorn sentinel.app:app --app-dir src --reload     # dashboard at http://127.0.0.1:8000
```

Headless/CI or a quick demo without the consent prompt: set `SENTINEL_ASSUME_CONSENT=1`.
Optional PDF export: `pip install reportlab` (or `pip install -r requirements-pdf.txt`); Markdown
export always works and `/report.pdf` returns a graceful 501 until reportlab is installed.

**Seed demo data (offline):** populate the store with five varied, realistic incidents — kitchen
fire, server-room water leak, slip-and-fall injury, overnight break-in, branch power outage — for
a demo or a clean screenshot. Runs fully offline (grounded rule layer + deterministic fallbacks;
no Ollama, no network, no key):

```bash
python scripts/seed_demo.py            # seed into ./sentinel.db (or $SENTINEL_DB)
python scripts/seed_demo.py --reset    # wipe existing incidents first
```

**Verify it's running:**

```bash
curl http://127.0.0.1:8000/healthz      # {"status":"ok","llm":"reachable|offline",...}
```

**Tests** (fully keyless/offline — no network or LLM needed):

```bash
pip install -r requirements-dev.txt   # pytest + httpx (test runner; not needed to run the app)
python -m pytest -q
```

### HTTP API

| Method & path | Purpose |
|---|---|
| `GET  /` | Dashboard (single-file HTML) |
| `GET  /healthz` | Health + LLM reachability |
| `POST /api/incidents` | Create incident (auto-drafts severity/summary/actions; `analyze:false` to skip) |
| `GET  /api/incidents` | List incidents (newest first) |
| `GET  /api/incidents/{id}` | Get one |
| `POST /api/incidents/{id}/analyze` | Re-draft severity + summary + actions |
| `PATCH /api/incidents/{id}` | Human edit: severity / summary / actions / status |
| `POST /api/incidents/{id}/evidence` | Upload an evidence file (stored locally, referenced in SQLite) |
| `GET  /api/incidents/{id}/report.md` | Markdown report |
| `GET  /api/incidents/{id}/report.pdf` | PDF report (501 with a clear message if `reportlab` absent) |

See `docs/demo-script.md` for a <3-min walkthrough and `docs/sample-run.md` for a real
captured offline run.

## HackTitan (State Farm) — the insurance angle

State Farm's business is paying out — and preventing — claims. Sentinel attacks both ends:

- **Faster, cleaner claim documentation.** A messy incident note becomes a structured,
  exportable report (severity + rationale, summary, next steps, evidence) in seconds — the kind
  of artifact that goes straight into a claim file.
- **Defensible, grounded severity.** Severity isn't "the AI said high." A transparent rule layer
  sets an auditable floor from an insurer-relevant taxonomy (injury, fire, water, electrical, gas,
  structural, intrusion, theft, outage, weather), reconciled with the LLM — so an adjuster can see
  *why*, line by line. Auditability is what an insurer (and a technical judge) actually trusts.
- **Loss prevention for homes & small businesses.** The recommended next steps are concrete,
  category-aware mitigation actions — contain the hazard, preserve evidence, notify the right
  owner — which is exactly the early action that keeps a small loss from becoming a large claim.
- **Privacy by construction.** It runs local-first, so sensitive incident and claim data never
  has to leave the policyholder's or agent's machine.

> Honest scope: this is an MVP. It makes no claim of measured loss-ratio impact — the value it
> demonstrates is grounded, reviewable severity and clean, exportable incident documentation.

Sentinel was also built with a second framing — **Dev Clash** (AI · cybersecurity · DevOps ·
automation · local-first incident intelligence). Same core, two pitches; see `docs/` for each.

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
