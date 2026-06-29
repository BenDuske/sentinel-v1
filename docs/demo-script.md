# Sentinel v1 — HackTitan demo script (< 3 minutes)

**Framing:** insurance / claims / incident documentation. Sentinel turns a messy incident note
into a **classified, summarized, action-ready, exportable** report — locally, so sensitive claim
data never leaves the building. The differentiator is **grounded, auditable severity**, not "the
AI said high."

Built by **Ben Duske**.

---

## 0. Setup (before you present)

```bash
pip install -r requirements.txt
# Offline is fine for the demo. To show the LLM-enriched path, also run a local
# OpenAI-compatible server (Ollama / llama.cpp) and point SENTINEL_LLM_BASE at it.
SENTINEL_ASSUME_CONSENT=1 uvicorn sentinel.app:app --app-dir src
# open http://127.0.0.1:8000
```

Have one browser tab on the dashboard. The header shows a live **LLM health dot**:
green = reachable, amber = offline (rule layer active).

---

## 1. The hook (20s)

> "When an incident happens — a flood, a fire, a break-in — the claim starts as a messy note.
> Sentinel turns that note into a defensible incident report in seconds, and it runs **entirely
> on your machine**, so the claim data never leaves the building. The key: the severity is
> **grounded and auditable**, not a black-box guess."

Point at the header: "Notice it works even with the LLM offline — watch."

## 2. Log an incident (40s)

Type into the form:

- **Title:** `Water leak in server room`
- **What happened:** `Burst pipe discovered overnight; ~2 inches of standing water around racks
  A3-A5. One technician slipped and twisted an ankle. UPS units at risk.`

Click **Log & analyze**. The detail panel fills instantly.

## 3. The differentiator — grounded severity (40s)

Point at the **"Why this severity"** box:

> "Severity is **HIGH** — and here's *why*. A deterministic rule layer matched
> `water/flood → high` on 'burst pipe, standing water' and `injury/medical → medium` on 'slip'.
> The LLM's judgment is reconciled with it and the **higher always wins** — a floor logic an
> insurer can audit line by line. Even with the LLM offline, the floor holds. No black box."

This is the line that wins technical + insurance judges: **explainable, defensible severity.**

## 4. Human-in-the-loop (30s)

> "The AI **drafts**; the human **decides**."

Edit the summary text and the recommended-actions box, change the severity dropdown if you like,
click **Save edits**. Show that the history table and report update.

## 5. Export the report (25s)

Click **Export Markdown**. A clean, professional incident report appears — severity + rationale,
summary, numbered next steps, evidence list, AI-assisted disclaimer footer. (PDF export is one
click too when `reportlab` is installed.)

> "That's a report you can attach to a claim file — generated, reviewed, and owned by a human."

## 6. Close (15s)

> "Local-first, so claim data stays private. Grounded, so severity is defensible. Human-in-the-
> loop, so a person owns every decision. And it degrades gracefully — it still works with no AI
> at all. That's Sentinel."

---

## Backup talking points (if asked)

- **Taxonomy:** 10 categories an insurer/facilities/security team cares about — injury/medical,
  fire/smoke, water/flood, electrical/power, gas/chemical, structural, security/intrusion, theft,
  outage, weather. See `src/sentinel/risk.py`.
- **Offline proof:** `docs/sample-run.md` is a real captured run with the LLM unreachable.
- **Safety/consent:** a real policy layer (`src/sentinel/policy.py`) — baseline safety prompt on
  every LLM call, a consent gate, and an input screen — not just a disclaimer.
- **Tests:** `python -m pytest -q` is fully keyless/offline (rule taxonomy, reconciliation, store
  CRUD, report export, full HTTP flow, and the safety layer).
