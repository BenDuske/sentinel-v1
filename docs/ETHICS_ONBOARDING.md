# Ethics Onboarding — Make Sentinel Yours

Sentinel v1 ships with a **baseline safety policy** (no illegal use, no sexual/abusive content,
AI output stays assistive — see `docs/legal/ACCEPTABLE_USE_POLICY.md`) that is always on and
cannot be disabled.

On top of that baseline, an organization deploying Sentinel can **layer in its own ethics,
mission, and boundaries** so the AI's summaries and recommendations speak in line with the
organization. This is the "onboarding" step — useful for an insurer, facilities team, or
security operation that wants consistent, disclosed, on-brand output.

## How it works
Create an `ethics.yaml` (copy `ethics.example.yaml`) describing your organization. At startup
Sentinel loads it and **prepends it to the LLM system prompt**, after the baseline safety policy
and before the task instructions. Order of precedence (highest first):

1. **Baseline safety policy** (built in — cannot be overridden by config)
2. **Your organization's ethics & boundaries** (`ethics.yaml`)
3. **The task instructions** (summary / recommended-actions / severity prompts)

So your mission and tone shape every answer, but **nothing in your config can loosen the
baseline safety rules** — config can only make Sentinel *more* restrictive or more specific,
never less safe.

## What you can set
- `organization` — name, mission, values, voice/tone.
- `goals` — what Sentinel should help users accomplish.
- `boundaries` — topics to avoid, disclaimers to always include, escalation rules ("for coverage
  determinations, direct to a human adjuster").
- `required_disclosures` — text Sentinel must include in relevant answers. For a regulated
  insurance/facilities deployment, e.g. *"AI risk scoring is decision-support, not an
  underwriting or safety determination."*

## Example
See [`../ethics.example.yaml`](../ethics.example.yaml). Point Sentinel at your file with:

```bash
# Windows (PowerShell):  $env:SENTINEL_ETHICS_FILE = ".\ethics.yaml"
# macOS / Linux:
export SENTINEL_ETHICS_FILE=./ethics.yaml
```

If no file is set, only the baseline safety policy applies.

## A note on limits
This steers a probabilistic model; it is **strong guidance, not a hard guarantee**. Keep a
human in the loop for high-stakes use, and keep your `required_disclosures` legally reviewed.
