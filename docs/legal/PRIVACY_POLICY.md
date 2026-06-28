# Privacy Policy

> **DRAFT — pending attorney review.** Provided in good faith for the open-source release;
> **not legal advice.** A commercial operator must have counsel review and adapt this before
> relying on it.

**Software:** Sentinel v1 ("the Software").
**Last updated:** 2026-06-28.

This policy describes how the **open-source Software** handles data. If you obtained the
Software from a third party who runs it as a hosted service, **their** privacy policy governs
that service — ask them.

## 1. The short version
The Software is **local-first**. Incident records live in a **SQLite database on your own
machine**; uploaded evidence files live in a local uploads/evidence directory on your machine.
The author of the Software operates **no servers**, collects **no telemetry**, and never
receives your data. The only data that leaves your machine is what you choose to send to the
LLM endpoint you configure (by default, a **local Ollama** model — fully offline) to get
severity scoring, summaries, and recommended actions.

## 2. What data is involved
- **Your incident data.** Titles, descriptions, severities, summaries, recommended actions, and
  status — written to a local **SQLite** store (`SENTINEL_DB`, default `./sentinel.db`). It
  stays on your machine.
- **Uploaded evidence files.** Any files you attach to an incident are written to your local
  evidence directory (`SENTINEL_EVIDENCE_DIR`, default `./evidence`). They stay on your machine.
- **Your API key (if any).** Read from your environment / `.env`. Used only to authenticate to
  the LLM endpoint you configure. The local Ollama default needs no key. The key is never
  transmitted anywhere else and is not logged.
- **Network calls to your LLM endpoint.** To analyze an incident, the Software sends the
  incident text to the endpoint's API and receives a completion. By **default that endpoint is
  a local Ollama model on your own machine, so nothing leaves it.** If you reconfigure it to a
  remote OpenAI-compatible endpoint, that provider processes the text under **its own** terms.

## 3. What the author collects
**Nothing.** No analytics, no telemetry, no crash reporting, no "phone home." You can verify
this — the source is open and dependency-light (standard library plus FastAPI/uvicorn/pydantic
for the local web service).

## 4. Third-party processors you choose
- **Local Ollama (default)** — runs on your machine; no data leaves it.
- **Other OpenAI-compatible endpoint (optional)** — if you point the Software at a remote
  endpoint, your incident text is sent there and **that** provider's privacy and data-usage
  terms apply. Review them before enabling a remote endpoint for sensitive incident data.

## 5. Your control over your data
Because storage is local files you own:
- **Access / export:** read the SQLite store directly, or use the Markdown/PDF report export.
- **Delete:** delete incident rows from the SQLite store, delete files from the evidence
  directory, or delete the whole store/directory. Deletion is immediate and permanent; there is
  no backup the author holds.
- **Portability:** the store is standard SQLite and reports export to Markdown — copy anywhere.

## 6. Children's privacy
The Software is **not directed to children under 13** (or the minimum age of digital consent
in your jurisdiction). Do not use it to knowingly collect data from children, and never to
process content sexualizing minors (see `ACCEPTABLE_USE_POLICY.md`).

## 7. Security
See `SECURITY.md`. In brief: keep your API key, the SQLite incident store, and the uploads/
evidence directory protected with your operating system's file permissions and disk
encryption; the Software does not add its own encryption layer to the local store.

## 8. Changes
Material changes to this policy will be noted in the repository's history. The "Last updated"
date above reflects the current version.

## 9. Contact
Open-source inquiries: via the GitHub repository's issues. Author: **Ben Duske**.
