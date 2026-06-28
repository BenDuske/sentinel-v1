# Security Policy

## Reporting a vulnerability
If you discover a security vulnerability in **Sentinel v1**, please report it
**privately** — do not open a public issue for an exploitable flaw.

- **Preferred:** open a [GitHub Security Advisory](https://github.com/BenDuske/sentinel-v1/security/advisories/new)
  (private disclosure to the maintainer).
- Include: affected version/commit, a description, reproduction steps, and impact.

Please give a reasonable window for a fix before any public disclosure. This is a
volunteer-maintained open-source project; we will acknowledge reports as promptly as we can.

## Scope
This Software is **local-first**: it runs on your machine, stores incidents in a local SQLite
database, keeps uploaded evidence in a local directory, and (by default) calls a **local**
Ollama LLM. The most relevant security considerations are therefore on the **operator** side:

- **API keys** (only needed for a remote endpoint) live in your environment / `.env`. Keep that
  file out of version control (`.gitignore` already excludes `.env`) and protect it with OS file
  permissions.
- **Incident store** (`SENTINEL_DB`, default `./sentinel.db`) is a plain SQLite file on disk. It
  may contain sensitive incident details. Protect it with filesystem permissions and, ideally,
  disk encryption.
- **Evidence uploads** (`SENTINEL_EVIDENCE_DIR`, default `./evidence`) are plain files on disk —
  protect them the same way.
- **Network egress** goes only to the LLM endpoint you configure (none, by default, with a local
  Ollama model). Verify you trust any remote endpoint before pointing the Software at it.

## What is in scope for a report
- Code paths that could leak your API key, incident data, or evidence to an unintended destination.
- Injection, path-traversal (including evidence-upload filename handling), or deserialization
  issues in the API, store, or web UI.
- Dependency vulnerabilities (FastAPI/uvicorn/pydantic/python-multipart and any you add).

## What is out of scope
- The behavior or content of the third-party LLM you configure.
- Misconfiguration of your own OS permissions or your own deployment (e.g. binding the server to
  a public interface without authentication in front of it).
- The inherent fallibility of AI output (see `docs/legal/DISCLAIMER.md`).
