# Sentinel v1 — © 2026 Ben Duske. Licensed under the MIT License (see LICENSE).
"""LLM helpers — local-first via any OpenAI-compatible endpoint (default: local Ollama).

Stdlib only. The LLM ENRICHES but is never a hard dependency: every function degrades to a
deterministic, useful fallback when the endpoint is unreachable, so the app works fully offline
(critical for the demo and for CI). The rule layer in risk.py still produces a defensible
severity; the helpers here still produce a clean summary and concrete recommended actions.
"""
import json
import urllib.request
import urllib.error
from . import config
from . import policy
from . import risk_fallback


def _with_safety(messages):
    """Prepend (or fold into) the safety + org-ethics preamble on the system message.

    Precedence: BASELINE_POLICY -> org ethics -> the task instructions the caller wrote. This
    is what makes the safety policy steer every local LLM call.
    """
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            msgs[i] = {"role": "system", "content": policy.system_preamble(m.get("content", ""))}
            return msgs
    return [{"role": "system", "content": policy.system_preamble("")}] + msgs


def chat(messages, temperature: float = 0.3, max_tokens: int = 700) -> str:
    messages = _with_safety(messages)
    body = json.dumps({
        "model": config.LLM_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.LLM_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_KEY}"
    req = urllib.request.Request(config.LLM_BASE.rstrip("/") + "/chat/completions",
                                 data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (urllib.error.URLError, KeyError, TimeoutError, OSError, ValueError):
        return ""  # offline / unreachable → caller degrades gracefully


def health() -> dict:
    """Lightweight reachability probe for /healthz. Never raises.

    Returns {"reachable": bool, "base": str, "model": str}. A short GET to /models on the
    OpenAI-compatible endpoint is enough to tell whether an LLM is configured and live.
    """
    url = config.LLM_BASE.rstrip("/") + "/models"
    headers = {}
    if config.LLM_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_KEY}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    reachable = False
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            reachable = 200 <= r.status < 300
    except Exception:
        reachable = False
    return {"reachable": reachable, "base": config.LLM_BASE, "model": config.LLM_MODEL}


def summarize(incident: dict) -> str:
    """LLM summary when reachable, else a deterministic, professional fallback summary."""
    text = f"Title: {incident['title']}\nDetails: {incident['description']}"
    out = chat([
        {"role": "system", "content": "You write concise, professional incident summaries "
         "for an insurance/security/facilities report. 3-5 sentences, factual, no speculation. "
         "Output the summary text only, no preamble."},
        {"role": "user", "content": text},
    ])
    out = _clean(out)
    return out or risk_fallback.fallback_summary(incident)


def recommend(incident: dict) -> list:
    """LLM next-steps when reachable, else a deterministic, category-aware action list."""
    text = (f"Incident: {incident['title']}\nDetails: {incident['description']}\n"
            f"Severity: {incident.get('severity')}")
    out = chat([
        {"role": "system", "content": "Give 3-5 concrete recommended next steps for this "
         "incident as a JSON array of short strings. JSON only, no prose."},
        {"role": "user", "content": text},
    ], temperature=0.2)
    out = _strip_think(out)
    actions = []
    try:
        arr = json.loads(out[out.find("["): out.rfind("]") + 1])
        actions = [str(x).strip() for x in arr if str(x).strip()][:5]
    except Exception:
        actions = [l.strip("-•* ").strip() for l in out.splitlines() if l.strip()][:5]
    return actions or risk_fallback.fallback_actions(incident)


def llm_severity(text: str) -> str:
    out = chat([
        {"role": "system", "content": "Classify the incident severity as exactly one of: "
         "low, medium, high, critical. Reply with only that one word."},
        {"role": "user", "content": text},
    ], temperature=0.0, max_tokens=16)
    out = _strip_think(out).lower()
    for s in ("critical", "high", "medium", "low"):
        if s in out:
            return s
    return ""


def _strip_think(s: str) -> str:
    """Remove <think>...</think> reasoning blocks some local models emit before the answer."""
    if not s:
        return ""
    low = s.lower()
    if "</think>" in low:
        s = s[low.rindex("</think>") + len("</think>"):]
    return s.strip()


def _clean(s: str) -> str:
    return _strip_think(s).strip().strip('"').strip()
