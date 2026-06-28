"""LLM helpers — local-first via any OpenAI-compatible endpoint (default: local Ollama).

Stdlib only. Falls back gracefully (empty/heuristic) if the LLM is unreachable, so the app
never hard-fails offline — the rule layer in risk.py still produces a defensible severity.
"""
import json
import urllib.request
import urllib.error
from . import config


def chat(messages, temperature: float = 0.3, max_tokens: int = 700) -> str:
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
    except (urllib.error.URLError, KeyError, TimeoutError, OSError):
        return ""  # offline / unreachable → caller degrades gracefully


def summarize(incident: dict) -> str:
    text = f"Title: {incident['title']}\nDetails: {incident['description']}"
    out = chat([
        {"role": "system", "content": "You write concise, professional incident summaries "
         "for an insurance/security/facilities report. 3-5 sentences, factual, no speculation."},
        {"role": "user", "content": text},
    ])
    return out


def recommend(incident: dict) -> list:
    text = (f"Incident: {incident['title']}\nDetails: {incident['description']}\n"
            f"Severity: {incident.get('severity')}")
    out = chat([
        {"role": "system", "content": "Give 3-5 concrete recommended next steps for this "
         "incident as a JSON array of short strings. JSON only."},
        {"role": "user", "content": text},
    ], temperature=0.2)
    try:
        arr = json.loads(out[out.find("["): out.rfind("]") + 1])
        return [str(x) for x in arr][:5]
    except Exception:
        return [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:5]


def llm_severity(text: str) -> str:
    out = chat([
        {"role": "system", "content": "Classify the incident severity as exactly one of: "
         "low, medium, high, critical. Reply with only that one word."},
        {"role": "user", "content": text},
    ], temperature=0.0, max_tokens=4).lower()
    for s in ("critical", "high", "medium", "low"):
        if s in out:
            return s
    return ""
