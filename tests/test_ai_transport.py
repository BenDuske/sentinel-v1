"""AI transport layer — the actual HTTP call to the LLM endpoint, with no network.

The parser tests (test_ai_parsers.py) covered what ai.py does with model *output*, but mocked
`ai.chat` itself, so the transport was never exercised: request construction, the safety-preamble
injection, the Bearer-auth header, and — most importantly — graceful degradation to "" when the
endpoint is unreachable. Those are the local-first + safety guarantees Sentinel leans on, so they
deserve direct tests.

We monkeypatch `urllib.request.urlopen` (imported as `ai.urllib.request.urlopen`) with a fake that
captures the outgoing Request and returns canned bytes, so nothing hits the network. The captured
Request lets us assert the two invariants that matter:
  * the BASELINE_POLICY safety preamble is folded into the system message of EVERY chat() call, and
  * a network/parse failure returns "" (caller then degrades) rather than raising.
"""
import json
import urllib.error

import pytest

from sentinel import ai, config, policy


class _FakeResp:
    """Minimal stand-in for the urlopen context manager."""
    def __init__(self, body=b"", status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(captured, body=b"", status=200, raise_exc=None):
    """Build a urlopen replacement that records the Request and returns/raises as configured."""
    def _fake(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        if raise_exc is not None:
            raise raise_exc
        return _FakeResp(body, status)
    return _fake


def _completion(content):
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


# ---- _with_safety -------------------------------------------------------------------------------

def test_with_safety_folds_into_existing_system_message():
    msgs = ai._with_safety([
        {"role": "system", "content": "You are a summarizer."},
        {"role": "user", "content": "hi"},
    ])
    # Same shape, system stays first, and the baseline policy now precedes the caller's persona.
    assert msgs[0]["role"] == "system"
    assert policy.BASELINE_POLICY in msgs[0]["content"]
    assert "You are a summarizer." in msgs[0]["content"]
    assert msgs[0]["content"].index(policy.BASELINE_POLICY) < msgs[0]["content"].index("You are a summarizer.")
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_with_safety_prepends_system_when_none_present():
    # Line 28 path: a message list with no system role still gets the safety preamble, up front.
    msgs = ai._with_safety([{"role": "user", "content": "hi"}])
    assert msgs[0]["role"] == "system"
    assert policy.BASELINE_POLICY in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


# ---- chat: success path -------------------------------------------------------------------------

def test_chat_returns_stripped_content_and_injects_safety(monkeypatch):
    captured = {}
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, _completion("  answer  ")))

    out = ai.chat([{"role": "user", "content": "classify this"}])
    assert out == "answer"

    body = json.loads(captured["req"].data.decode("utf-8"))
    assert body["model"] == config.LLM_MODEL
    assert body["stream"] is False
    # The safety preamble is injected on EVERY call, even one the caller sent with no system msg.
    system_msgs = [m for m in body["messages"] if m["role"] == "system"]
    assert system_msgs and policy.BASELINE_POLICY in system_msgs[0]["content"]
    # Endpoint + method are the OpenAI-compatible chat completions POST.
    assert captured["req"].get_full_url().endswith("/chat/completions")
    assert captured["req"].get_method() == "POST"


def test_chat_passes_temperature_and_max_tokens(monkeypatch):
    captured = {}
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, _completion("x")))
    ai.chat([{"role": "user", "content": "q"}], temperature=0.0, max_tokens=16)
    body = json.loads(captured["req"].data.decode("utf-8"))
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 16


def test_chat_sets_bearer_auth_header_when_key_present(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "LLM_KEY", "secret-token")
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, _completion("x")))
    ai.chat([{"role": "user", "content": "q"}])
    # urllib normalizes header keys to Capitalized form.
    assert captured["req"].get_header("Authorization") == "Bearer secret-token"


def test_chat_no_auth_header_when_key_absent(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "LLM_KEY", "")
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, _completion("x")))
    ai.chat([{"role": "user", "content": "q"}])
    assert captured["req"].get_header("Authorization") is None


# ---- chat: graceful degradation -----------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    urllib.error.URLError("unreachable"),
    TimeoutError("slow"),
    OSError("connection refused"),
])
def test_chat_returns_empty_on_network_error(monkeypatch, exc):
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen({}, raise_exc=exc))
    assert ai.chat([{"role": "user", "content": "q"}]) == ""


def test_chat_returns_empty_on_malformed_response(monkeypatch):
    # Valid JSON but missing the choices structure → KeyError, caught → "".
    monkeypatch.setattr(ai.urllib.request, "urlopen",
                        _capture_urlopen({}, body=b'{"unexpected": true}'))
    assert ai.chat([{"role": "user", "content": "q"}]) == ""


def test_chat_returns_empty_on_non_json_body(monkeypatch):
    # Endpoint returned junk (ValueError from json.loads) → caught → "".
    monkeypatch.setattr(ai.urllib.request, "urlopen",
                        _capture_urlopen({}, body=b"not json at all"))
    assert ai.chat([{"role": "user", "content": "q"}]) == ""


def test_chat_returns_empty_when_content_is_null(monkeypatch):
    # A well-formed envelope whose content is null must not crash — `(None or "")` -> "".
    body = json.dumps({"choices": [{"message": {"content": None}}]}).encode("utf-8")
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen({}, body=body))
    assert ai.chat([{"role": "user", "content": "q"}]) == ""


# ---- health -------------------------------------------------------------------------------------

def test_health_reachable_on_2xx(monkeypatch):
    captured = {}
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, status=200))
    h = ai.health()
    assert h == {"reachable": True, "base": config.LLM_BASE, "model": config.LLM_MODEL}
    assert captured["req"].get_full_url().endswith("/models")
    assert captured["req"].get_method() == "GET"


def test_health_not_reachable_on_non_2xx(monkeypatch):
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen({}, status=500))
    assert ai.health()["reachable"] is False


def test_health_sets_auth_header_when_key_present(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "LLM_KEY", "k")
    monkeypatch.setattr(ai.urllib.request, "urlopen", _capture_urlopen(captured, status=200))
    ai.health()
    assert captured["req"].get_header("Authorization") == "Bearer k"


def test_health_returns_false_on_exception(monkeypatch):
    # Any transport failure → reachable False, never raises.
    monkeypatch.setattr(ai.urllib.request, "urlopen",
                        _capture_urlopen({}, raise_exc=urllib.error.URLError("down")))
    h = ai.health()
    assert h["reachable"] is False
    assert h["base"] == config.LLM_BASE
