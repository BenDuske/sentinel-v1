"""AI helper parsing layer — the offline/local-model robustness path.

ai.py's public helpers (summarize / recommend / llm_severity) and the private parsers
(_strip_think / _clean) turn messy local-model output into clean structured data, and degrade to
deterministic fallbacks when the endpoint is offline. That degrade-gracefully behavior is Sentinel's
local-first differentiator, but until now only risk.py's use of llm_severity was covered — the
parsers themselves had no direct tests. These exercise them with no network (ai.chat monkeypatched
to return canned model text), so they pin the real quirks local models emit: <think> reasoning
blocks, ```json code fences, bullet lists instead of JSON, and empty/offline responses.
"""
from sentinel import ai, risk_fallback


# ---- _strip_think -------------------------------------------------------------------------------

def test_strip_think_removes_reasoning_block():
    assert ai._strip_think("<think>weighing options</think>\nHIGH") == "HIGH"


def test_strip_think_uses_last_close_tag():
    # Some models emit multiple/nested think fragments; keep only what follows the final close.
    assert ai._strip_think("<think>a</think>mid<think>b</think>ANSWER") == "ANSWER"


def test_strip_think_passthrough_when_absent():
    assert ai._strip_think("just the answer") == "just the answer"


def test_strip_think_empty_is_empty():
    assert ai._strip_think("") == ""


# ---- _clean -------------------------------------------------------------------------------------

def test_clean_strips_think_and_wrapping_quotes():
    assert ai._clean('<think>x</think>"Final report text."') == "Final report text."


# ---- llm_severity -------------------------------------------------------------------------------

def test_llm_severity_extracts_each_level(monkeypatch):
    for word in ("low", "medium", "high", "critical"):
        monkeypatch.setattr(ai, "chat", lambda *a, _w=word, **k: _w.upper())
        assert ai.llm_severity("anything") == word


def test_llm_severity_ignores_severity_words_inside_think_block(monkeypatch):
    # A reasoning block that mentions a HIGHER severity must not leak into the classification —
    # this is exactly why llm_severity strips <think> before scanning for the level word.
    monkeypatch.setattr(ai, "chat",
                        lambda *a, **k: "<think>this is not critical, probably</think>\nmedium")
    assert ai.llm_severity("anything") == "medium"


def test_llm_severity_empty_when_offline(monkeypatch):
    monkeypatch.setattr(ai, "chat", lambda *a, **k: "")
    assert ai.llm_severity("anything") == ""


# ---- recommend --------------------------------------------------------------------------------

def test_recommend_parses_clean_json_array(monkeypatch):
    monkeypatch.setattr(ai, "chat",
                        lambda *a, **k: '["Isolate the area", "Notify facilities", "Log it"]')
    actions = ai.recommend({"title": "Leak", "description": "water on floor"})
    assert actions == ["Isolate the area", "Notify facilities", "Log it"]


def test_recommend_parses_json_inside_code_fence_and_think(monkeypatch):
    # Real local-model output: a reasoning block, then the JSON wrapped in a ```json fence.
    monkeypatch.setattr(ai, "chat", lambda *a, **k:
                        '<think>ok</think>\n```json\n["Call 911", "Evacuate"]\n```')
    actions = ai.recommend({"title": "Fire", "description": "smoke seen"})
    assert actions == ["Call 911", "Evacuate"]


def test_recommend_falls_back_to_line_split_for_bullet_list(monkeypatch):
    monkeypatch.setattr(ai, "chat", lambda *a, **k:
                        "- Shut off water\n* Call facilities\n- Document damage")
    actions = ai.recommend({"title": "Leak", "description": "burst pipe"})
    assert actions == ["Shut off water", "Call facilities", "Document damage"]


def test_recommend_caps_at_five(monkeypatch):
    monkeypatch.setattr(ai, "chat", lambda *a, **k: '["a","b","c","d","e","f","g"]')
    assert ai.recommend({"title": "x", "description": "y"}) == ["a", "b", "c", "d", "e"]


def test_recommend_uses_deterministic_fallback_when_offline(monkeypatch):
    inc = {"title": "Burst pipe", "description": "water damage", "severity": "high"}
    monkeypatch.setattr(ai, "chat", lambda *a, **k: "")
    assert ai.recommend(inc) == risk_fallback.fallback_actions(inc)


# ---- summarize --------------------------------------------------------------------------------

def test_summarize_returns_cleaned_llm_text(monkeypatch):
    monkeypatch.setattr(ai, "chat",
                        lambda *a, **k: '<think>drafting</think>"A concise incident summary."')
    out = ai.summarize({"title": "Outage", "description": "server down"})
    assert out == "A concise incident summary."


def test_summarize_uses_deterministic_fallback_when_offline(monkeypatch):
    inc = {"title": "Outage", "description": "server down", "severity": "high"}
    monkeypatch.setattr(ai, "chat", lambda *a, **k: "")
    assert ai.summarize(inc) == risk_fallback.fallback_summary(inc)
