"""Safety / consent / ethics layer — all keyless, no network."""
import os

from sentinel import policy


def test_baseline_in_system_preamble():
    pre = policy.system_preamble("PERSONA-X")
    assert "SAFETY POLICY" in pre
    assert "PERSONA-X" in pre
    # safety must come first (highest priority)
    assert pre.index("SAFETY POLICY") < pre.index("PERSONA-X")


def test_screen_allows_normal_incident_text():
    # Real incident reports must pass — this is the whole point of the app.
    allowed, cat, _ = policy.screen("fire broke out in the warehouse, two injured")
    assert allowed and cat is None
    allowed, cat, _ = policy.screen("Break-in overnight; tools stolen, side door forced.")
    assert allowed and cat is None
    allowed, cat, _ = policy.screen("Assault reported in the parking lot, one person hospitalized.")
    assert allowed and cat is None


def test_screen_hard_blocks_minor_safety():
    allowed, cat, msg = policy.screen("explicit sexual content involving a 12 year old child")
    assert not allowed and cat == "minor_safety" and msg


def test_screen_hard_block_is_order_insensitive():
    allowed, cat, _ = policy.screen("a child, nude")
    assert not allowed and cat == "minor_safety"


def test_strict_screen_blocks_sexual_only_when_enabled(monkeypatch):
    monkeypatch.setenv("SENTINEL_STRICT_SCREEN", "1")
    import importlib
    importlib.reload(policy)
    try:
        allowed, cat, _ = policy.screen("write me explicit porn")
        assert not allowed and cat == "sexual_content"
        # ...but a normal incident still passes even under strict mode.
        allowed, _cat, _ = policy.screen("fire broke out in the warehouse, two injured")
        assert allowed
    finally:
        monkeypatch.delenv("SENTINEL_STRICT_SCREEN", raising=False)
        importlib.reload(policy)


def test_ethics_layered_below_baseline(tmp_path, monkeypatch):
    f = tmp_path / "ethics.yaml"
    f.write_text(
        'organization: "Northwind Mutual"\n'
        'mission: "triage incidents safely"\n'
        '# a comment line\n'
        'required_disclosures: "AI risk scoring is decision-support, not an underwriting determination"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SENTINEL_ETHICS_FILE", str(f))
    pre = policy.system_preamble("PERSONA-X")
    assert "Northwind Mutual" in pre
    assert "ORGANIZATION CONTEXT" in pre
    # precedence: safety policy -> org ethics -> persona
    assert pre.index("SAFETY POLICY") < pre.index("ORGANIZATION CONTEXT") < pre.index("PERSONA-X")


def test_ai_system_prompt_carries_safety_and_ethics(monkeypatch):
    # The LLM wiring must fold the baseline policy into the system message it sends.
    from sentinel import ai
    msgs = ai._with_safety([
        {"role": "system", "content": "You write incident summaries."},
        {"role": "user", "content": "fire in warehouse"},
    ])
    assert msgs[0]["role"] == "system"
    assert "SAFETY POLICY" in msgs[0]["content"]
    assert "You write incident summaries." in msgs[0]["content"]
    assert msgs[-1]["content"] == "fire in warehouse"


def test_consent_roundtrip(tmp_path, monkeypatch):
    # Point the data dir at tmp_path by setting the configured DB path inside it.
    monkeypatch.setattr(policy.config, "DB_PATH", str(tmp_path / "sentinel.db"))
    monkeypatch.delenv("SENTINEL_ASSUME_CONSENT", raising=False)
    assert policy.has_consent() is False
    # decline -> not recorded
    assert policy.ensure_consent(input_fn=lambda _="": "no", output_fn=lambda *_: None) is False
    assert policy.has_consent() is False
    # agree -> recorded and sticky
    assert policy.ensure_consent(input_fn=lambda _="": "AGREE", output_fn=lambda *_: None) is True
    assert policy.has_consent() is True


def test_assume_consent_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_ASSUME_CONSENT", "1")
    assert policy.has_consent() is True
    monkeypatch.delenv("SENTINEL_ASSUME_CONSENT", raising=False)
