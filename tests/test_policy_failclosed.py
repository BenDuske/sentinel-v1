"""Fail-closed and resilience guarantees for the safety/consent layer.

These pin the *defensive* branches of policy.py that the happy-path tests in
test_policy.py don't exercise: a corrupt or stale consent record must NEVER be
read as consent, an interrupted first-run prompt must not silently grant it, and
an unreadable ethics file must degrade to empty rather than crash. All keyless,
no network, no real user data.
"""
import json
import os

from sentinel import policy


# --- screen(): empty input is not "prohibited content" ---------------------------------------
def test_screen_empty_and_none_pass():
    # Emptiness is handled elsewhere in the pipeline; the safety screen must not
    # treat "" / None as a block (that would be a false positive on every blank field).
    assert policy.screen("") == (True, None, None)
    assert policy.screen(None) == (True, None, None)


# --- has_consent(): must FAIL CLOSED on a bad record -----------------------------------------
def _point_data_dir_at(tmp_path, monkeypatch):
    monkeypatch.setattr(policy.config, "DB_PATH", str(tmp_path / "sentinel.db"))
    monkeypatch.delenv("SENTINEL_ASSUME_CONSENT", raising=False)


def test_corrupt_consent_file_is_not_consent(tmp_path, monkeypatch):
    _point_data_dir_at(tmp_path, monkeypatch)
    p = policy._consent_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json")
    # A malformed file must be treated as "no valid consent", never granted.
    assert policy.has_consent() is False


def test_stale_policy_version_is_not_consent(tmp_path, monkeypatch):
    _point_data_dir_at(tmp_path, monkeypatch)
    p = policy._consent_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"version": "1999-01-01", "agreed": True}, fh)
    # Consent recorded against a superseded policy version does not carry forward.
    assert policy.has_consent() is False
    # ...and the current version is honored (sanity anchor for the version check).
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"version": policy.POLICY_VERSION, "agreed": True}, fh)
    assert policy.has_consent() is True


# --- ensure_consent(): interrupted prompt must not grant OR record consent --------------------
def test_interrupted_prompt_does_not_grant_or_record(tmp_path, monkeypatch):
    _point_data_dir_at(tmp_path, monkeypatch)

    def eof(_=""):
        raise EOFError()

    def ctrlc(_=""):
        raise KeyboardInterrupt()

    for boom in (eof, ctrlc):
        assert policy.ensure_consent(input_fn=boom, output_fn=lambda *_: None) is False
        assert not os.path.exists(policy._consent_path())
        assert policy.has_consent() is False


def test_ensure_consent_shortcircuits_when_already_agreed(tmp_path, monkeypatch):
    _point_data_dir_at(tmp_path, monkeypatch)
    policy.record_consent()

    def must_not_be_called(_=""):
        raise AssertionError("input_fn must not run once consent is on file")

    # Already-consented path returns True without re-prompting.
    assert policy.ensure_consent(input_fn=must_not_be_called, output_fn=lambda *_: None) is True


# --- load_ethics(): unreadable file degrades to {}, never raises ------------------------------
def test_load_ethics_on_unreadable_path_returns_empty(tmp_path, monkeypatch):
    # Point the ethics file at a directory: os.path.exists() is True but open() raises.
    monkeypatch.setenv("SENTINEL_ETHICS_FILE", str(tmp_path))
    assert policy.load_ethics() == {}
    # A missing path is also empty (and system_preamble stays baseline-only).
    monkeypatch.setenv("SENTINEL_ETHICS_FILE", str(tmp_path / "nope.yaml"))
    assert policy.load_ethics() == {}
    assert policy.ethics_preamble({}) == ""


# --- _data_dir(): falls back to ~/.sentinel when no DB path is configured ----------------------
def test_data_dir_falls_back_to_home_sentinel(monkeypatch):
    monkeypatch.setattr(policy.config, "DB_PATH", "")
    d = policy._data_dir()
    assert os.path.basename(d) == ".sentinel"
    assert os.path.isabs(d)
