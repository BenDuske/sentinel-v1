"""Generate Sentinel demo narration MP3s via ElevenLabs Ben-clone voice.

Reads credentials from ~/.openclaw/credentials/elevenlabs.env, splits the
demo into 6 beats + intro/backup, and writes one MP3 per beat under
sentinel-v1/media/narration/.
"""
from __future__ import annotations
import os
import sys
import pathlib
import urllib.request
import json

BEATS = [
    ("00-intro", (
        "Sentinel — a local-first incident and claims intelligence tool. "
        "Built by Ben Duske."
    )),
    ("01-hook", (
        "When an incident happens — a flood, a fire, a break-in — the claim starts as a messy note. "
        "Sentinel turns that note into a defensible incident report in seconds, "
        "and it runs entirely on your machine, so the claim data never leaves the building. "
        "The key: the severity is grounded and auditable, not a black-box guess."
    )),
    ("02-log", (
        "Watch. I log a real incident — a water leak in the server room, burst pipe overnight, "
        "two inches of standing water around the racks, one technician slipped and twisted an ankle, "
        "U-P-S units at risk. Click log and analyze. The detail panel fills instantly."
    )),
    ("03-grounded-severity", (
        "Severity is HIGH — and here's why. A deterministic rule layer matched water and flood, "
        "high, on burst pipe and standing water, and injury and medical, medium, on slip. "
        "The L-L-M's judgment is reconciled with it, and the higher always wins — "
        "a floor logic an insurer can audit line by line. "
        "Even with the L-L-M offline, the floor holds. No black box."
    )),
    ("04-human-in-loop", (
        "The A-I drafts. The human decides. "
        "Edit the summary, adjust recommended actions, override severity if you need to. "
        "Save edits — the history table and report update. "
        "Every decision is owned by a person."
    )),
    ("05-export", (
        "Export Markdown. A clean, professional incident report — "
        "severity plus rationale, summary, numbered next steps, evidence list, "
        "A-I-assisted disclaimer footer. "
        "P-D-F export is one click too. "
        "That's a report you can attach to a claim file — generated, reviewed, and owned by a human."
    )),
    ("06-close", (
        "Local-first, so claim data stays private. "
        "Grounded, so severity is defensible. "
        "Human-in-the-loop, so a person owns every decision. "
        "And it degrades gracefully — it still works with no A-I at all. "
        "That's Sentinel."
    )),
]


def load_env(path: pathlib.Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def tts(api_key: str, voice_id: str, text: str, out_path: pathlib.Path) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    body = json.dumps({
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.85, "style": 0.15},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("xi-api-key", api_key)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "audio/mpeg")
    with urllib.request.urlopen(req, timeout=120) as resp:
        out_path.write_bytes(resp.read())


def main() -> int:
    creds = pathlib.Path.home() / ".openclaw" / "credentials" / "elevenlabs.env"
    env = load_env(creds)
    api_key = env["ELEVENLABS_API_KEY"]
    voice_id = env["ELEVENLABS_BEN_CLONE_VOICE_ID"]

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "media" / "narration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, text in BEATS:
        out_path = out_dir / f"{name}.mp3"
        print(f"[tts] {name} -> {out_path}", flush=True)
        tts(api_key, voice_id, text, out_path)
        print(f"[tts] {name} OK ({out_path.stat().st_size} bytes)", flush=True)

    print(f"[tts] all done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
