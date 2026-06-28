"""Configuration via env (see .env.example)."""
import os

LLM_BASE  = os.environ.get("SENTINEL_LLM_BASE", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("SENTINEL_LLM_MODEL", "qwen3:30b")
LLM_KEY   = os.environ.get("SENTINEL_LLM_KEY", "")
DB_PATH   = os.environ.get("SENTINEL_DB", "./sentinel.db")
EVIDENCE_DIR = os.environ.get("SENTINEL_EVIDENCE_DIR", "./evidence")
