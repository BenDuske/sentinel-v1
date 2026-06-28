"""Make `from sentinel import ...` importable for tests without installing the package.

Sentinel uses a src/ layout (src/sentinel). Adding src/ to sys.path here lets pytest run from
the repo root with `python -m pytest -q` and no install step.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
