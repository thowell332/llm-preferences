"""Ensure `lp` is importable when tests run from any cwd."""
from __future__ import annotations

import sys
from pathlib import Path

_linear_probes = Path(__file__).resolve().parents[1]
_utility_analysis = _linear_probes.parents[1]
for p in (_utility_analysis, _linear_probes):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
