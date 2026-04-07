#!/usr/bin/env python3
"""
Linear probes CLI. Implementation is in the `lp` package alongside this file.
"""
from __future__ import annotations

import os
import sys
import traceback

# utility_analysis/ is two levels up: linear_probes -> experiments -> utility_analysis
_UTILITY_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _UTILITY_ANALYSIS not in sys.path:
    sys.path.insert(0, _UTILITY_ANALYSIS)

# Same directory as this script (so `import lp` resolves)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lp.cli import main

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
