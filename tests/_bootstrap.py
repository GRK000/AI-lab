from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROM_SCRATCH = ROOT / "from-scratch"

if str(FROM_SCRATCH) not in sys.path:
    sys.path.insert(0, str(FROM_SCRATCH))
