"""Central paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ESG_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
CHROMA_DIR = PROCESSED_DIR / "chroma"

for _d in (RAW_DIR, PROCESSED_DIR, GROUND_TRUTH_DIR):
    _d.mkdir(parents=True, exist_ok=True)
