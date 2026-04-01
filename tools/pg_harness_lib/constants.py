from __future__ import annotations

import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = ROOT / "experiments" / "presets"
RUNS_DIR = ROOT / "experiments" / "runs"

FLOAT_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"
FINAL_METRIC_RE = re.compile(
    rf"^(?P<label>(?:final[^\s:]*|benchmark_[^\s:]*))[^\n]*?val_loss:(?P<val_loss>{FLOAT_RE})[^\n]*?val_bpb:(?P<val_bpb>{FLOAT_RE})",
)
ARTIFACT_RE = re.compile(r"^Total submission size(?: [^:]+)?: (?P<bytes>\d+) bytes")
MODEL_RE = re.compile(r"^Serialized model(?: [^:]+)?: (?P<bytes>\d+) bytes")
CODE_RE = re.compile(r"^Code size: (?P<bytes>\d+) bytes")
STEP_RE = re.compile(r"step:(?P<step>\d+)/(?:\d+)")

TARGET_SNAPSHOTS: dict[str, dict[str, Any]] = {
    "main-track-sota": {
        "label": "main-track-sota",
        "target_bpb": 1.1147,
        "as_of": "2026-04-01",
        "source": "https://github.com/openai/parameter-golf/blob/main/README.md",
        "notes": "Main-track best score visible in the public README leaderboard on 2026-04-01.",
    },
    "main-track-sota-2026-04-01": {
        "label": "main-track-sota-2026-04-01",
        "target_bpb": 1.1147,
        "as_of": "2026-04-01",
        "source": "https://github.com/openai/parameter-golf/blob/main/README.md",
        "notes": "Pinned snapshot of the main-track leaderboard on 2026-04-01.",
    },
}
DEFAULT_TARGET = "main-track-sota"

