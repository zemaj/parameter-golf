from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

try:
    from experiments.hidden_ladder.hidden_ladder import config_summary, install_hidden_ladder
except ModuleNotFoundError:
    from hidden_ladder import config_summary, install_hidden_ladder


DEFAULT_CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights"
)


def _append_env_pattern(key: str, pattern: str) -> None:
    seed = DEFAULT_CONTROL_TENSOR_NAME_PATTERNS if key in {
        "CONTROL_TENSOR_NAME_PATTERNS",
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    } else ""
    existing = [part for part in os.environ.get(key, seed).split(",") if part]
    if pattern not in existing:
        existing.append(pattern)
    os.environ[key] = ",".join(existing)


def load_baseline_module():
    baseline_path = Path(__file__).resolve().parents[2] / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("baseline_train_gpt", baseline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline trainer from {baseline_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _append_env_pattern("CONTROL_TENSOR_NAME_PATTERNS", "hidden_ladder")
    _append_env_pattern("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", "hidden_ladder")
    baseline = load_baseline_module()
    config = install_hidden_ladder(baseline)
    print(config_summary(config), flush=True)
    baseline.main()


if __name__ == "__main__":
    main()
