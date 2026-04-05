from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from experiments.adaptive_qk.adaptive_qk import config_summary, install_adaptive_qk
except ModuleNotFoundError:
    from adaptive_qk import config_summary, install_adaptive_qk


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
    baseline = load_baseline_module()
    config = install_adaptive_qk(baseline)
    print(config_summary(config), flush=True)
    baseline.main()


if __name__ == "__main__":
    main()
