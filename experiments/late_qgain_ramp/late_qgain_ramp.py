from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LateQGainRampConfig:
    enabled: bool
    start_layer: int
    target_qk_gain_init: float


def load_config() -> LateQGainRampConfig:
    return LateQGainRampConfig(
        enabled=bool(int(os.environ.get("LATE_QGAIN_RAMP_ENABLE", "1"))),
        start_layer=int(os.environ["LATE_QGAIN_RAMP_START_LAYER"]),
        target_qk_gain_init=float(os.environ["LATE_QGAIN_RAMP_TARGET_QK_GAIN_INIT"]),
    )


def config_summary(config: LateQGainRampConfig) -> str:
    return (
        f"candidate:late_qgain_ramp enabled:{int(config.enabled)} "
        f"start_layer:{config.start_layer} target_qk_gain_init:{config.target_qk_gain_init}"
    )


def install_late_qgain_ramp(baseline) -> LateQGainRampConfig:
    config = load_config()
    if not config.enabled:
        return config

    original_gpt_init = baseline.GPT.__init__

    def gpt_init_with_late_ramp(self, *args, **kwargs):
        original_gpt_init(self, *args, **kwargs)
        num_blocks = len(self.blocks)
        if num_blocks == 0:
            raise ValueError("Expected at least one transformer block.")
        if not (0 <= config.start_layer < num_blocks):
            raise ValueError(
                f"LATE_QGAIN_RAMP_START_LAYER={config.start_layer} must be in [0, {num_blocks - 1}]"
            )
        late_count = num_blocks - config.start_layer
        base_q_gain = float(self.blocks[config.start_layer].attn.q_gain[0].item())
        schedule: list[float] = []
        with torch.no_grad():
            for layer_idx in range(config.start_layer, num_blocks):
                frac = float(layer_idx - config.start_layer + 1) / float(late_count)
                gain = base_q_gain + (config.target_qk_gain_init - base_q_gain) * frac
                self.blocks[layer_idx].attn.q_gain.fill_(gain)
                schedule.append(gain)
        schedule_str = ",".join(f"{gain:.3f}" for gain in schedule)
        print(
            f"candidate:late_qgain_ramp_schedule start:{config.start_layer} "
            f"base:{base_q_gain:.3f} target:{config.target_qk_gain_init:.3f} "
            f"values:{schedule_str}",
            flush=True,
        )

    baseline.GPT.__init__ = gpt_init_with_late_ramp
    return config
