from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LastLayerQGainConfig:
    enabled: bool
    qk_gain_init: float


def load_config() -> LastLayerQGainConfig:
    return LastLayerQGainConfig(
        enabled=bool(int(os.environ.get("LAST_LAYER_QGAIN_ENABLE", "1"))),
        qk_gain_init=float(os.environ["LAST_LAYER_QK_GAIN_INIT"]),
    )


def config_summary(config: LastLayerQGainConfig) -> str:
    return (
        f"candidate:last_layer_qgain enabled:{int(config.enabled)} "
        f"last_layer_qk_gain_init:{config.qk_gain_init}"
    )


def install_last_layer_qgain(baseline) -> LastLayerQGainConfig:
    config = load_config()
    if not config.enabled:
        return config

    original_gpt_init = baseline.GPT.__init__

    def gpt_init_with_last_layer_override(self, *args, **kwargs):
        original_gpt_init(self, *args, **kwargs)
        if len(self.blocks) == 0:
            raise ValueError("Expected at least one transformer block.")
        with torch.no_grad():
            self.blocks[-1].attn.q_gain.fill_(config.qk_gain_init)

    baseline.GPT.__init__ = gpt_init_with_last_layer_override
    return config
