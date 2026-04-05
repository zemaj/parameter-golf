from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AdaptiveLastLayerQGain(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        head_groups: int,
        log_delta_clamp: float,
        effective_q_gain_max: float,
    ) -> None:
        super().__init__()
        if head_groups <= 0:
            raise ValueError(f"head_groups must be positive, got {head_groups}")
        if num_heads % head_groups != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by head_groups={head_groups}")
        if log_delta_clamp <= 0.0:
            raise ValueError(f"log_delta_clamp must be positive, got {log_delta_clamp}")
        if effective_q_gain_max <= 0.0:
            raise ValueError(f"effective_q_gain_max must be positive, got {effective_q_gain_max}")
        self.num_heads = num_heads
        self.head_groups = head_groups
        self.log_delta_clamp = float(log_delta_clamp)
        self.effective_q_gain_max = float(effective_q_gain_max)
        self.log_delta = nn.Parameter(torch.zeros(head_groups, dtype=torch.float32))

    def per_head_multiplier(self, base_q_gain: Tensor) -> Tensor:
        dtype = base_q_gain.dtype
        device = base_q_gain.device
        heads_per_group = self.num_heads // self.head_groups
        log_delta = self.log_delta.to(device=device, dtype=dtype).repeat_interleave(heads_per_group)
        log_delta = log_delta.clamp(-self.log_delta_clamp, self.log_delta_clamp)
        max_log_delta = math.log(self.effective_q_gain_max) - torch.log(base_q_gain.to(dtype=torch.float32).clamp_min(1e-6))
        max_log_delta = max_log_delta.to(device=device, dtype=dtype)
        log_delta = torch.minimum(log_delta, max_log_delta)
        return log_delta.exp().to(device=device, dtype=dtype)


def attach_adaptive_last_layer_qgain(
    blocks: nn.ModuleList,
    *,
    enabled: bool,
    num_heads: int,
    head_groups: int,
    log_delta_clamp: float,
    effective_q_gain_max: float,
) -> None:
    if not enabled:
        return
    if len(blocks) == 0:
        raise ValueError("Expected at least one transformer block for adaptive last-layer q_gain.")
    blocks[-1].attn.adaptive_last_layer_qgain = AdaptiveLastLayerQGain(
        num_heads=num_heads,
        head_groups=head_groups,
        log_delta_clamp=log_delta_clamp,
        effective_q_gain_max=effective_q_gain_max,
    )
