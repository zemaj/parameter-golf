from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class AdaptiveQKConfig:
    enabled: bool
    mode: str
    num_pos_buckets: int
    use_token_rms: bool
    log_delta_clamp: float
    effective_q_gain_max: float
    head_groups: int
    late_start_layer: int
    pos_start_layer: int


def load_config() -> AdaptiveQKConfig:
    return AdaptiveQKConfig(
        enabled=bool(int(os.environ.get("ADAPTIVE_QK_ENABLE", "1"))),
        mode=os.environ.get("ADAPTIVE_QK_MODE", "rms_pos"),
        num_pos_buckets=max(int(os.environ.get("ADAPTIVE_QK_NUM_POS_BUCKETS", "8")), 1),
        use_token_rms=bool(int(os.environ.get("ADAPTIVE_QK_USE_TOKEN_RMS", "1"))),
        log_delta_clamp=float(os.environ.get("ADAPTIVE_QK_LOG_DELTA_CLAMP", "1.25")),
        effective_q_gain_max=float(os.environ.get("ADAPTIVE_QK_EFFECTIVE_Q_GAIN_MAX", "0.0")),
        head_groups=max(int(os.environ.get("ADAPTIVE_QK_HEAD_GROUPS", "4")), 1),
        late_start_layer=max(int(os.environ.get("ADAPTIVE_QK_LATE_START_LAYER", "6")), 0),
        pos_start_layer=max(int(os.environ.get("ADAPTIVE_QK_POS_START_LAYER", "7")), 0),
    )


def config_summary(config: AdaptiveQKConfig) -> str:
    token_rms = 1 if config.use_token_rms else 0
    return (
        f"candidate:adaptive_qk enabled:{int(config.enabled)} mode:{config.mode} "
        f"pos_buckets:{config.num_pos_buckets} token_rms:{token_rms} "
        f"head_groups:{config.head_groups} late_start:{config.late_start_layer} "
        f"pos_start:{config.pos_start_layer} log_delta_clamp:{config.log_delta_clamp} "
        f"effective_q_gain_max:{config.effective_q_gain_max}"
    )


def install_adaptive_qk(baseline) -> AdaptiveQKConfig:
    config = load_config()
    if not config.enabled:
        return config

    if config.mode not in {"rms_pos", "late_headgroup", "late_headgroup_pos"}:
        raise ValueError(f"Unsupported ADAPTIVE_QK_MODE={config.mode}")

    class AdaptiveCausalSelfAttention(baseline.CausalSelfAttention):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            num_kv_heads: int,
            rope_base: float,
            qk_gain_init: float,
        ):
            super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
            self.layer_index = -1
            self.num_pos_buckets = config.num_pos_buckets
            self.log_delta_clamp = config.log_delta_clamp
            self.head_groups = min(config.head_groups, num_heads)
            head_group_ids = torch.arange(num_heads, dtype=torch.int64) * self.head_groups // num_heads
            self.register_buffer("head_group_ids", head_group_ids, persistent=False)
            if config.mode == "rms_pos":
                self.q_gain_pos_bias = nn.Parameter(torch.zeros(self.num_pos_buckets, num_heads, dtype=torch.float32))
                if config.use_token_rms:
                    self.q_gain_token_rms_weight = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
                else:
                    self.register_parameter("q_gain_token_rms_weight", None)
                self.register_parameter("q_gain_group_bias", None)
                self.register_parameter("q_gain_group_pos_bias", None)
            else:
                self.q_gain_group_bias = nn.Parameter(torch.zeros(self.head_groups, dtype=torch.float32))
                if config.mode == "late_headgroup_pos":
                    self.q_gain_group_pos_bias = nn.Parameter(torch.zeros(self.num_pos_buckets, self.head_groups, dtype=torch.float32))
                else:
                    self.register_parameter("q_gain_group_pos_bias", None)
                self.register_parameter("q_gain_pos_bias", None)
                self.register_parameter("q_gain_token_rms_weight", None)

        def _expand_group_values(self, group_values: Tensor) -> Tensor:
            if group_values.ndim == 1:
                return group_values[self.head_group_ids]
            if group_values.ndim == 2:
                return group_values[:, self.head_group_ids]
            raise ValueError(f"Unsupported group value rank {group_values.ndim}")

        def _rms_pos_multiplier(self, x: Tensor) -> Tensor:
            bsz, seqlen, _ = x.shape
            log_delta = self.q_gain_pos_bias.new_zeros((bsz, self.num_heads, seqlen))
            pos_ids = torch.clamp(
                (torch.arange(seqlen, device=x.device) * self.num_pos_buckets) // max(seqlen, 1),
                max=self.num_pos_buckets - 1,
            )
            pos_delta = self.q_gain_pos_bias[pos_ids].transpose(0, 1)[None, :, :]
            log_delta = log_delta + pos_delta
            if self.q_gain_token_rms_weight is not None:
                token_rms = x.float().square().mean(dim=-1).add(1e-6).sqrt()
                token_rms = token_rms / token_rms.mean(dim=1, keepdim=True).clamp_min(1e-6)
                token_feature = token_rms.log().clamp(-2.0, 2.0)
                log_delta = log_delta + self.q_gain_token_rms_weight[None, :, None] * token_feature[:, None, :]
            log_delta = torch.clamp(log_delta, min=-self.log_delta_clamp, max=self.log_delta_clamp)
            return log_delta.exp().to(dtype=x.dtype)

        def _late_headgroup_multiplier(self, x: Tensor) -> Tensor:
            bsz, seqlen, _ = x.shape
            log_delta = x.new_zeros((bsz, self.num_heads, seqlen), dtype=torch.float32)
            late_bias = self._expand_group_values(self.q_gain_group_bias).to(dtype=torch.float32)
            late_scale = 1.0 if self.layer_index >= config.late_start_layer else 0.0
            log_delta = log_delta + late_bias[None, :, None] * late_scale
            if self.q_gain_group_pos_bias is not None:
                pos_ids = torch.clamp(
                    (torch.arange(seqlen, device=x.device) * self.num_pos_buckets) // max(seqlen, 1),
                    max=self.num_pos_buckets - 1,
                )
                pos_group = self.q_gain_group_pos_bias[pos_ids]
                pos_delta = self._expand_group_values(pos_group).transpose(0, 1)[None, :, :]
                pos_scale = 1.0 if self.layer_index >= config.pos_start_layer else 0.0
                log_delta = log_delta + pos_delta * pos_scale
            log_delta = torch.clamp(log_delta, min=-self.log_delta_clamp, max=self.log_delta_clamp)
            return log_delta.exp().to(dtype=x.dtype)

        def _adaptive_q_gain_multiplier(self, x: Tensor) -> Tensor:
            if config.mode == "rms_pos":
                return self._rms_pos_multiplier(x)
            return self._late_headgroup_multiplier(x)

        def forward(self, x: Tensor) -> Tensor:
            bsz, seqlen, dim = x.shape
            q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = baseline.apply_rotary_emb(q, cos, sin)
            k = baseline.apply_rotary_emb(k, cos, sin)
            dynamic_mult = self._adaptive_q_gain_multiplier(x)
            effective_q_gain = self.q_gain.to(dtype=q.dtype)[None, :, None] * dynamic_mult
            if config.effective_q_gain_max > 0.0:
                effective_q_gain = effective_q_gain.clamp(max=config.effective_q_gain_max)
            q = q * effective_q_gain[:, :, :, None]
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            return self.proj(y)

    baseline.CausalSelfAttention = AdaptiveCausalSelfAttention
    original_gpt_init = baseline.GPT.__init__

    def gpt_init_with_layer_indices(self, *args, **kwargs):
        original_gpt_init(self, *args, **kwargs)
        for layer_idx, block in enumerate(self.blocks):
            block.attn.layer_index = layer_idx

    baseline.GPT.__init__ = gpt_init_with_layer_indices
    return config
