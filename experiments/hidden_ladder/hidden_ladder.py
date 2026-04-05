from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class HiddenLadderConfig:
    enabled: bool
    rank: int
    tap_layers: tuple[int, ...]


def _parse_tap_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not layers:
        raise ValueError("HIDDEN_LADDER_TAP_LAYERS must contain at least one layer index")
    if len(set(layers)) != len(layers):
        raise ValueError(f"HIDDEN_LADDER_TAP_LAYERS contains duplicates: {raw}")
    return layers


def load_config() -> HiddenLadderConfig:
    return HiddenLadderConfig(
        enabled=bool(int(os.environ.get("HIDDEN_LADDER_ENABLE", "1"))),
        rank=max(int(os.environ.get("HIDDEN_LADDER_RANK", "2")), 0),
        tap_layers=_parse_tap_layers(os.environ.get("HIDDEN_LADDER_TAP_LAYERS", "7,8")),
    )


def config_summary(config: HiddenLadderConfig) -> str:
    taps = ",".join(str(layer) for layer in config.tap_layers)
    return (
        f"candidate:hidden_ladder enabled:{int(config.enabled)} "
        f"rank:{config.rank} taps:{taps}"
    )


class HiddenLadder(nn.Module):
    def __init__(self, dim: int, rank: int, tap_layers: tuple[int, ...]):
        super().__init__()
        self.tap_layers = tap_layers
        self.rank = rank
        init_scale = 0.02
        self.down_projs = nn.ParameterDict(
            {
                str(layer): nn.Parameter(torch.randn(dim, rank, dtype=torch.float32) * init_scale)
                for layer in tap_layers
            }
        )
        self.up_proj = nn.Parameter(torch.randn(rank, dim, dtype=torch.float32) * init_scale)
        self.gates = nn.Parameter(torch.zeros(len(tap_layers), dtype=torch.float32))

    def forward(self, x: Tensor, saved_states: dict[int, Tensor]) -> Tensor:
        update = x.new_zeros(x.shape)
        up_proj = self.up_proj.to(dtype=x.dtype)
        for gate, layer in zip(self.gates, self.tap_layers):
            tap_state = saved_states[layer]
            down_proj = self.down_projs[str(layer)].to(dtype=x.dtype)
            low_rank = torch.matmul(tap_state, down_proj)
            update = update + gate.to(dtype=x.dtype) * torch.matmul(low_rank, up_proj)
        return x + update


def install_hidden_ladder(baseline) -> HiddenLadderConfig:
    config = load_config()
    if not config.enabled or config.rank <= 0:
        return config

    original_gpt_init = baseline.GPT.__init__

    def gpt_init_with_hidden_ladder(self, *args, **kwargs):
        original_gpt_init(self, *args, **kwargs)
        if len(self.blocks) == 0:
            raise ValueError("Expected at least one transformer block.")
        if min(config.tap_layers) < 0 or max(config.tap_layers) >= len(self.blocks):
            raise ValueError(
                f"HIDDEN_LADDER_TAP_LAYERS={config.tap_layers} out of range for num_layers={len(self.blocks)}"
            )
        self.blocks[-1].hidden_ladder = HiddenLadder(
            dim=self.tok_emb.embedding_dim,
            rank=config.rank,
            tap_layers=config.tap_layers,
        )

    def gpt_forward_with_hidden_ladder(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        saved_states: dict[int, Tensor] = {}

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if i in self.blocks[-1].hidden_ladder.tap_layers:
                saved_states[i] = x
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            x = self.blocks[block_idx](x, x0)
            if block_idx in self.blocks[-1].hidden_ladder.tap_layers:
                saved_states[block_idx] = x

        x = self.blocks[-1].hidden_ladder(x, saved_states)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    baseline.GPT.__init__ = gpt_init_with_hidden_ladder
    baseline.GPT.forward = gpt_forward_with_hidden_ladder
    return config
