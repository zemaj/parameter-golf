"""Hash lexical prior experiment for Parameter Golf.

This wrapper keeps the root baseline trainer intact and injects a small,
hash-indexed auxiliary prior branch that mixes directly into logits.

The idea is intentionally weird-but-plausible:
- build a tiny lexical prior from previous-token ids only,
- use multiple hashes into a fixed bucket table (count-sketch style),
- project to vocab logits and blend before softcapping.

All controls are env-driven so generic harnesses can sweep settings and parse
stdout for the "candidate:" and "aux_hash_prior:" markers.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import the baseline trainer from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import train_gpt as baseline


class Hyperparameters(baseline.Hyperparameters):
    aux_prior_enable = bool(int(os.environ.get("AUX_PRIOR_ENABLE", "1")))
    aux_prior_buckets = int(os.environ.get("AUX_PRIOR_BUCKETS", 4096))
    aux_prior_dim = int(os.environ.get("AUX_PRIOR_DIM", 16))
    aux_prior_hashes = int(os.environ.get("AUX_PRIOR_HASHES", 2))
    aux_prior_scale_init = float(os.environ.get("AUX_PRIOR_SCALE_INIT", 0.03))


class HashLexicalPrior(nn.Module):
    """Tiny hashed n-gram style prior mixed into logits."""

    def __init__(self, vocab_size: int, buckets: int, prior_dim: int, num_hashes: int, scale_init: float):
        super().__init__()
        if buckets <= 0:
            raise ValueError(f"AUX_PRIOR_BUCKETS must be positive, got {buckets}")
        if prior_dim <= 0:
            raise ValueError(f"AUX_PRIOR_DIM must be positive, got {prior_dim}")
        if num_hashes <= 0:
            raise ValueError(f"AUX_PRIOR_HASHES must be positive, got {num_hashes}")

        self.vocab_size = vocab_size
        self.buckets = buckets
        self.num_hashes = num_hashes
        self.embed = nn.Embedding(buckets * num_hashes, prior_dim)
        self.proj = baseline.CastedLinear(prior_dim, vocab_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

        # Fixed hash coefficients (odd multipliers) for deterministic reproducibility.
        mul = [1_000_003 + 2 * i for i in range(num_hashes)]
        add = [3_000_017 + 2 * i for i in range(num_hashes)]
        self.register_buffer("_hash_mul", torch.tensor(mul, dtype=torch.int64), persistent=False)
        self.register_buffer("_hash_add", torch.tensor(add, dtype=torch.int64), persistent=False)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.weight)

    def _hash_indices(self, prev_ids: Tensor) -> Tensor:
        # prev_ids: [B, T], returns [H, B, T]
        seq_len = prev_ids.size(1)
        pos = torch.arange(seq_len, device=prev_ids.device, dtype=torch.int64)[None, :]
        prev64 = prev_ids.to(dtype=torch.int64)
        hashed = []
        for h in range(self.num_hashes):
            mixed = (prev64 + 1) * self._hash_mul[h] + (pos + 1) * self._hash_add[h]
            bucket = torch.remainder(mixed, self.buckets) + h * self.buckets
            hashed.append(bucket)
        return torch.stack(hashed, dim=0)

    def forward(self, input_ids: Tensor) -> Tensor:
        # Causal lexical context: only previous token contributes at each position.
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        idx = self._hash_indices(prev_ids)
        h = self.embed(idx).sum(dim=0) * (1.0 / math.sqrt(self.num_hashes))
        logits = self.proj(h)
        return logits * self.scale.to(dtype=logits.dtype)


class GPT(baseline.GPT):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            tie_embeddings=tie_embeddings,
            tied_embed_init_std=tied_embed_init_std,
            logit_softcap=logit_softcap,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
        )

        args = Hyperparameters()
        if args.aux_prior_enable:
            if not self.blocks:
                raise ValueError("AUX_PRIOR_ENABLE requires at least one transformer block")
            # Attach inside blocks so baseline optimizer param-group split picks it up.
            self.blocks[0].hash_aux_prior = HashLexicalPrior(
                vocab_size=vocab_size,
                buckets=args.aux_prior_buckets,
                prior_dim=args.aux_prior_dim,
                num_hashes=args.aux_prior_hashes,
                scale_init=args.aux_prior_scale_init,
            )

    def _aux_prior(self) -> nn.Module | None:
        if not self.blocks:
            return None
        return getattr(self.blocks[0], "hash_aux_prior", None)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)

        aux_prior = self._aux_prior()
        if aux_prior is not None:
            logits_proj = logits_proj + aux_prior(input_ids).reshape_as(logits_proj)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def main() -> None:
    baseline.Hyperparameters = Hyperparameters
    baseline.GPT = GPT

    if os.environ.get("RANK", "0") == "0":
        args = Hyperparameters()
        print(
            "candidate:hash_aux_prior "
            f"enabled={int(args.aux_prior_enable)} "
            f"buckets={args.aux_prior_buckets} dim={args.aux_prior_dim} "
            f"hashes={args.aux_prior_hashes} scale_init={args.aux_prior_scale_init}"
        )
        if args.aux_prior_enable:
            est_params = args.aux_prior_buckets * args.aux_prior_dim * args.aux_prior_hashes
            est_params += args.aux_prior_dim * args.vocab_size + 1
            print(f"aux_hash_prior:enabled est_params={est_params}")
        else:
            print("aux_hash_prior:disabled")

    baseline.main()


if __name__ == "__main__":
    main()
