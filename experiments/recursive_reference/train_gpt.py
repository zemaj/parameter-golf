"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from experiments.recursive_reference.export_diagnostics import build_quant_diagnostic_lines, classify_tensor_family
except ModuleNotFoundError:
    from export_diagnostics import build_quant_diagnostic_lines, classify_tensor_family

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    dyn_qk_pos_buckets = int(os.environ.get("DYN_QK_POS_BUCKETS", 8))
    recur_start_layer = int(os.environ.get("RECUR_START_LAYER", 4))
    recur_num_layers = int(os.environ.get("RECUR_NUM_LAYERS", 2))
    recur_extra_passes = int(os.environ.get("RECUR_EXTRA_PASSES", 1))
    recur_start_step = int(os.environ.get("RECUR_START_STEP", 0))
    recur_warmup_steps = int(os.environ.get("RECUR_WARMUP_STEPS", 0))
    dual_lane_start_layer = int(os.environ.get("DUAL_LANE_START_LAYER", 7))
    lora_rank = int(os.environ.get("RECUR_LORA_RANK", 2))
    lora_alpha = float(os.environ.get("RECUR_LORA_ALPHA", 1.0))
    xsa_lambda_init = float(os.environ.get("XSA_LAMBDA_INIT", 1.0))
    delta_rank = int(os.environ.get("DELTA_RANK", 16))
    enable_recur = bool(int(os.environ.get("ENABLE_RECUR", "1")))
    enable_dual_lanes = bool(int(os.environ.get("ENABLE_DUAL_LANES", "1")))
    enable_dynamic_qk = bool(int(os.environ.get("ENABLE_DYNAMIC_QK", "1")))
    enable_xsa = bool(int(os.environ.get("ENABLE_XSA", "1")))
    enable_delta_subspace = bool(int(os.environ.get("ENABLE_DELTA_SUBSPACE", "1")))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 2.0))
    late_mlp_mult = float(os.environ.get("LATE_MLP_MULT", "0"))
    late_mlp_start_layer = int(os.environ.get("LATE_MLP_START_LAYER", str(num_layers)))
    late_num_kv_heads = int(os.environ.get("LATE_NUM_KV_HEADS", "0"))
    late_kv_start_layer = int(os.environ.get("LATE_KV_START_LAYER", str(num_layers)))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,xsa_lambda,lane_mix,delta_scale,lane_merge,q_lora,o_lora,delta_basis,state_proj",
    ).split(",")
    if pattern
)
INT8_FORCE_PASSTHROUGH_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_FORCE_PASSTHROUGH_PATTERNS", "").split(",") if pattern
)
INT8_FORCE_INT8_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_FORCE_INT8_PATTERNS", "").split(",") if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("INT8_KEEP_FLOAT_MAX_NUMEL", "65536"))
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", "99.99984"))
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_LOG_TOP_TENSORS = int(os.environ.get("INT8_LOG_TOP_TENSORS", "8"))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def matches_any_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


def resolve_mlp_hidden_dim(dim: int, mult: float) -> int:
    if mult <= 0:
        raise ValueError(f"mlp multiplier must be positive, got {mult}")
    return max(64, int(round((dim * mult) / 64.0)) * 64)

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if matches_any_pattern(name, INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    entries: list[dict[str, object]] = []
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            entries.append(
                {
                    "name": name,
                    "shape": list(t.shape),
                    "numel": int(t.numel()),
                    "baseline_bytes": tensor_nbytes(t),
                    "payload_bytes": tensor_nbytes(t),
                    "mode": "nonfloat_passthrough",
                    "family": classify_tensor_family(name),
                }
            )
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        force_passthrough = matches_any_pattern(name, INT8_FORCE_PASSTHROUGH_PATTERNS)
        force_int8 = matches_any_pattern(name, INT8_FORCE_INT8_PATTERNS)
        if force_passthrough and force_int8:
            raise ValueError(f"Tensor {name} matched both INT8_FORCE_PASSTHROUGH_PATTERNS and INT8_FORCE_INT8_PATTERNS")
        if force_passthrough or (not force_int8 and t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            entries.append(
                {
                    "name": name,
                    "shape": list(t.shape),
                    "numel": int(t.numel()),
                    "baseline_bytes": tensor_nbytes(t),
                    "payload_bytes": tensor_nbytes(kept),
                    "mode": "fp32_passthrough" if kept.dtype == torch.float32 else "fp16_passthrough",
                    "family": classify_tensor_family(name),
                }
            )
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        entries.append(
            {
                "name": name,
                "shape": list(t.shape),
                "numel": int(t.numel()),
                "baseline_bytes": tensor_nbytes(t),
                "payload_bytes": tensor_nbytes(q) + tensor_nbytes(s),
                "mode": "int8_per_row" if s.ndim > 0 else "int8_per_tensor",
                "family": classify_tensor_family(name),
            }
        )

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    stats["entries"] = entries
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class LowRankAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.a = nn.Parameter(torch.empty(rank, in_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(out_dim, rank, dtype=torch.float32))
        nn.init.normal_(self.a, mean=0.0, std=1.0 / math.sqrt(in_dim))

    def forward(self, x: Tensor) -> Tensor:
        inner = F.linear(x, self.a.to(dtype=x.dtype))
        return F.linear(inner, self.b.to(dtype=x.dtype)) * (self.alpha / self.rank)


class StaticQGain(nn.Module):
    def __init__(self, num_heads: int, qk_gain_init: float):
        super().__init__()
        if qk_gain_init <= 0.0:
            raise ValueError(f"qk_gain_init must be positive, got {qk_gain_init}")
        self.log_gain = nn.Parameter(torch.full((num_heads,), math.log(qk_gain_init), dtype=torch.float32))

    def forward(self, x: Tensor, lane_partner: Tensor | None, pass_index: int, recur_strength: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        log_gain = torch.clamp(self.log_gain, min=math.log(0.25), max=math.log(8.0))
        return log_gain.exp().to(dtype=x.dtype)[None, :, None].expand(bsz, -1, seqlen)


class DynamicQGain(nn.Module):
    def __init__(self, num_heads: int, qk_gain_init: float, max_passes: int, num_pos_buckets: int):
        super().__init__()
        if qk_gain_init <= 0.0:
            raise ValueError(f"qk_gain_init must be positive, got {qk_gain_init}")
        self.num_pos_buckets = max(num_pos_buckets, 1)
        log_init = math.log(qk_gain_init)
        self.q_gain_base = nn.Parameter(torch.full((num_heads,), log_init, dtype=torch.float32))
        self.q_gain_layer_bias = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.q_gain_pass_bias = nn.Parameter(torch.zeros(max_passes, num_heads, dtype=torch.float32))
        if max_passes > 1:
            with torch.no_grad():
                self.q_gain_pass_bias[1:].fill_(math.log(1.15))
        self.q_gain_pos_bias = nn.Parameter(torch.zeros(self.num_pos_buckets, num_heads, dtype=torch.float32))
        self.q_gain_feature_weights = nn.Parameter(torch.zeros(4, num_heads, dtype=torch.float32))

    def forward(self, x: Tensor, lane_partner: Tensor | None, pass_index: int, recur_strength: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        x32 = x.float()
        main_rms = x32.square().mean(dim=-1).add_(1e-6).sqrt_()
        if lane_partner is None:
            lane_ratio = torch.full_like(main_rms, 0.5)
        else:
            other_rms = lane_partner.float().square().mean(dim=-1).add_(1e-6).sqrt_()
            lane_ratio = main_rms / (main_rms + other_rms + 1e-6)
        pos = torch.arange(seqlen, device=x.device, dtype=main_rms.dtype)
        pos = pos / max(seqlen - 1, 1)
        pos = pos[None, :].expand(bsz, seqlen)
        recur_feat = main_rms * 0.0 + recur_strength.to(dtype=main_rms.dtype)
        features = torch.stack((main_rms, lane_ratio, pos, recur_feat), dim=-1)
        token_delta = torch.einsum("btf,fh->bht", features, self.q_gain_feature_weights)
        pos_ids = torch.clamp((torch.arange(seqlen, device=x.device) * self.num_pos_buckets) // max(seqlen, 1), max=self.num_pos_buckets - 1)
        pos_delta = self.q_gain_pos_bias[pos_ids].transpose(0, 1)[None, :, :]
        log_gain = (
            self.q_gain_base[None, :, None]
            + self.q_gain_layer_bias[None, :, None]
            + self.q_gain_pass_bias[min(pass_index, self.q_gain_pass_bias.size(0) - 1)][None, :, None]
            + pos_delta
            + token_delta
        )
        log_gain = torch.clamp(log_gain, min=math.log(0.25), max=math.log(8.0))
        return log_gain.exp().to(dtype=x.dtype)


class DynamicCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        max_passes: int,
        num_pos_buckets: int,
        lora_rank: int,
        lora_alpha: float,
        xsa_lambda_init: float,
        enable_dynamic_qk: bool,
        enable_xsa: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.enable_xsa = enable_xsa
        self.q_gain = (
            DynamicQGain(num_heads, qk_gain_init, max_passes=max_passes, num_pos_buckets=num_pos_buckets)
            if enable_dynamic_qk
            else StaticQGain(num_heads, qk_gain_init)
        )
        self.xsa_lambda = (
            nn.Parameter(torch.full((max_passes, num_heads), xsa_lambda_init, dtype=torch.float32))
            if enable_xsa
            else None
        )
        self.q_lora = (
            nn.ModuleList([LowRankAdapter(dim, dim, lora_rank, lora_alpha) for _ in range(max_passes - 1)])
            if lora_rank > 0 and max_passes > 1
            else nn.ModuleList()
        )
        self.o_lora = (
            nn.ModuleList([LowRankAdapter(dim, dim, lora_rank, lora_alpha) for _ in range(max_passes - 1)])
            if lora_rank > 0 and max_passes > 1
            else nn.ModuleList()
        )

    def _xsa_mean(self, pass_index: int) -> float:
        if self.xsa_lambda is None:
            return 0.0
        return float(self.xsa_lambda[min(pass_index, self.xsa_lambda.size(0) - 1)].mean().item())

    def forward(
        self,
        x: Tensor,
        lane_partner: Tensor | None,
        pass_index: int,
        recur_strength: Tensor,
        telemetry: dict[str, object] | None = None,
        telemetry_key: str | None = None,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        q_proj = self.c_q(x)
        if pass_index > 0 and self.q_lora:
            q_proj = q_proj + recur_strength.to(dtype=x.dtype) * self.q_lora[min(pass_index - 1, len(self.q_lora) - 1)](x)
        q = q_proj.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        gain = self.q_gain(x, lane_partner, pass_index, recur_strength)
        if telemetry is not None and telemetry_key is not None:
            telemetry[f"{telemetry_key}_pass{pass_index}"] = (
                float(gain.float().mean().item()),
                float(gain.float().std(unbiased=False).item()),
                float(gain.float().amin().item()),
                float(gain.float().amax().item()),
                self._xsa_mean(pass_index),
            )
        q = q * gain[:, :, :, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.enable_xsa and self.xsa_lambda is not None:
            if self.num_kv_heads != self.num_heads:
                self_v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            else:
                self_v = v
            xsa_lambda = self.xsa_lambda[min(pass_index, self.xsa_lambda.size(0) - 1)].to(dtype=y.dtype)[None, :, None, None]
            proj_coeff = (y * self_v).sum(dim=-1, keepdim=True) / (self_v.square().sum(dim=-1, keepdim=True) + 1e-6)
            y = y - xsa_lambda * proj_coeff * self_v
        y_flat = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        out = self.proj(y_flat)
        if pass_index > 0 and self.o_lora:
            out = out + recur_strength.to(dtype=x.dtype) * self.o_lora[min(pass_index - 1, len(self.o_lora) - 1)](y_flat)
        return out


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden_dim, bias=False)
        self.proj = CastedLinear(hidden_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class RecursiveBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        rope_base: float,
        qk_gain_init: float,
        max_passes: int,
        num_pos_buckets: int,
        lora_rank: int,
        lora_alpha: float,
        xsa_lambda_init: float,
        dual_lane_enabled: bool,
        enable_dynamic_qk: bool,
        enable_xsa: bool,
        layer_index: int,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.dual_lane_enabled = dual_lane_enabled
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = DynamicCausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            max_passes=max_passes,
            num_pos_buckets=num_pos_buckets,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            xsa_lambda_init=xsa_lambda_init,
            enable_dynamic_qk=enable_dynamic_qk,
            enable_xsa=enable_xsa,
        )
        self.mlp = MLP(dim, mlp_hidden_dim)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        if dual_lane_enabled:
            self.lane_mix_attn_read = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.lane_mix_mlp_read = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.lane_mix_attn_to_a = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.lane_mix_attn_to_m = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.lane_mix_mlp_to_a = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.lane_mix_mlp_to_m = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.dual_attn_norm = RMSNorm()
            self.dual_mlp_norm = RMSNorm()
        else:
            self.register_parameter("lane_mix_attn_read", None)
            self.register_parameter("lane_mix_mlp_read", None)
            self.register_parameter("lane_mix_attn_to_a", None)
            self.register_parameter("lane_mix_attn_to_m", None)
            self.register_parameter("lane_mix_mlp_to_a", None)
            self.register_parameter("lane_mix_mlp_to_m", None)
            self.dual_attn_norm = None
            self.dual_mlp_norm = None

    def forward(
        self,
        a: Tensor,
        m: Tensor | None,
        x0: Tensor,
        pass_index: int,
        recur_strength: Tensor,
        telemetry: dict[str, object] | None = None,
        capture_lane_metrics: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        scale = recur_strength.to(dtype=a.dtype) if pass_index > 0 else a.new_tensor(1.0)
        mix = self.resid_mix.to(dtype=a.dtype)
        attn_scale = self.attn_scale.to(dtype=a.dtype)[None, None, :]
        mlp_scale = self.mlp_scale.to(dtype=a.dtype)[None, None, :]
        if not self.dual_lane_enabled or m is None:
            x = mix[0][None, None, :] * a + mix[1][None, None, :] * x0
            attn_out = scale * self.attn(
                self.attn_norm(x),
                lane_partner=None,
                pass_index=pass_index,
                recur_strength=recur_strength,
                telemetry=telemetry,
                telemetry_key=f"layer{self.layer_index}",
            )
            x = x + attn_scale * attn_out
            x = x + mlp_scale * (scale * self.mlp(self.mlp_norm(x)))
            return x, m
        a_base = mix[0][None, None, :] * a + mix[1][None, None, :] * x0
        attn_in = self.dual_attn_norm(a_base + self.lane_mix_attn_read.to(dtype=a.dtype)[None, None, :] * m)
        mlp_in = self.dual_mlp_norm(m + self.lane_mix_mlp_read.to(dtype=a.dtype)[None, None, :] * a_base)
        attn_out = scale * self.attn(
            attn_in,
            lane_partner=mlp_in,
            pass_index=pass_index,
            recur_strength=recur_strength,
            telemetry=telemetry,
            telemetry_key=f"layer{self.layer_index}",
        )
        mlp_out = scale * self.mlp(mlp_in)
        if capture_lane_metrics and telemetry is not None:
            telemetry[f"lane{self.layer_index}"] = (
                float(self.lane_mix_attn_to_a.abs().mean().item()),
                float(self.lane_mix_attn_to_m.abs().mean().item()),
                float(self.lane_mix_mlp_to_a.abs().mean().item()),
                float(self.lane_mix_mlp_to_m.abs().mean().item()),
            )
        a = (
            a_base
            + attn_scale * self.lane_mix_attn_to_a.to(dtype=a.dtype)[None, None, :] * attn_out
            + mlp_scale * self.lane_mix_mlp_to_a.to(dtype=a.dtype)[None, None, :] * mlp_out
        )
        m = (
            m
            + attn_scale * self.lane_mix_attn_to_m.to(dtype=a.dtype)[None, None, :] * attn_out
            + mlp_scale * self.lane_mix_mlp_to_m.to(dtype=a.dtype)[None, None, :] * mlp_out
        )
        return a, m


class CausalDeltaSubspace(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"delta rank must be positive, got {rank}")
        self.state_proj = CastedLinear(dim, rank, bias=False)
        # Keep the branch output as an exact no-op at init via zero delta_scale, while
        # giving delta_scale an immediate gradient so the branch is not dead on arrival.
        self.delta_basis = nn.Parameter(torch.empty(rank, dim, dtype=torch.float32))
        self.delta_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        nn.init.normal_(self.delta_basis, mean=0.0, std=1.0 / math.sqrt(dim))

    def forward(self, x: Tensor, telemetry: dict[str, object] | None = None) -> Tensor:
        prefix_sum = torch.cumsum(x.float(), dim=1)
        denom = torch.arange(1, x.size(1) + 1, device=x.device, dtype=prefix_sum.dtype).view(1, -1, 1)
        summary = prefix_sum / denom
        z = torch.tanh(self.state_proj(summary.to(dtype=x.dtype)))
        delta = z @ self.delta_basis.to(dtype=x.dtype)
        scaled_delta = self.delta_scale.to(dtype=x.dtype)[None, None, :] * delta
        if telemetry is not None:
            delta_norm = scaled_delta.float().norm(dim=-1).mean()
            hidden_norm = x.float().norm(dim=-1).mean().clamp_min(1e-6)
            telemetry["delta_ratio"] = float((delta_norm / hidden_norm).item())
        return x + scaled_delta


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        late_mlp_mult: float,
        late_mlp_start_layer: int,
        late_num_kv_heads: int,
        late_kv_start_layer: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        dyn_qk_pos_buckets: int,
        recur_start_layer: int,
        recur_num_layers: int,
        recur_extra_passes: int,
        recur_start_step: int,
        recur_warmup_steps: int,
        dual_lane_start_layer: int,
        lora_rank: int,
        lora_alpha: float,
        xsa_lambda_init: float,
        delta_rank: int,
        enable_recur: bool,
        enable_dual_lanes: bool,
        enable_dynamic_qk: bool,
        enable_xsa: bool,
        enable_delta_subspace: bool,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if recur_extra_passes < 0:
            raise ValueError(f"recur_extra_passes must be non-negative, got {recur_extra_passes}")
        if not 0 <= recur_start_layer < num_layers:
            raise ValueError(f"recur_start_layer must be in [0, {num_layers}), got {recur_start_layer}")
        if recur_num_layers <= 0:
            raise ValueError(f"recur_num_layers must be positive, got {recur_num_layers}")
        if not 0 <= dual_lane_start_layer <= num_layers:
            raise ValueError(f"dual_lane_start_layer must be in [0, {num_layers}], got {dual_lane_start_layer}")
        if lora_rank < 0:
            raise ValueError(f"lora_rank must be non-negative, got {lora_rank}")
        if delta_rank < 0:
            raise ValueError(f"delta_rank must be non-negative, got {delta_rank}")
        if late_mlp_mult < 0:
            raise ValueError(f"late_mlp_mult must be non-negative, got {late_mlp_mult}")
        if not 0 <= late_mlp_start_layer <= num_layers:
            raise ValueError(f"late_mlp_start_layer must be in [0, {num_layers}], got {late_mlp_start_layer}")
        if late_num_kv_heads < 0:
            raise ValueError(f"late_num_kv_heads must be non-negative, got {late_num_kv_heads}")
        if late_num_kv_heads > 0 and num_heads % late_num_kv_heads != 0:
            raise ValueError(
                f"late_num_kv_heads must divide num_heads, got num_heads={num_heads} late_num_kv_heads={late_num_kv_heads}"
            )
        if not 0 <= late_kv_start_layer <= num_layers:
            raise ValueError(f"late_kv_start_layer must be in [0, {num_layers}], got {late_kv_start_layer}")
        recur_end_layer = min(recur_start_layer + recur_num_layers, num_layers)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.recur_start_step = recur_start_step
        self.recur_warmup_steps = recur_warmup_steps
        self.enable_recur = enable_recur
        self.enable_dual_lanes = enable_dual_lanes
        self.enable_dynamic_qk = enable_dynamic_qk
        self.enable_xsa = enable_xsa
        self.enable_delta_subspace = enable_delta_subspace
        self.mlp_mult = mlp_mult
        self.late_mlp_mult = late_mlp_mult
        self.late_mlp_start_layer = late_mlp_start_layer
        self.num_kv_heads = num_kv_heads
        self.late_num_kv_heads = late_num_kv_heads
        self.late_kv_start_layer = late_kv_start_layer
        self.recur_extra_passes = recur_extra_passes if enable_recur else 0
        self.recur_start_layer = recur_start_layer
        self.recur_end_layer = recur_end_layer
        self.dual_lane_start_layer = dual_lane_start_layer
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        max_passes = max(1, self.recur_extra_passes + 1)
        mlp_hidden_dims = [
            resolve_mlp_hidden_dim(
                model_dim,
                late_mlp_mult if late_mlp_mult > 0 and i >= late_mlp_start_layer else mlp_mult,
            )
            for i in range(num_layers)
        ]
        kv_heads_per_layer = [
            late_num_kv_heads if late_num_kv_heads > 0 and i >= late_kv_start_layer else num_kv_heads
            for i in range(num_layers)
        ]
        self.mlp_hidden_dims = tuple(mlp_hidden_dims)
        self.kv_heads_per_layer = tuple(kv_heads_per_layer)
        self.blocks = nn.ModuleList(
            [
                RecursiveBlock(
                    model_dim,
                    num_heads,
                    kv_heads_per_layer[i],
                    mlp_hidden_dims[i],
                    rope_base,
                    qk_gain_init,
                    max_passes=max_passes,
                    num_pos_buckets=dyn_qk_pos_buckets,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    xsa_lambda_init=xsa_lambda_init,
                    dual_lane_enabled=(enable_dual_lanes and i >= dual_lane_start_layer),
                    enable_dynamic_qk=enable_dynamic_qk,
                    enable_xsa=enable_xsa,
                    layer_index=i,
                )
                for i in range(num_layers)
            ]
        )
        if enable_dual_lanes:
            self.lane_merge = nn.Parameter(torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float())
        else:
            self.register_parameter("lane_merge", None)
        self.delta_adapter = CausalDeltaSubspace(model_dim, delta_rank) if enable_delta_subspace and delta_rank > 0 else None
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.register_buffer("recur_gate", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("recur_active", torch.tensor(0, dtype=torch.int32), persistent=False)
        self._base_execution_plan = self._build_execution_plan(num_layers, include_recur=False)
        self._recur_execution_plan = self._build_execution_plan(num_layers, include_recur=True)
        self._active_execution_plan = self._base_execution_plan
        self._init_weights()

    def _build_execution_plan(self, num_layers: int, *, include_recur: bool) -> list[tuple[int, int, bool, int]]:
        plan: list[tuple[int, int, bool, int]] = []
        for layer_idx in range(num_layers):
            skip_idx = layer_idx - self.num_encoder_layers if layer_idx >= self.num_encoder_layers else -1
            plan.append((layer_idx, 0, layer_idx < self.num_encoder_layers, skip_idx))
            if include_recur and self.recur_extra_passes > 0 and layer_idx == self.recur_end_layer - 1:
                for pass_index in range(1, self.recur_extra_passes + 1):
                    for recur_idx in range(self.recur_start_layer, self.recur_end_layer):
                        plan.append((recur_idx, pass_index, False, -1))
        return plan

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def set_step(self, step: int) -> None:
        if self.recur_extra_passes <= 0:
            self.recur_gate.fill_(0.0)
            self.recur_active.zero_()
            self._active_execution_plan = self._base_execution_plan
            return
        if step < self.recur_start_step:
            gate = 0.0
            active = 0
        elif self.recur_warmup_steps <= 0:
            gate = 1.0
            active = 1
        else:
            gate = min(max((step - self.recur_start_step) / max(self.recur_warmup_steps, 1), 0.0), 1.0)
            active = 1
        self.recur_gate.fill_(float(gate))
        self.recur_active.fill_(active)
        self._active_execution_plan = self._recur_execution_plan if active else self._base_execution_plan

    def _merge_lanes(self, a: Tensor, m: Tensor | None) -> Tensor:
        if m is None or self.lane_merge is None:
            return a
        mix = self.lane_merge.to(dtype=a.dtype)
        return mix[0][None, None, :] * a + mix[1][None, None, :] * m

    def _telemetry_layers(self) -> tuple[int, int, int]:
        last = len(self.blocks) - 1
        mid = len(self.blocks) // 2
        return (0, mid, last)

    def mlp_schedule_summary(self) -> str:
        if not self.mlp_hidden_dims:
            return "mlp_hidden:none"
        if len(set(self.mlp_hidden_dims)) == 1:
            return f"mlp_hidden:all={self.mlp_hidden_dims[0]}"
        return (
            f"mlp_hidden:base={self.mlp_hidden_dims[0]} "
            f"late={self.mlp_hidden_dims[-1]} start={self.late_mlp_start_layer}"
        )

    def kv_schedule_summary(self) -> str:
        if not self.kv_heads_per_layer:
            return "kv_heads:none"
        if len(set(self.kv_heads_per_layer)) == 1:
            return f"kv_heads:all={self.kv_heads_per_layer[0]}"
        return (
            f"kv_heads:base={self.kv_heads_per_layer[0]} "
            f"late={self.kv_heads_per_layer[-1]} start={self.late_kv_start_layer}"
        )

    def _run_backbone(
        self, input_ids: Tensor, *, telemetry: dict[str, object] | None = None
    ) -> tuple[Tensor, dict[str, object] | None]:
        a = self.tok_emb(input_ids)
        a = F.rms_norm(a, (a.size(-1),))
        x0 = a
        m: Tensor | None = None
        skips: list[Tensor] = []
        watch_layers = set(self._telemetry_layers()) if telemetry is not None else set()
        for layer_idx, pass_index, store_skip, skip_idx in self._active_execution_plan:
            if skip_idx >= 0 and skips:
                a = a + self.skip_weights[skip_idx].to(dtype=a.dtype)[None, None, :] * skips.pop()
            if self.enable_dual_lanes and m is None and layer_idx >= self.dual_lane_start_layer:
                m = a
            recur_strength = self.recur_gate if pass_index > 0 else self.recur_gate.new_tensor(1.0)
            capture_lane_metrics = telemetry is not None and layer_idx == self.dual_lane_start_layer
            block_telemetry = telemetry if (layer_idx in watch_layers or capture_lane_metrics) else None
            a, m = self.blocks[layer_idx](
                a,
                m,
                x0,
                pass_index=pass_index,
                recur_strength=recur_strength,
                telemetry=block_telemetry,
                capture_lane_metrics=capture_lane_metrics,
            )
            if store_skip:
                skips.append(a)
        x = self._merge_lanes(a, m)
        if self.delta_adapter is not None:
            x = self.delta_adapter(x, telemetry=telemetry)
        elif telemetry is not None:
            telemetry["delta_ratio"] = 0.0
        if telemetry is not None:
            telemetry["recur_gate"] = float(self.recur_gate.item())
            telemetry["recur_active"] = int(self.recur_active.item())
            if self.delta_adapter is None:
                telemetry["delta_enabled"] = 0
            else:
                telemetry["delta_enabled"] = 1
        return x, telemetry

    def collect_telemetry(self, input_ids: Tensor) -> str:
        telemetry: dict[str, object] = {}
        with torch.no_grad():
            self._run_backbone(input_ids, telemetry=telemetry)
        layer_bits = []
        for layer_idx in self._telemetry_layers():
            pass_entries = []
            pass_index = 0
            while True:
                key = f"layer{layer_idx}_pass{pass_index}"
                if key not in telemetry:
                    break
                mean_gain, std_gain, min_gain, max_gain, xsa_mean = telemetry[key]
                pass_entries.append(
                    f"p{pass_index}:gain={mean_gain:.3f}/{std_gain:.3f}/{min_gain:.3f}/{max_gain:.3f}:xsa={xsa_mean:.3f}"
                )
                pass_index += 1
            if pass_entries:
                layer_bits.append(f"l{layer_idx}[{'|'.join(pass_entries)}]")
        lane = telemetry.get(f"lane{self.dual_lane_start_layer}")
        lane_msg = "disabled"
        if isinstance(lane, tuple):
            lane_msg = (
                f"a2a={lane[0]:.3f},a2m={lane[1]:.3f},m2a={lane[2]:.3f},m2m={lane[3]:.3f}"
            )
        return (
            f"telemetry:recur_gate={telemetry.get('recur_gate', 0.0):.3f} "
            f"recur_active={telemetry.get('recur_active', 0)} "
            f"delta_ratio={telemetry.get('delta_ratio', 0.0):.4f} "
            f"lane_mix={lane_msg} "
            f"{' '.join(layer_bits)}"
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x, _ = self._run_backbone(input_ids)
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


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        late_mlp_mult=args.late_mlp_mult,
        late_mlp_start_layer=args.late_mlp_start_layer,
        late_num_kv_heads=args.late_num_kv_heads,
        late_kv_start_layer=args.late_kv_start_layer,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        dyn_qk_pos_buckets=args.dyn_qk_pos_buckets,
        recur_start_layer=args.recur_start_layer,
        recur_num_layers=args.recur_num_layers,
        recur_extra_passes=args.recur_extra_passes,
        recur_start_step=args.recur_start_step,
        recur_warmup_steps=args.recur_warmup_steps,
        dual_lane_start_layer=args.dual_lane_start_layer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        xsa_lambda_init=args.xsa_lambda_init,
        delta_rank=args.delta_rank,
        enable_recur=args.enable_recur,
        enable_dual_lanes=args.enable_dual_lanes,
        enable_dynamic_qk=args.enable_dynamic_qk,
        enable_xsa=args.enable_xsa,
        enable_delta_subspace=args.enable_delta_subspace,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    needs_find_unused_parameters = distributed
    model: nn.Module = (
        DDP(
            compiled_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=needs_find_unused_parameters,
        )
        if distributed
        else compiled_model
    )

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    body_named_params = [
        (name, p)
        for name, p in base_model.named_parameters()
        if not name.startswith("tok_emb.") and not name.startswith("lm_head.")
    ]
    matrix_params = [
        p
        for name, p in body_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in body_named_params
        if p.ndim != 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(
        "candidate:recursive_reference "
        f"recur_start_layer={args.recur_start_layer} recur_num_layers={args.recur_num_layers} "
        f"recur_extra_passes={args.recur_extra_passes} dual_lane_start_layer={args.dual_lane_start_layer} "
        f"lora_rank={args.lora_rank} delta_rank={args.delta_rank} "
        f"enable_recur={int(args.enable_recur)} enable_dual_lanes={int(args.enable_dual_lanes)} "
        f"enable_dynamic_qk={int(args.enable_dynamic_qk)} enable_xsa={int(args.enable_xsa)} "
        f"enable_delta_subspace={int(args.enable_delta_subspace)}"
    )
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"late_num_kv_heads:{args.late_num_kv_heads} late_kv_start:{args.late_kv_start_layer} "
        f"{base_model.kv_schedule_summary()}"
    )
    log0(
        f"mlp_shape:model_dim:{args.model_dim} base_mult:{args.mlp_mult:.3f} "
        f"late_mult:{args.late_mlp_mult:.3f} late_start:{args.late_mlp_start_layer} "
        f"{base_model.mlp_schedule_summary()}"
    )
    log0(
        f"recursive_stack:start:{args.recur_start_layer} len:{args.recur_num_layers} extra_passes:{args.recur_extra_passes} "
        f"gate_start:{args.recur_start_step} gate_warmup:{args.recur_warmup_steps} dual_lane_start:{args.dual_lane_start_layer} "
        f"enable_recur:{int(args.enable_recur)} enable_dual_lanes:{int(args.enable_dual_lanes)}"
    )
    log0(
        f"dynamic_qk:pos_buckets:{args.dyn_qk_pos_buckets} xsa_lambda_init:{args.xsa_lambda_init} "
        f"lora_rank:{args.lora_rank} lora_alpha:{args.lora_alpha} delta_rank:{args.delta_rank} "
        f"enable_dynamic_qk:{int(args.enable_dynamic_qk)} enable_xsa:{int(args.enable_xsa)} "
        f"enable_delta_subspace:{int(args.enable_delta_subspace)}"
    )
    log0(
        f"export_int8:clip_percentile:{INT8_CLIP_PERCENTILE} keep_float_max_numel:{INT8_KEEP_FLOAT_MAX_NUMEL} "
        f"force_passthrough:{','.join(INT8_FORCE_PASSTHROUGH_PATTERNS) or '-'} "
        f"force_int8:{','.join(INT8_FORCE_INT8_PATTERNS) or '-'}"
    )
    log0(f"ddp_find_unused_parameters:{int(needs_find_unused_parameters)}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        def run_warmup_step(sim_step: int, label: str) -> None:
            base_model.set_step(sim_step)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(label)
        for warmup_step in range(args.warmup_steps):
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                run_warmup_step(warmup_step, f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
            else:
                run_warmup_step(warmup_step, "warmup_step:silent")
        if base_model.enable_recur and args.recur_start_step > 0:
            run_warmup_step(
                args.recur_start_step + max(args.recur_warmup_steps, 1),
                "warmup_step:recur_path",
            )
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        base_model.set_step(step)
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            telemetry_ids = x[:1]
            log0(base_model.collect_telemetry(telemetry_ids))

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
        for line in build_quant_diagnostic_lines(quant_stats, top_k=INT8_LOG_TOP_TENSORS):
            log0(line)

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
