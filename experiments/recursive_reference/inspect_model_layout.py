#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_gpt  # noqa: E402
from export_diagnostics import classify_tensor_family  # noqa: E402


def _parse_set(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _load_preset_env(preset_name: str) -> dict[str, str]:
    path = REPO_ROOT / "experiments" / "presets" / f"{preset_name}.json"
    payload = json.loads(path.read_text())
    return {str(k): str(v) for k, v in payload.get("env", {}).items()}


def _instantiate_args(env_updates: dict[str, str]) -> train_gpt.Hyperparameters:
    args = train_gpt.Hyperparameters()
    attr_to_env = {
        "run_id": "RUN_ID",
        "seed": "SEED",
        "val_batch_size": "VAL_BATCH_SIZE",
        "val_loss_every": "VAL_LOSS_EVERY",
        "train_log_every": "TRAIN_LOG_EVERY",
        "iterations": "ITERATIONS",
        "warmdown_iters": "WARMDOWN_ITERS",
        "warmup_steps": "WARMUP_STEPS",
        "train_batch_tokens": "TRAIN_BATCH_TOKENS",
        "train_seq_len": "TRAIN_SEQ_LEN",
        "max_wallclock_seconds": "MAX_WALLCLOCK_SECONDS",
        "qk_gain_init": "QK_GAIN_INIT",
        "dyn_qk_pos_buckets": "DYN_QK_POS_BUCKETS",
        "recur_start_layer": "RECUR_START_LAYER",
        "recur_num_layers": "RECUR_NUM_LAYERS",
        "recur_extra_passes": "RECUR_EXTRA_PASSES",
        "recur_start_step": "RECUR_START_STEP",
        "recur_warmup_steps": "RECUR_WARMUP_STEPS",
        "dual_lane_start_layer": "DUAL_LANE_START_LAYER",
        "lora_rank": "RECUR_LORA_RANK",
        "lora_alpha": "RECUR_LORA_ALPHA",
        "xsa_lambda_init": "XSA_LAMBDA_INIT",
        "delta_rank": "DELTA_RANK",
        "enable_recur": "ENABLE_RECUR",
        "enable_dual_lanes": "ENABLE_DUAL_LANES",
        "enable_dynamic_qk": "ENABLE_DYNAMIC_QK",
        "enable_xsa": "ENABLE_XSA",
        "enable_delta_subspace": "ENABLE_DELTA_SUBSPACE",
        "vocab_size": "VOCAB_SIZE",
        "num_layers": "NUM_LAYERS",
        "num_kv_heads": "NUM_KV_HEADS",
        "model_dim": "MODEL_DIM",
        "num_heads": "NUM_HEADS",
        "mlp_mult": "MLP_MULT",
        "late_mlp_mult": "LATE_MLP_MULT",
        "late_mlp_start_layer": "LATE_MLP_START_LAYER",
        "late_num_kv_heads": "LATE_NUM_KV_HEADS",
        "late_kv_start_layer": "LATE_KV_START_LAYER",
        "tie_embeddings": "TIE_EMBEDDINGS",
        "rope_base": "ROPE_BASE",
        "logit_softcap": "LOGIT_SOFTCAP",
        "embed_lr": "EMBED_LR",
        "head_lr": "HEAD_LR",
        "tied_embed_lr": "TIED_EMBED_LR",
        "tied_embed_init_std": "TIED_EMBED_INIT_STD",
        "matrix_lr": "MATRIX_LR",
        "scalar_lr": "SCALAR_LR",
        "muon_momentum": "MUON_MOMENTUM",
        "muon_backend_steps": "MUON_BACKEND_STEPS",
        "muon_momentum_warmup_start": "MUON_MOMENTUM_WARMUP_START",
        "muon_momentum_warmup_steps": "MUON_MOMENTUM_WARMUP_STEPS",
        "beta1": "BETA1",
        "beta2": "BETA2",
        "adam_eps": "ADAM_EPS",
        "grad_clip_norm": "GRAD_CLIP_NORM",
        "data_path": "DATA_PATH",
        "tokenizer_path": "TOKENIZER_PATH",
    }
    for attr, env_key in attr_to_env.items():
        if env_key not in env_updates:
            continue
        current = getattr(args, attr)
        raw = env_updates[env_key]
        if isinstance(current, bool):
            value = bool(int(raw))
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(args, attr, value)
    return args


def _build_model(args: train_gpt.Hyperparameters) -> train_gpt.GPT:
    return train_gpt.GPT(
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
    )


def _fmt_mib(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.3f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect recursive_reference model parameter layout without launching training.")
    parser.add_argument("--preset", help="Preset name from experiments/presets without .json")
    parser.add_argument("--set", action="append", default=[], help="Override env values like KEY=VALUE")
    args_ns = parser.parse_args()

    env_updates: dict[str, str] = {}
    if args_ns.preset:
        env_updates.update(_load_preset_env(args_ns.preset))
    env_updates.update(_parse_set(args_ns.set))

    args = _instantiate_args(env_updates)
    model = _build_model(args)

    family_bytes: dict[str, int] = {}
    family_params: dict[str, int] = {}
    total_bytes = 0
    total_params = 0
    for name, param in model.state_dict().items():
        nbytes = param.numel() * param.element_size()
        family = classify_tensor_family(name)
        total_bytes += nbytes
        total_params += param.numel()
        family_bytes[family] = family_bytes.get(family, 0) + nbytes
        family_params[family] = family_params.get(family, 0) + int(param.numel())

    print(f"preset={args_ns.preset or '-'}")
    print(f"model_params={total_params}")
    print(f"raw_state_bytes={total_bytes} ({_fmt_mib(total_bytes)})")
    print(f"mlp_schedule={model.mlp_schedule_summary()}")
    print(f"kv_schedule={model.kv_schedule_summary()}")
    print("families:")
    for family, nbytes in sorted(family_bytes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {family}: params={family_params[family]} bytes={nbytes} ({_fmt_mib(nbytes)})")


if __name__ == "__main__":
    main()
