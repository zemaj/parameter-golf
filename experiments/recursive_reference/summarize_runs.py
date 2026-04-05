#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

try:
    from experiments.recursive_reference.log_export_diagnostics import parse_export_diagnostics
except ModuleNotFoundError:
    from log_export_diagnostics import parse_export_diagnostics


STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+"
    r"train_time:(?P<train_time>[0-9.]+)ms\s+step_avg:(?P<step_avg>[0-9.]+)ms$"
)
PEAK_RE = re.compile(r"^peak memory allocated:\s+(?P<alloc>\d+)\s+MiB\s+reserved:\s+(?P<reserved>\d+)\s+MiB$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_stdout(stdout_path: Path) -> dict[str, Any]:
    lines = stdout_path.read_text().splitlines()
    last_val: dict[str, Any] | None = None
    peak: dict[str, int] | None = None
    for line in lines:
        line = line.strip()
        step_match = STEP_RE.match(line)
        if step_match:
            last_val = {
                "step": int(step_match.group("step")),
                "iterations": int(step_match.group("iters")),
                "val_loss": float(step_match.group("val_loss")),
                "val_bpb": float(step_match.group("val_bpb")),
                "train_time_ms": float(step_match.group("train_time")),
                "step_avg_ms": float(step_match.group("step_avg")),
            }
            continue
        peak_match = PEAK_RE.match(line)
        if peak_match:
            peak = {
                "allocated_mib": int(peak_match.group("alloc")),
                "reserved_mib": int(peak_match.group("reserved")),
            }
    export_diag = parse_export_diagnostics(lines)
    return {
        "last_pre_roundtrip": last_val,
        "peak_memory": peak,
        "export_diagnostics": export_diag,
    }


def _extract_env(env_path: Path) -> dict[str, Any]:
    env = _load_json(env_path)
    keep = [
        "RUN_ID",
        "ENABLE_RECUR",
        "ENABLE_DUAL_LANES",
        "ENABLE_DYNAMIC_QK",
        "ENABLE_XSA",
        "ENABLE_DELTA_SUBSPACE",
        "RECUR_START_LAYER",
        "RECUR_NUM_LAYERS",
        "RECUR_EXTRA_PASSES",
        "RECUR_START_STEP",
        "RECUR_WARMUP_STEPS",
        "DUAL_LANE_START_LAYER",
        "RECUR_LORA_RANK",
        "RECUR_LORA_ALPHA",
        "DYN_QK_POS_BUCKETS",
        "MODEL_DIM",
        "MLP_MULT",
        "LATE_MLP_MULT",
        "LATE_MLP_START_LAYER",
        "LATE_NUM_KV_HEADS",
        "LATE_KV_START_LAYER",
        "QK_GAIN_INIT",
        "XSA_LAMBDA_INIT",
        "DELTA_RANK",
        "INT8_KEEP_FLOAT_MAX_NUMEL",
        "INT8_CLIP_PERCENTILE",
        "INT8_FORCE_PASSTHROUGH_PATTERNS",
        "INT8_FORCE_INT8_PATTERNS",
    ]
    return {key: env[key] for key in keep if key in env}


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def summarize_run(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    stdout_path = run_dir / "stdout.log"
    env_path = run_dir / "env.json"
    result = _load_json(result_path) if result_path.exists() else {}
    stdout = _parse_stdout(stdout_path) if stdout_path.exists() else {"last_pre_roundtrip": None, "peak_memory": None}
    env = _extract_env(env_path) if env_path.exists() else {}
    finals = {item["label"]: item for item in result.get("final_metrics", [])}
    exact = finals.get("final_int8_zlib_roundtrip_exact")
    approx = finals.get("final_int8_zlib_roundtrip")
    last_pre = stdout.get("last_pre_roundtrip")
    peak = stdout.get("peak_memory") or {}
    export_diag = stdout.get("export_diagnostics") or {}
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "returncode": result.get("returncode"),
        "latest_step": result.get("latest_step"),
        "final_int8_zlib_roundtrip_exact_val_bpb": exact.get("val_bpb") if exact else None,
        "final_int8_zlib_roundtrip_val_bpb": approx.get("val_bpb") if approx else None,
        "last_pre_roundtrip_val_bpb": last_pre.get("val_bpb") if last_pre else None,
        "last_pre_roundtrip_step": last_pre.get("step") if last_pre else None,
        "step_avg_ms": last_pre.get("step_avg_ms") if last_pre else None,
        "train_time_ms": last_pre.get("train_time_ms") if last_pre else None,
        "artifact_bytes": result.get("artifact_bytes"),
        "code_bytes": result.get("code_bytes"),
        "model_bytes": result.get("model_bytes"),
        "peak_memory_allocated_mib": peak.get("allocated_mib"),
        "peak_memory_reserved_mib": peak.get("reserved_mib"),
        "int8_diag_modes": export_diag.get("modes", []),
        "int8_diag_families": export_diag.get("families", []),
        "int8_diag_top_payload": export_diag.get("top_payload", []),
        "largest_payload_mode": export_diag.get("largest_mode"),
        "largest_payload_family": export_diag.get("largest_family"),
        "quant_gap": (
            (exact.get("val_bpb") if exact else None) - last_pre.get("val_bpb")
            if exact and last_pre
            else None
        ),
        "env": env,
    }


def add_baseline_deltas(rows: list[dict[str, Any]], baseline_name: str) -> None:
    baseline = next((row for row in rows if row["name"] == baseline_name), None)
    if baseline is None:
        raise ValueError(f"baseline run not found: {baseline_name}")
    base_bpb = baseline["final_int8_zlib_roundtrip_exact_val_bpb"]
    base_step = baseline["step_avg_ms"]
    base_bytes = baseline["artifact_bytes"]
    for row in rows:
        row["delta_vs_baseline_bpb"] = (
            row["final_int8_zlib_roundtrip_exact_val_bpb"] - base_bpb
            if row["final_int8_zlib_roundtrip_exact_val_bpb"] is not None and base_bpb is not None
            else None
        )
        row["bytes_over_baseline"] = (
            row["artifact_bytes"] - base_bytes if row["artifact_bytes"] is not None and base_bytes is not None else None
        )
        row["delta_step_avg_ms"] = (
            row["step_avg_ms"] - base_step if row["step_avg_ms"] is not None and base_step is not None else None
        )
        row["delta_bpb_per_ms"] = _safe_div(row["delta_vs_baseline_bpb"], row["delta_step_avg_ms"])
        kb_delta = row["bytes_over_baseline"] / 1024.0 if row["bytes_over_baseline"] is not None else None
        row["delta_bpb_per_kb"] = _safe_div(row["delta_vs_baseline_bpb"], kb_delta)
        row["is_baseline"] = row["name"] == baseline_name


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = [
        "rank",
        "name",
        "exact_bpb",
        "vs_base",
        "quant_gap",
        "step_avg_ms",
        "artifact_bytes",
        "top_family",
        "peak_mem_mib",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["name"],
                    _fmt(row["final_int8_zlib_roundtrip_exact_val_bpb"]),
                    _fmt(row["delta_vs_baseline_bpb"], signed=True),
                    _fmt(row["quant_gap"], signed=True),
                    _fmt(row["step_avg_ms"]),
                    _fmt_int(row["artifact_bytes"]),
                    row["largest_payload_family"] or "-",
                    _fmt_int(row["peak_memory_allocated_mib"]),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recursive-reference screening runs.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to include")
    parser.add_argument("--baseline", required=True, help="Baseline run directory name")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", help="Optional output Markdown table path")
    args = parser.parse_args()

    rows = [summarize_run(Path(run_dir).resolve()) for run_dir in args.run_dirs]
    add_baseline_deltas(rows, baseline_name=args.baseline)
    rows.sort(
        key=lambda row: (
            row["final_int8_zlib_roundtrip_exact_val_bpb"] is None,
            row["final_int8_zlib_roundtrip_exact_val_bpb"] if row["final_int8_zlib_roundtrip_exact_val_bpb"] is not None else float("inf"),
        )
    )

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"baseline": args.baseline, "rows": rows}, indent=2) + "\n")
    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, out_md)


if __name__ == "__main__":
    main()
