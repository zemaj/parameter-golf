from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pg_harness_lib.constants import ARTIFACT_RE, CODE_RE, DEFAULT_TARGET, FINAL_METRIC_RE, MODEL_RE, STEP_RE, TARGET_SNAPSHOTS
from pg_harness_lib.presets import dump_json, load_json


@dataclass
class FinalMetric:
    label: str
    val_loss: float
    val_bpb: float
    line_number: int
    raw_line: str


def parse_log(path: Path) -> dict[str, Any]:
    finals: list[FinalMetric] = []
    artifact_bytes: int | None = None
    model_bytes: int | None = None
    code_bytes: int | None = None
    latest_step: int | None = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        final_match = FINAL_METRIC_RE.search(line)
        if final_match:
            finals.append(
                FinalMetric(
                    label=final_match.group("label"),
                    val_loss=float(final_match.group("val_loss")),
                    val_bpb=float(final_match.group("val_bpb")),
                    line_number=idx,
                    raw_line=line,
                )
            )
        artifact_match = ARTIFACT_RE.search(line)
        if artifact_match:
            artifact_bytes = int(artifact_match.group("bytes"))
        model_match = MODEL_RE.search(line)
        if model_match:
            model_bytes = int(model_match.group("bytes"))
        code_match = CODE_RE.search(line)
        if code_match:
            code_bytes = int(code_match.group("bytes"))
        step_match = STEP_RE.search(line)
        if step_match:
            latest_step = int(step_match.group("step"))

    best_final = min(finals, key=lambda item: item.val_bpb) if finals else None
    return {
        "path": str(path),
        "latest_step": latest_step,
        "artifact_bytes": artifact_bytes,
        "model_bytes": model_bytes,
        "code_bytes": code_bytes,
        "final_metrics": [item.__dict__ for item in finals],
        "best_final": best_final.__dict__ if best_final else None,
    }


def resolve_target(target: str | None) -> dict[str, Any] | None:
    if target is None:
        return None
    raw = target.strip()
    if not raw or raw.lower() in {"none", "off", "false", "no"}:
        return None
    snapshot = TARGET_SNAPSHOTS.get(raw)
    if snapshot is not None:
        return dict(snapshot)
    try:
        value = float(raw)
    except ValueError as exc:
        known = ", ".join(sorted(TARGET_SNAPSHOTS))
        raise ValueError(f"Unknown target '{target}'. Use a float or one of: {known}") from exc
    return {"label": raw, "target_bpb": value, "as_of": None, "source": None, "notes": None}


def load_result_for_path(item: str) -> tuple[str, dict[str, Any], Path | None]:
    path = Path(item)
    if path.is_dir():
        result_path = path / "result.json"
        if result_path.exists():
            result = load_json(result_path)
        else:
            result = parse_log(path / "stdout.log")
        return path.name, result, path.resolve()
    resolved = path.resolve()
    return resolved.name, parse_log(resolved), None


def build_row(
    *,
    name: str,
    result: dict[str, Any],
    target: dict[str, Any] | None,
    run_dir: Path | None = None,
    preset: str | None = None,
    overrides: dict[str, str] | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    best = result.get("best_final") or {}
    val_bpb = best.get("val_bpb")
    delta = None
    beats_target = None
    if target is not None and val_bpb is not None:
        delta = val_bpb - target["target_bpb"]
        beats_target = val_bpb < target["target_bpb"]
    return {
        "name": name,
        "preset": preset,
        "label": best.get("label"),
        "val_bpb": val_bpb,
        "val_loss": best.get("val_loss"),
        "delta_vs_target_bpb": delta,
        "delta_vs_baseline_bpb": None,
        "beats_target": beats_target,
        "is_baseline": False,
        "artifact_bytes": result.get("artifact_bytes"),
        "latest_step": result.get("latest_step"),
        "returncode": returncode if returncode is not None else result.get("returncode"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "overrides": overrides or {},
    }


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: float("inf") if item["val_bpb"] is None else item["val_bpb"])


def apply_baseline(rows: list[dict[str, Any]], baseline: str | None) -> dict[str, Any] | None:
    if baseline is None:
        return None

    baseline_row = next(
        (row for row in rows if row.get("name") == baseline or row.get("preset") == baseline),
        None,
    )
    if baseline_row is None:
        known_values = ({row.get("name") or "" for row in rows} | {row.get("preset") or "" for row in rows}) - {""}
        known = ", ".join(sorted(known_values))
        raise ValueError(f"Unknown baseline '{baseline}'. Available names/presets: {known}")

    baseline_bpb = baseline_row.get("val_bpb")
    for row in rows:
        row["is_baseline"] = row is baseline_row
        if baseline_bpb is None or row.get("val_bpb") is None:
            row["delta_vs_baseline_bpb"] = None
        else:
            row["delta_vs_baseline_bpb"] = row["val_bpb"] - baseline_bpb

    return {
        "name": baseline_row.get("name"),
        "preset": baseline_row.get("preset"),
        "val_bpb": baseline_bpb,
    }


def format_float(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def print_rows(
    rows: list[dict[str, Any]],
    target: dict[str, Any] | None,
    output_format: str,
    baseline: dict[str, Any] | None = None,
) -> None:
    payload = {"target": target, "baseline": baseline, "rows": rows}
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if target is None:
        print("Target: none")
    else:
        detail = f"Target: {target['label']} = {target['target_bpb']:.4f} val_bpb"
        if target.get("as_of"):
            detail += f" (snapshot as of {target['as_of']})"
        print(detail)
    show_baseline = any(row.get("delta_vs_baseline_bpb") is not None or row.get("is_baseline") for row in rows)
    headers = ["rank", "name", "val_bpb"]
    if show_baseline:
        headers.append("vs_base")
    headers.extend(["delta", "status", "step", "artifact", "label"])
    table: list[list[str]] = [headers]
    for index, row in enumerate(rows, start=1):
        status = "pending"
        if row.get("returncode") is not None:
            status = "ok" if row["returncode"] == 0 else f"rc={row['returncode']}"
        if row["val_bpb"] is None and row.get("run_dir"):
            status = "materialized"
        elif row["val_bpb"] is not None and row.get("returncode") is None:
            status = "parsed"
        table.append(
            [
                str(index),
                row["name"],
                format_float(row.get("val_bpb")),
            ]
            + ([format_float(row.get("delta_vs_baseline_bpb"), signed=True)] if show_baseline else [])
            + [
                format_float(row.get("delta_vs_target_bpb"), signed=True),
                status,
                format_int(row.get("latest_step")),
                format_int(row.get("artifact_bytes")),
                row.get("label") or "-",
            ]
        )
    widths = [max(len(line[idx]) for line in table) for idx in range(len(headers))]
    for ridx, line in enumerate(table):
        print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(line)))
        if ridx == 0:
            print("  ".join("-" * width for width in widths))


def maybe_write_summary(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out_path = Path(path)
    if not out_path.is_absolute():
        from pg_harness_lib.constants import ROOT

        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, payload)
