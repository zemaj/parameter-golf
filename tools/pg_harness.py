#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "experiments" / "presets"
RUNS_DIR = ROOT / "experiments" / "runs"

FLOAT_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"
FINAL_METRIC_RE = re.compile(
    rf"^(?P<label>(?:final[^\s:]*|benchmark_[^\s:]*))[^\n]*?val_loss:(?P<val_loss>{FLOAT_RE})[^\n]*?val_bpb:(?P<val_bpb>{FLOAT_RE})",
)
ARTIFACT_RE = re.compile(r"^Total submission size(?: [^:]+)?: (?P<bytes>\d+) bytes")
MODEL_RE = re.compile(r"^Serialized model(?: [^:]+)?: (?P<bytes>\d+) bytes")
CODE_RE = re.compile(r"^Code size: (?P<bytes>\d+) bytes")
STEP_RE = re.compile(r"step:(?P<step>\d+)/(?:\d+)")

TARGET_SNAPSHOTS: dict[str, dict[str, Any]] = {
    "main-track-sota": {
        "label": "main-track-sota",
        "target_bpb": 1.1147,
        "as_of": "2026-04-01",
        "source": "https://github.com/openai/parameter-golf/blob/main/README.md",
        "notes": "Main-track best score visible in the public README leaderboard on 2026-04-01.",
    },
    "main-track-sota-2026-04-01": {
        "label": "main-track-sota-2026-04-01",
        "target_bpb": 1.1147,
        "as_of": "2026-04-01",
        "source": "https://github.com/openai/parameter-golf/blob/main/README.md",
        "notes": "Pinned snapshot of the main-track leaderboard on 2026-04-01.",
    },
}
DEFAULT_TARGET = "main-track-sota"


@dataclass
class FinalMetric:
    label: str
    val_loss: float
    val_bpb: float
    line_number: int
    raw_line: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preset_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path.resolve()
    if path.suffix != ".json":
        path = PRESETS_DIR / f"{name_or_path}.json"
    else:
        path = PRESETS_DIR / path.name
    return path.resolve()


def discover_presets() -> list[Path]:
    if not PRESETS_DIR.exists():
        return []
    return sorted(PRESETS_DIR.glob("*.json"))


def parse_set_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        overrides[key] = value
    return overrides


def parse_grid_overrides(items: list[str]) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Grid override must look like KEY=VALUE1,VALUE2, got: {item}")
        key, raw_values = item.split("=", 1)
        key = key.strip()
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not key or not values:
            raise ValueError(f"Grid override must include a key and at least one value: {item}")
        parsed.append((key, values))
    return parsed


def build_command(preset: dict[str, Any], env: dict[str, str]) -> list[str]:
    del env
    script = Path(preset["script"])
    if not script.is_absolute():
        script = (ROOT / script).resolve()

    launch = preset.get("launch", {})
    launcher = launch.get("type", "python")
    if launcher == "torchrun":
        command = [
            launch.get("binary", "torchrun"),
            "--standalone",
            f"--nproc_per_node={int(launch.get('nproc_per_node', 1))}",
            str(script),
        ]
    elif launcher == "python":
        command = [launch.get("binary", sys.executable), str(script)]
    else:
        raise ValueError(f"Unsupported launcher type: {launcher}")

    command.extend(str(arg) for arg in preset.get("args", []))
    if preset.get("command_suffix"):
        command.extend(str(arg) for arg in preset["command_suffix"])
    return command


def tracked_env(preset: dict[str, Any], overrides: dict[str, str]) -> dict[str, str]:
    tracked = {str(key): str(value) for key, value in preset.get("env", {}).items()}
    tracked.update(overrides)
    return tracked


def build_env(preset: dict[str, Any], overrides: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(tracked_env(preset, overrides))
    return env


def command_shell_line(tracked: dict[str, str], command: list[str], workdir: Path) -> str:
    env_bits = [f"{key}={shlex.quote(value)}" for key, value in sorted(tracked.items())]
    cmd_bits = " ".join(shlex.quote(part) for part in command)
    return f"cd {shlex.quote(str(workdir))}\n{' '.join(env_bits)} {cmd_bits}\n"


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "run"


def combo_suffix(overrides: dict[str, str]) -> str:
    if not overrides:
        return "base"
    bits = [f"{key.lower()}-{value}" for key, value in sorted(overrides.items())]
    return slugify("__".join(bits))


def materialize_run(
    preset_name: str,
    preset: dict[str, Any],
    env_overrides: dict[str, str],
    run_name: str | None,
) -> tuple[Path, dict[str, str], list[str]]:
    tracked = tracked_env(preset, env_overrides)
    env = build_env(preset, env_overrides)
    command = build_command(preset, env)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = slugify(run_name or preset.get("name") or preset_name)
    run_dir = RUNS_DIR / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "preset": preset_name,
        "description": preset.get("description", ""),
        "script": preset["script"],
        "created_at": timestamp,
        "cwd": str(ROOT),
        "run_dir": str(run_dir),
        "notes": preset.get("notes", ""),
    }
    dump_json(run_dir / "run.json", metadata)
    dump_json(run_dir / "env.json", tracked)
    dump_json(run_dir / "command.json", {"command": command})
    (run_dir / "command.sh").write_text(command_shell_line(tracked, command, ROOT), encoding="utf-8")
    return run_dir, env, command


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


def expand_grid(base_overrides: dict[str, str], grids: list[tuple[str, list[str]]]) -> Iterator[dict[str, str]]:
    if not grids:
        yield dict(base_overrides)
        return
    keys = [key for key, _ in grids]
    values = [grid_values for _, grid_values in grids]
    for combo in itertools.product(*values):
        overrides = dict(base_overrides)
        overrides.update({key: value for key, value in zip(keys, combo)})
        yield overrides


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
    row = {
        "name": name,
        "preset": preset,
        "label": best.get("label"),
        "val_bpb": val_bpb,
        "val_loss": best.get("val_loss"),
        "delta_vs_target_bpb": delta,
        "beats_target": beats_target,
        "artifact_bytes": result.get("artifact_bytes"),
        "latest_step": result.get("latest_step"),
        "returncode": returncode if returncode is not None else result.get("returncode"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "overrides": overrides or {},
    }
    return row


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: float("inf") if item["val_bpb"] is None else item["val_bpb"])


def format_float(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def print_rows(rows: list[dict[str, Any]], target: dict[str, Any] | None, output_format: str) -> None:
    payload = {"target": target, "rows": rows}
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
    headers = ["rank", "name", "val_bpb", "delta", "status", "step", "artifact", "label"]
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
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, payload)


def cmd_list(args: argparse.Namespace) -> int:
    del args
    presets = discover_presets()
    for path in presets:
        payload = load_json(path)
        desc = payload.get("description", "")
        print(f"{path.stem}: {desc}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = preset_path(args.preset)
    payload = load_json(path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(TARGET_SNAPSHOTS, indent=2, sort_keys=True))
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    path = preset_path(args.preset)
    payload = load_json(path)
    overrides = parse_set_overrides(args.set or [])
    run_dir, _, command = materialize_run(path.stem, payload, overrides, args.name)
    print(run_dir)
    print(" ".join(shlex.quote(part) for part in command))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    path = preset_path(args.preset)
    payload = load_json(path)
    overrides = parse_set_overrides(args.set or [])
    run_dir, env, command = materialize_run(path.stem, payload, overrides, args.name)
    log_path = run_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    result = parse_log(log_path)
    result["returncode"] = proc.returncode
    dump_json(run_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return proc.returncode


def cmd_parse(args: argparse.Namespace) -> int:
    result = parse_log(Path(args.log_path).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    target = resolve_target(args.target)
    rows = []
    for item in args.paths:
        name, result, run_dir = load_result_for_path(item)
        rows.append(build_row(name=name, result=result, target=target, run_dir=run_dir))
    rows = sort_rows(rows)
    payload = {"target": target, "rows": rows}
    maybe_write_summary(args.summary_json, payload)
    print_rows(rows, target, args.format)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    path = preset_path(args.preset)
    payload = load_json(path)
    base_overrides = parse_set_overrides(args.set or [])
    grid_overrides = parse_grid_overrides(args.grid or [])
    target = resolve_target(args.target)
    rows = []
    overall_returncode = 0
    base_name = args.name or path.stem

    for index, overrides in enumerate(expand_grid(base_overrides, grid_overrides), start=1):
        combo_name = f"{base_name}-{index:02d}-{combo_suffix(overrides)}"
        run_dir, env, command = materialize_run(path.stem, payload, overrides, combo_name)
        if args.materialize_only:
            rows.append(
                build_row(
                    name=run_dir.name,
                    result={},
                    target=target,
                    run_dir=run_dir,
                    preset=path.stem,
                    overrides=overrides,
                )
            )
            continue

        log_path = run_dir / "stdout.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        result = parse_log(log_path)
        result["returncode"] = proc.returncode
        dump_json(run_dir / "result.json", result)
        rows.append(
            build_row(
                name=run_dir.name,
                result=result,
                target=target,
                run_dir=run_dir,
                preset=path.stem,
                overrides=overrides,
                returncode=proc.returncode,
            )
        )
        if proc.returncode != 0 and overall_returncode == 0:
            overall_returncode = proc.returncode
        if proc.returncode != 0 and args.stop_on_error:
            break

    rows = sort_rows(rows)
    payload_out = {
        "target": target,
        "preset": path.stem,
        "materialize_only": bool(args.materialize_only),
        "rows": rows,
    }
    maybe_write_summary(args.summary_json, payload_out)
    print_rows(rows, target, args.format)
    return overall_returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter Golf experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List available presets")
    list_parser.set_defaults(func=cmd_list)

    show_parser = sub.add_parser("show", help="Show a preset")
    show_parser.add_argument("preset")
    show_parser.set_defaults(func=cmd_show)

    targets_parser = sub.add_parser("targets", help="List built-in leaderboard target snapshots")
    targets_parser.set_defaults(func=cmd_targets)

    materialize_parser = sub.add_parser("materialize", help="Create a run directory without executing")
    materialize_parser.add_argument("preset")
    materialize_parser.add_argument("--name")
    materialize_parser.add_argument("--set", action="append")
    materialize_parser.set_defaults(func=cmd_materialize)

    run_parser = sub.add_parser("run", help="Create a run directory and execute the command")
    run_parser.add_argument("preset")
    run_parser.add_argument("--name")
    run_parser.add_argument("--set", action="append")
    run_parser.set_defaults(func=cmd_run)

    parse_parser = sub.add_parser("parse-log", help="Parse metrics from a log file")
    parse_parser.add_argument("log_path")
    parse_parser.set_defaults(func=cmd_parse)

    compare_parser = sub.add_parser("compare", help="Compare logs or run directories")
    compare_parser.add_argument("paths", nargs="+")
    compare_parser.add_argument("--target", default=DEFAULT_TARGET)
    compare_parser.add_argument("--format", choices=("table", "json"), default="table")
    compare_parser.add_argument("--summary-json")
    compare_parser.set_defaults(func=cmd_compare)

    sweep_parser = sub.add_parser("sweep", help="Materialize or run a cartesian grid of overrides for one preset")
    sweep_parser.add_argument("preset")
    sweep_parser.add_argument("--name")
    sweep_parser.add_argument("--set", action="append")
    sweep_parser.add_argument("--grid", action="append")
    sweep_parser.add_argument("--materialize-only", action="store_true")
    sweep_parser.add_argument("--stop-on-error", action="store_true")
    sweep_parser.add_argument("--target", default=DEFAULT_TARGET)
    sweep_parser.add_argument("--format", choices=("table", "json"), default="table")
    sweep_parser.add_argument("--summary-json")
    sweep_parser.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
