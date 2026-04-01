from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from pg_harness_lib.constants import DEFAULT_TARGET
from pg_harness_lib.presets import (
    build_env,
    combo_suffix,
    discover_presets,
    dump_json,
    expand_grid,
    load_json,
    materialize_run,
    parse_grid_overrides,
    parse_set_overrides,
    preset_path,
)
from pg_harness_lib.results import (
    apply_baseline,
    build_row,
    load_result_for_path,
    maybe_write_summary,
    parse_log,
    print_rows,
    resolve_target,
    sort_rows,
)
from pg_harness_lib.runpod_screening import RunpodScreeningSession


def executor_for(preset: dict[str, object], override: str | None) -> str:
    if override is not None:
        return override
    launch = preset.get("launch", {})
    if isinstance(launch, dict):
        return str(launch.get("executor", "local"))
    return "local"


def run_local(run_dir: Path, preset: dict[str, object], overrides: dict[str, str]) -> int:
    env = build_env(preset, overrides)
    command = load_json(run_dir / "command.json")["command"]
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(command, cwd=Path(run_dir).parents[2], env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc.returncode


def finalize_run(run_dir: Path, returncode: int) -> dict[str, object]:
    result = parse_log(run_dir / "stdout.log")
    result["returncode"] = returncode
    dump_json(run_dir / "result.json", result)
    return result


def execute_one(
    *,
    preset_path_value: Path,
    payload: dict[str, object],
    overrides: dict[str, str],
    run_name: str | None,
    executor_override: str | None,
    remote_session: RunpodScreeningSession | None,
) -> tuple[Path, dict[str, object], int]:
    run_dir, _, _ = materialize_run(preset_path_value.stem, payload, overrides, run_name)
    executor = executor_for(payload, executor_override)
    if executor == "local":
        returncode = run_local(run_dir, payload, overrides)
    elif executor == "runpod-screening":
        if remote_session is None:
            raise RuntimeError("Remote session is required for runpod-screening execution.")
        returncode = remote_session.run(run_dir=run_dir, preset_name=preset_path_value.stem, preset=payload, overrides=overrides)
    else:
        raise ValueError(f"Unsupported executor: {executor}")
    result = finalize_run(run_dir, returncode)
    return run_dir, result, returncode


def cmd_list(args: argparse.Namespace) -> int:
    del args
    for path in discover_presets():
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
    from pg_harness_lib.constants import TARGET_SNAPSHOTS

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
    remote_session = None
    executor = executor_for(payload, args.executor)
    if executor == "runpod-screening":
        remote_session = RunpodScreeningSession(keep_pod_running=args.keep_pod_running)
    try:
        if remote_session is None:
            run_dir, result, returncode = execute_one(
                preset_path_value=path,
                payload=payload,
                overrides=overrides,
                run_name=args.name,
                executor_override=args.executor,
                remote_session=None,
            )
        else:
            with remote_session:
                run_dir, result, returncode = execute_one(
                    preset_path_value=path,
                    payload=payload,
                    overrides=overrides,
                    run_name=args.name,
                    executor_override=args.executor,
                    remote_session=remote_session,
                )
        del run_dir
        print(json.dumps(result, indent=2, sort_keys=True))
        return returncode
    finally:
        remote_session = None


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
    baseline = apply_baseline(rows, args.baseline)
    payload = {"target": target, "baseline": baseline, "rows": rows}
    maybe_write_summary(args.summary_json, payload)
    print_rows(rows, target, args.format, baseline)
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
    executor = executor_for(payload, args.executor)

    remote_session = RunpodScreeningSession(keep_pod_running=args.keep_pod_running) if executor == "runpod-screening" and not args.materialize_only else None
    try:
        if remote_session is None:
            context = None
        else:
            context = remote_session.__enter__()
        for index, overrides in enumerate(expand_grid(base_overrides, grid_overrides), start=1):
            combo_name = f"{base_name}-{index:02d}-{combo_suffix(overrides)}"
            if args.materialize_only:
                run_dir, _, _ = materialize_run(path.stem, payload, overrides, combo_name)
                rows.append(build_row(name=run_dir.name, result={}, target=target, run_dir=run_dir, preset=path.stem, overrides=overrides))
                continue
            run_dir, result, returncode = execute_one(
                preset_path_value=path,
                payload=payload,
                overrides=overrides,
                run_name=combo_name,
                executor_override=args.executor,
                remote_session=context,
            )
            rows.append(
                build_row(
                    name=run_dir.name,
                    result=result,
                    target=target,
                    run_dir=run_dir,
                    preset=path.stem,
                    overrides=overrides,
                    returncode=returncode,
                )
            )
            if returncode != 0 and overall_returncode == 0:
                overall_returncode = returncode
            if returncode != 0 and args.stop_on_error:
                break
    finally:
        if remote_session is not None:
            remote_session.__exit__(None, None, None)

    rows = sort_rows(rows)
    baseline = apply_baseline(rows, args.baseline)
    payload_out = {
        "target": target,
        "baseline": baseline,
        "preset": path.stem,
        "materialize_only": bool(args.materialize_only),
        "rows": rows,
    }
    maybe_write_summary(args.summary_json, payload_out)
    print_rows(rows, target, args.format, baseline)
    return overall_returncode


def cmd_study(args: argparse.Namespace) -> int:
    target = resolve_target(args.target)
    common_overrides = parse_set_overrides(args.set or [])
    rows = []
    overall_returncode = 0
    remote_session: RunpodScreeningSession | None = None
    try:
        for preset_name in args.presets:
            path = preset_path(preset_name)
            payload = load_json(path)
            executor = executor_for(payload, args.executor)
            if executor == "runpod-screening":
                if remote_session is None:
                    remote_session = RunpodScreeningSession(keep_pod_running=args.keep_pod_running)
                    remote_session.__enter__()
                session = remote_session
            else:
                session = None

            run_dir, result, returncode = execute_one(
                preset_path_value=path,
                payload=payload,
                overrides=common_overrides,
                run_name=args.name_prefix + "-" + path.stem if args.name_prefix else None,
                executor_override=args.executor,
                remote_session=session,
            )
            rows.append(
                build_row(
                    name=run_dir.name,
                    result=result,
                    target=target,
                    run_dir=run_dir,
                    preset=path.stem,
                    overrides=common_overrides,
                    returncode=returncode,
                )
            )
            if returncode != 0 and overall_returncode == 0:
                overall_returncode = returncode
            if returncode != 0 and args.stop_on_error:
                break
    finally:
        if remote_session is not None:
            remote_session.__exit__(None, None, None)

    rows = sort_rows(rows)
    baseline_name = args.baseline if args.baseline is not None else (args.presets[0] if args.presets else None)
    baseline = apply_baseline(rows, baseline_name)
    payload = {"target": target, "baseline": baseline, "presets": args.presets, "rows": rows}
    maybe_write_summary(args.summary_json, payload)
    print_rows(rows, target, args.format, baseline)
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
    run_parser.add_argument("--executor", choices=("local", "runpod-screening"))
    run_parser.add_argument("--keep-pod-running", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    parse_parser = sub.add_parser("parse-log", help="Parse metrics from a log file")
    parse_parser.add_argument("log_path")
    parse_parser.set_defaults(func=cmd_parse)

    compare_parser = sub.add_parser("compare", help="Compare logs or run directories")
    compare_parser.add_argument("paths", nargs="+")
    compare_parser.add_argument("--target", default=DEFAULT_TARGET)
    compare_parser.add_argument("--baseline")
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
    sweep_parser.add_argument("--target", default="none")
    sweep_parser.add_argument("--baseline")
    sweep_parser.add_argument("--format", choices=("table", "json"), default="table")
    sweep_parser.add_argument("--summary-json")
    sweep_parser.add_argument("--executor", choices=("local", "runpod-screening"))
    sweep_parser.add_argument("--keep-pod-running", action="store_true")
    sweep_parser.set_defaults(func=cmd_sweep)

    study_parser = sub.add_parser("study", help="Run several presets sequentially and compare them in one summary")
    study_parser.add_argument("presets", nargs="+")
    study_parser.add_argument("--name-prefix")
    study_parser.add_argument("--set", action="append")
    study_parser.add_argument("--stop-on-error", action="store_true")
    study_parser.add_argument("--target", default="none")
    study_parser.add_argument("--baseline", help="Baseline preset/name to compare the study against. Defaults to the first preset.")
    study_parser.add_argument("--format", choices=("table", "json"), default="table")
    study_parser.add_argument("--summary-json")
    study_parser.add_argument("--executor", choices=("local", "runpod-screening"))
    study_parser.add_argument("--keep-pod-running", action="store_true")
    study_parser.set_defaults(func=cmd_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
