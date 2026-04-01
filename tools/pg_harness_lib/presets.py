from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pg_harness_lib.constants import PRESETS_DIR, ROOT, RUNS_DIR


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


def build_command(root_dir: Path, preset: dict[str, Any]) -> list[str]:
    script = Path(preset["script"])
    if not script.is_absolute():
        script = (root_dir / script).resolve()

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


def shell_env_command(tracked: dict[str, str], command: list[str]) -> str:
    env_bits = [f"{key}={shlex.quote(value)}" for key, value in sorted(tracked.items())]
    cmd_bits = " ".join(shlex.quote(part) for part in command)
    return f"{' '.join(env_bits)} {cmd_bits}" if env_bits else cmd_bits


def command_shell_line(tracked: dict[str, str], command: list[str], workdir: Path) -> str:
    return f"cd {shlex.quote(str(workdir))}\n{shell_env_command(tracked, command)}\n"


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "run"


def combo_suffix(overrides: dict[str, str]) -> str:
    if not overrides:
        return "base"
    bits = [f"{key.lower()}-{value}" for key, value in sorted(overrides.items())]
    return slugify("__".join(bits))


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


def materialize_run(
    preset_name: str,
    preset: dict[str, Any],
    env_overrides: dict[str, str],
    run_name: str | None,
) -> tuple[Path, dict[str, str], list[str]]:
    tracked = tracked_env(preset, env_overrides)
    env = build_env(preset, env_overrides)
    command = build_command(ROOT, preset)
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
        "executor": preset.get("launch", {}).get("executor", "local"),
    }
    dump_json(run_dir / "run.json", metadata)
    dump_json(run_dir / "env.json", tracked)
    dump_json(run_dir / "command.json", {"command": command})
    (run_dir / "command.sh").write_text(command_shell_line(tracked, command, ROOT), encoding="utf-8")
    return run_dir, env, command

