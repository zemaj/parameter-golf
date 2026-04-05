#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from log_export_diagnostics import parse_export_diagnostics


def _load_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _resolve_log_path(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if path.is_dir():
        return path / "stdout.log"
    return path


def _row_for_path(path: Path) -> dict[str, Any]:
    lines = _load_lines(path)
    parsed = parse_export_diagnostics(lines)
    return {
        "path": str(path),
        "largest_mode": parsed["largest_mode"],
        "largest_family": parsed["largest_family"],
        "modes": parsed["modes"],
        "families": parsed["families"],
        "top_payload": parsed["top_payload"],
    }


def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.2f}MiB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f}KiB"
    return f"{nbytes}B"


def _print_text(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(row["path"])
        if not row["modes"] and not row["families"] and not row["top_payload"]:
            print("  no int8 export diagnostics found")
            continue
        print(f"  largest_mode: {row['largest_mode'] or '-'}")
        print(f"  largest_family: {row['largest_family'] or '-'}")
        if row["families"]:
            print("  families:")
            for item in row["families"]:
                print(
                    f"    {item['name']}: payload={_fmt_bytes(item['payload_bytes'])} "
                    f"base={_fmt_bytes(item['baseline_bytes'])} ratio={item['ratio']:.2f}x tensors={item['tensors']}"
                )
        if row["top_payload"]:
            print("  top_payload:")
            for item in row["top_payload"]:
                print(
                    f"    {item['name']} [{item['mode']}/{item['family']}] "
                    f"{_fmt_bytes(item['payload_bytes'])}/{_fmt_bytes(item['baseline_bytes'])}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect int8 export diagnostics from recursive-reference logs.")
    parser.add_argument("paths", nargs="+", help="Run directories or stdout.log paths")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    rows = [_row_for_path(_resolve_log_path(path_str)) for path_str in args.paths]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return
    _print_text(rows)


if __name__ == "__main__":
    main()
