from __future__ import annotations

import re
from typing import Any


_BYTE_RE = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>B|KiB|MiB)$")


def _parse_bytes(text: str) -> int:
    match = _BYTE_RE.match(text)
    if match is None:
        raise ValueError(f"unsupported byte string: {text}")
    value = float(match.group("value"))
    unit = match.group("unit")
    factor = {"B": 1, "KiB": 1024, "MiB": 1024 * 1024}[unit]
    return int(round(value * factor))


def _parse_bucket(segment: str) -> dict[str, Any]:
    parts = segment.split(":")
    if len(parts) != 5:
        raise ValueError(f"unsupported bucket segment: {segment}")
    name = parts[0]
    kv = {}
    for item in parts[1:]:
        key, value = item.split("=", 1)
        kv[key] = value
    return {
        "name": name,
        "payload_bytes": _parse_bytes(kv["payload"]),
        "baseline_bytes": _parse_bytes(kv["base"]),
        "ratio": None if kv["ratio"] == "-" else float(kv["ratio"].removesuffix("x")),
        "tensors": int(kv["tensors"]),
    }


def parse_bucket_line(line: str, prefix: str) -> list[dict[str, Any]]:
    head = f"{prefix} "
    if not line.startswith(head):
        return []
    payload = line[len(head) :].strip()
    if not payload:
        return []
    return [_parse_bucket(segment.strip()) for segment in payload.split(" | ")]


def parse_top_payload_line(line: str) -> list[dict[str, Any]]:
    prefix = "int8_diag:top_payload "
    if not line.startswith(prefix):
        return []
    payload = line[len(prefix) :].strip()
    if not payload:
        return []
    entries: list[dict[str, Any]] = []
    for segment in payload.split(" ; "):
        left, sizes = segment.split("=", 1)
        path_and_group, payload_size = sizes.split("/", 1)
        name, group = left.rsplit("[", 1)
        mode, family = group.removesuffix("]").split("/", 1)
        entries.append(
            {
                "name": name,
                "mode": mode,
                "family": family,
                "payload_bytes": _parse_bytes(path_and_group),
                "baseline_bytes": _parse_bytes(payload_size),
            }
        )
    return entries


def parse_export_diagnostics(lines: list[str]) -> dict[str, Any]:
    modes: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    top_payload: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("int8_diag:modes "):
            modes = parse_bucket_line(line, "int8_diag:modes")
        elif line.startswith("int8_diag:families "):
            families = parse_bucket_line(line, "int8_diag:families")
        elif line.startswith("int8_diag:top_payload "):
            top_payload = parse_top_payload_line(line)
    largest_mode = max(modes, key=lambda item: item["payload_bytes"])["name"] if modes else None
    largest_family = max(families, key=lambda item: item["payload_bytes"])["name"] if families else None
    return {
        "modes": modes,
        "families": families,
        "top_payload": top_payload,
        "largest_mode": largest_mode,
        "largest_family": largest_family,
    }
