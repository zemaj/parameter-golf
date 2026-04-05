from __future__ import annotations

from collections import defaultdict
from typing import Any


_CONTROL_SNIPPETS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "xsa_lambda",
    "lane_mix",
    "lane_merge",
)


def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.2f}MiB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f}KiB"
    return f"{nbytes}B"


def _fmt_ratio(baseline_bytes: int, payload_bytes: int) -> str:
    if payload_bytes <= 0:
        return "-"
    return f"{baseline_bytes / payload_bytes:.2f}x"


def classify_tensor_family(name: str) -> str:
    if name.startswith("tok_emb.") or name.startswith("lm_head."):
        return "embed"
    if "q_lora" in name or "o_lora" in name:
        return "lora"
    if "delta_" in name or "state_proj" in name:
        return "delta"
    if any(snippet in name for snippet in _CONTROL_SNIPPETS):
        return "control"
    if ".attn." in name or name.endswith(("c_q.weight", "c_k.weight", "c_v.weight")):
        return "attn"
    if ".mlp." in name or name.endswith(("fc.weight", "proj.weight")):
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def build_quant_diagnostic_lines(quant_stats: dict[str, Any], *, top_k: int = 8) -> list[str]:
    entries = list(quant_stats.get("entries", []))
    if not entries:
        return []

    mode_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "payload_bytes": 0, "baseline_bytes": 0})
    family_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "payload_bytes": 0, "baseline_bytes": 0})

    for entry in entries:
        mode = str(entry["mode"])
        family = str(entry["family"])
        payload_bytes = int(entry["payload_bytes"])
        baseline_bytes = int(entry["baseline_bytes"])

        mode_totals[mode]["tensors"] += 1
        mode_totals[mode]["payload_bytes"] += payload_bytes
        mode_totals[mode]["baseline_bytes"] += baseline_bytes

        family_totals[family]["tensors"] += 1
        family_totals[family]["payload_bytes"] += payload_bytes
        family_totals[family]["baseline_bytes"] += baseline_bytes

    def summarize_bucket(name: str, bucket: dict[str, int]) -> str:
        return (
            f"{name}:payload={_fmt_bytes(bucket['payload_bytes'])}"
            f":base={_fmt_bytes(bucket['baseline_bytes'])}"
            f":ratio={_fmt_ratio(bucket['baseline_bytes'], bucket['payload_bytes'])}"
            f":tensors={bucket['tensors']}"
        )

    mode_parts = [
        summarize_bucket(name, bucket)
        for name, bucket in sorted(mode_totals.items(), key=lambda item: (-item[1]["payload_bytes"], item[0]))
    ]
    family_parts = [
        summarize_bucket(name, bucket)
        for name, bucket in sorted(family_totals.items(), key=lambda item: (-item[1]["payload_bytes"], item[0]))
    ]

    lines = [
        "int8_diag:modes " + " | ".join(mode_parts),
        "int8_diag:families " + " | ".join(family_parts),
    ]

    top_entries = sorted(entries, key=lambda entry: (-int(entry["payload_bytes"]), str(entry["name"])))[: max(0, top_k)]
    if top_entries:
        parts = []
        for entry in top_entries:
            parts.append(
                f"{entry['name']}[{entry['mode']}/{entry['family']}]"
                f"={_fmt_bytes(int(entry['payload_bytes']))}"
                f"/{_fmt_bytes(int(entry['baseline_bytes']))}"
            )
        lines.append("int8_diag:top_payload " + " ; ".join(parts))

    return lines
