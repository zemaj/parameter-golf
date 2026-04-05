# Recursive Reference Matrix

This matrix turns the current recursive-reference situation into a concrete run plan.

Two facts drive the structure:

- the stripped 11-layer backbone is the asset
- `backbone_q15` is the raw-best result and only misses the cap by a small amount

So the matrix is split into two main goals:

1. recover roughly `71 KB` around the strong `q15` backbone
2. probe cheap hidden-structure ideas that do not materially blow up `step_avg`

Always rank by `final_int8_zlib_roundtrip_exact val_bpb`.

## Current Anchors

- `baseline_clean`: `1.33151323`, `13,576,812` bytes
- `backbone_only`: `1.31914022`, `15,649,910` bytes, current best legal result
- `backbone_q15`: `1.31534529`, `16,070,854` bytes, current best raw result and over the cap by `70,854` bytes

The goal of the next phase is to preserve the `backbone_q15` quality profile while pulling it back under `16,000,000` bytes. If that fails, the fallback legal anchor to beat remains `backbone_only`.

## Phase 0: Re-anchor The Batch

Run these first in every fresh screening batch:

1. `baseline_screening`
2. `recursive_reference_backbone_only_screening`
3. `recursive_reference_backbone_q15_screening`

Why:

- screening variance is real, even on the same pod family
- promotion decisions should be made against fresh matched anchors
- `backbone_q15` is strong enough that tiny regressions or improvements matter

Promotion rule:

- only treat a small delta as real if the re-run `backbone_only` stays close to its existing anchor

Prune rule:

- if the fresh `backbone_only` re-run lands materially worse than about `1.3200`, treat the batch as noisy and do not over-promote tiny wins

## Phase 1: Legalize `q15`

Run these first, in order:

1. `recursive_reference_backbone_q15_late_mlp1875_screening`
2. `recursive_reference_backbone_q15_late_kv2_screening`
3. `recursive_reference_backbone_q15_late_combo_screening`

Why:

- they preserve the strong `q15` backbone instead of shrinking the whole model
- they trim late dense blocks, which is where the raw state bytes actually live
- they are the highest-upside supported byte-recovery variants in the current code

Promotion rule:

- promote any candidate that is under `16,000,000` bytes and beats `backbone_only`
- also keep any near-legal candidate at or below about `1.3170` exact and no larger than about `16,040,000` bytes for Phase 3 export surgery

Prune rule:

- prune any candidate that is still over cap and no better than `backbone_only`
- prune any candidate with clearly worse `step_avg` and no byte win

## Phase 2: Cheap hidden-structure probes

These are the lowest-cost modeling probes that are already supported:

1. `recursive_reference_backbone_q15_dynamic_qk_screening`
2. `recursive_reference_backbone_q15_dynamic_qk_xsa_screening`

Why:

- these are the closest supported approximations to the proposed late-QK / deep-XSA path
- they avoid recurrence, dual lanes, and delta overhead
- they can show whether the stripped q15 backbone still has room for cheap quality recovery

Promotion rule:

- only keep these if they remain within a small speed tax and show a real round-trip gain on the best legal or near-legal q15-adjacent parent

Prune rule:

- prune if they add overhead without at least about `0.0010` exact improvement

## Phase 3: Export surgery around the best q15-adjacent run

Use these only if the strong q15-like architecture still misses the byte cap:

1. `recursive_reference_backbone_q15_export_clip99990_screening`
2. `recursive_reference_backbone_q15_export_control_int8_screening`

Why:

- previous broad export forcing was too blunt
- these are narrow, supported probes using the new `int8_diag:*` tooling
- they are meant to answer whether selective export policy can save the last bytes without wrecking round-trip quality

Use `inspect_export_diag.py` after each run to see whether payload is still dominated by dense MLP/attention tensors or by passthrough families.

Promotion rule:

- keep export-policy changes only if they save meaningful bytes while holding the round-trip hit to roughly `<= 0.0015` versus the structural parent

Prune rule:

- prune anything that saves very few bytes or behaves like the earlier blunt `qsmall` attempt

## Phase 4: Recurrence 2.0 approximations

These are the closest currently supported recurrence retries:

1. `recursive_reference_backbone_q15_recur4_r2_delayed_screening`
2. `recursive_reference_backbone_q15_recur34_r2_delayed_screening`

Why:

- they keep recurrence tiny and delayed
- they use rank-2 recurrent LoRA
- they avoid bundling recurrence with dual lanes, dynamic QK, XSA, and delta

Important:

- these are only approximations of the proposed recurrence 2.0 path
- current code does not yet support progressive pass growth, explicit contractive pass weights, or error-feedback recurrence

Promotion rule:

- recurrence only stays alive if it beats the stripped non-recurrent backbone family on round-trip metric without a large step-time penalty

Prune rule:

- prune any recurrence retry that drifts above about `1.3205` exact or pays a clear step-time premium without compensating quality

## Not Yet Supported

These ideas are interesting, but the current codebase does not support them cleanly enough to preset yet:

- per-layer or deep-only XSA controls
- per-layer/head-wise QK shaping beyond the current global dynamic QK module
- recurrence error feedback or progressive pass schedules
- hinge-aware macroblock recurrence
- late-FFN taper schedules beyond the current single late multiplier
- adjacent-layer KV sharing in the late tail
- hidden ladder / fiber-bank style side-state architectures

Those need code changes first. They should not be approximated by fake presets.

## Code-Needed Follow-On Matrix

If the supported matrix above fails to produce a legal q15-adjacent winner, the next code-backed matrix should be built in this order:

1. late MLP taper schedules such as `992, 992, 960, 928`
2. late-tail KV sharing across the last adjacent pair
3. selective export sweeps driven by `int8_diag:*` payload families
4. deep-only XSA and piecewise late-QK gains
5. recurrence 2.0 with explicit contractive pass scaling and error feedback
6. hidden-ladder or fiber-bank side-state architectures

Those are intentionally separated from the supported preset batches above so we do not blur "ready to run now" with "needs implementation first."

## Suggested Run Batches

Byte recovery batch:

```bash
python3 tools/pg_harness.py study \
  recursive_reference_backbone_q15_late_mlp1875_screening \
  recursive_reference_backbone_q15_late_kv2_screening \
  recursive_reference_backbone_q15_late_combo_screening \
  --executor runpod-screening \
  --keep-pod-running
```

Cheap hidden-structure batch:

```bash
python3 tools/pg_harness.py study \
  recursive_reference_backbone_q15_dynamic_qk_screening \
  recursive_reference_backbone_q15_dynamic_qk_xsa_screening \
  --executor runpod-screening \
  --keep-pod-running
```

Export surgery batch:

```bash
python3 tools/pg_harness.py study \
  recursive_reference_backbone_q15_export_clip99990_screening \
  recursive_reference_backbone_q15_export_control_int8_screening \
  --executor runpod-screening \
  --keep-pod-running
```

Recurrence retry batch:

```bash
python3 tools/pg_harness.py study \
  recursive_reference_backbone_q15_recur4_r2_delayed_screening \
  recursive_reference_backbone_q15_recur34_r2_delayed_screening \
  --executor runpod-screening \
  --keep-pod-running
```
