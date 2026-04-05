# Testing The Recursive Reference

This note is for screening and optimizing `experiments/recursive_reference/train_gpt.py` on the current `1xH100` harness path.

Treat this trainer as a research scaffold, not a contest-ready run.

## What To Optimize For

When comparing candidates, the ranking metric is:

- `final_int8_zlib_roundtrip_exact val_bpb`

Do not rank variants by float validation alone. If a run looks better before export but worse after int8+zlib round-trip, treat it as a regression.

## What To Record

Capture these lines from logs for every run:

- `final_int8_zlib_roundtrip_exact val_bpb`
- `final_int8_zlib_roundtrip val_bpb`
- last pre-roundtrip `val_bpb`
- `Serialized model int8+zlib`
- `Total submission size int8+zlib`
- `int8_diag:modes`
- `int8_diag:families`
- `int8_diag:top_payload`
- `peak memory allocated`
- final `step_avg`
- all env vars controlling the recursive stack

Useful derived fields:

- `quant_gap = final_int8_zlib_roundtrip_exact_val_bpb - float_final_val_bpb`
- `bytes_over_baseline`
- `ms_per_step`
- `delta_bpb_per_ms`
- `delta_bpb_per_kb`

## Important Current Behavior

### `RECUR_START_STEP` is now a true compute schedule

This experiment copy now keeps separate no-recurrence and recurrence execution plans.

Before `RECUR_START_STEP`, extra recurrent passes are not executed at all. After the switch step, recurrence uses the replay plan and `recur_gate` still controls the learning warmup.

Implication:

- delayed recurrence is now a real wall-clock knob
- timing sweeps should compare both quality and `step_avg`

### Corrective matrices now route through the control-style path

This experiment copy treats these as control-style parameters so they route to Adam and stay fp32 during training:

- `q_lora.*.a`
- `q_lora.*.b`
- `o_lora.*.a`
- `o_lora.*.b`
- `delta_adapter.state_proj.weight`
- `delta_adapter.delta_basis`

The export-side caution still stands: many of these tensors are already small enough to avoid aggressive int8 treatment, so the sharper concern is training policy, not serialization fidelity.

### Bytes matter more than params

Several new small tensors are already protected from aggressive int8 treatment by `CONTROL_TENSOR_NAME_PATTERNS` and `INT8_KEEP_FLOAT_MAX_NUMEL`.

That helps fidelity, but it means parameter count alone is not a good proxy for final artifact size.

The trainer now also logs compact export diagnostics:

- `int8_diag:modes` groups payload by export mode such as `int8_per_row`, `fp16_passthrough`, and `fp32_passthrough`
- `int8_diag:families` groups payload by tensor family such as embeddings, attention, MLP, control tensors, LoRA, and delta
- `int8_diag:top_payload` lists the largest payload contributors directly from the exported state

Use those lines when a run is over cap or when a training-good candidate degrades after round-trip.

For a quick readout from a completed run directory or `stdout.log`, use:

```bash
python3 experiments/recursive_reference/inspect_export_diag.py <run_dir_or_log>
```

For a quick CPU-only estimate of parameter count, raw state bytes, and late-layer schedules before launching a run, use:

```bash
python3 experiments/recursive_reference/inspect_model_layout.py --preset <preset_name>
```

### Cross-boundary recurrence needs extra care

If the recurrence window crosses the encoder/decoder split on an odd-depth model, the main pass of the first decoder block still gets its normal skip connection, but the replayed pass does not re-consume that skip-like signal.

That does not make hinge-crossing inherently wrong, but it does make the semantics less clean for ablations because the repeated decoder pass is not operating on the same kind of state as the main decoder pass.

### The delta subspace no longer starts dead

This experiment copy uses a small random `delta_basis` with zero `delta_scale`, so the branch still starts as an exact no-op while `delta_scale` gets a nonzero learning signal immediately.

## Suggested Test Ladder

### Supported preset matrix

Use the matrix below as a clean default queue. Every preset maps directly to existing trainer knobs; nothing here assumes unsupported code paths.

Anchors:

1. `recursive_reference_backbone_only_screening`
2. `recursive_reference_backbone_q15_screening`

Late FFN / KV byte recovery:

1. `recursive_reference_backbone_q15_late_mlp1875_screening`
2. `recursive_reference_backbone_q15_late_mlp175_screening`
3. `recursive_reference_backbone_q15_late_kv2_screening`
4. `recursive_reference_backbone_q15_late_combo_screening`

Low-overhead quality probes:

1. `recursive_reference_backbone_q15_dynamic_qk_screening`
2. `recursive_reference_backbone_q15_dynamic_qk_xsa_screening`

Targeted export probes:

1. `recursive_reference_backbone_q15_export_clip99990_screening`
2. `recursive_reference_backbone_q15_export_control_int8_screening`

Recurrence 2.0 probes:

1. `recursive_reference_backbone_q15_recur4_r2_delayed_screening`
2. `recursive_reference_backbone_q15_recur34_r2_delayed_screening`
3. `recursive_reference_recur2_l45_delayed_screening`

The most important ordering rule is still:

1. keep `recursive_reference_backbone_q15_screening` as the quality anchor
2. try byte-recovery variants before reintroducing extra compute
3. treat `recursive_reference_recur2_l45_delayed_screening` as a semantics probe, not a clean ablation

### Stage 0: smoke

Objective:

- confirm compile stability
- confirm no NaNs
- confirm final int8 round-trip eval completes

Suggested command:

```bash
RUN_ID=recursive_reference_smoke \
ITERATIONS=50 \
WARMUP_STEPS=5 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 \
NUM_LAYERS=11 \
QK_GAIN_INIT=4.0 \
ENABLE_RECUR=1 \
ENABLE_DUAL_LANES=1 \
ENABLE_DYNAMIC_QK=1 \
ENABLE_XSA=1 \
ENABLE_DELTA_SUBSPACE=1 \
RECUR_START_LAYER=4 \
RECUR_NUM_LAYERS=2 \
RECUR_EXTRA_PASSES=1 \
RECUR_START_STEP=3000 \
RECUR_WARMUP_STEPS=500 \
DUAL_LANE_START_LAYER=7 \
RECUR_LORA_RANK=2 \
RECUR_LORA_ALPHA=0.6 \
DYN_QK_POS_BUCKETS=8 \
XSA_LAMBDA_INIT=1.0 \
DELTA_RANK=16 \
python3 experiments/recursive_reference/train_gpt.py
```

### Stage 1: throughput profiling

Objective:

- measure actual `step_avg`
- measure memory cost by feature family

Use short runs with:

- `VAL_LOSS_EVERY=0`
- fixed batch and sequence settings
- identical environment except for the feature under test

Suggested matrix:

1. backbone only
2. dynamic QK only
3. dynamic QK + XSA
4. recurrence window
5. recurrence + dual lanes
6. recurrence + dual lanes + delta subspace

### Stage 2: short ranking runs

Objective:

- eliminate obviously bad ideas before 600 second runs

Rank candidates by:

1. round-trip `val_bpb`
2. final size
3. speed

### Stage 3: full 600 second candidates

Only promote variants that already look good on:

- round-trip metric
- step time
- final artifact bytes

Promising stripped-backbone follow-ups are now preset-backed:

- `recursive_reference_backbone_q15_late_mlp1875_screening`
- `recursive_reference_backbone_q15_late_mlp175_screening`
- `recursive_reference_backbone_q15_late_kv2_screening`
- `recursive_reference_backbone_q15_late_combo_screening`

Those are still the highest-confidence path because they trim dense late layers rather than shrinking the whole model or adding more routing overhead.

## Unsupported paths that need code work first

Do not invent presets for these yet:

- deep-only XSA or layer-band XSA
- explicit late-only / band-limited QK shaping controls
- KV sharing across adjacent layers
- per-matrix mixed-bit search with automated selection
- hidden ladder fusion
- fiber-bank side-state paths

## Implemented Controls

The current trainer now supports:

- feature flags: `ENABLE_RECUR`, `ENABLE_DUAL_LANES`, `ENABLE_DYNAMIC_QK`, `ENABLE_XSA`, and `ENABLE_DELTA_SUBSPACE`
- true delayed recurrence without paying replay compute before the switch step
- clean zero-rank ablations for `RECUR_LORA_RANK` and `DELTA_RANK`
- fractional `MLP_MULT` plus late-layer taper controls via `LATE_MLP_MULT` and `LATE_MLP_START_LAYER`
- targeted export overrides via `INT8_FORCE_PASSTHROUGH_PATTERNS` and `INT8_FORCE_INT8_PATTERNS`
- lightweight parseable telemetry for recurrence state and representative feature usage
