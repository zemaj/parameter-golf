# Recursive Reference

This experiment adds a self-contained trainer variant under `experiments/` for a mid-stack recurrence idea with:

- configurable extra passes over a layer window,
- a late dual-lane residual pathway,
- dynamic per-head/per-pass/per-position QK gain,
- XSA-style novelty projection in attention,
- pass-specific low-rank adapters on `Q` and output projections,
- a causal low-rank output delta subspace.

The trainer lives at `experiments/recursive_reference/train_gpt.py` and is based on the current root `train_gpt.py` baseline rather than a submission record.

Testing guidance for the current `1xH100` screening flow lives in `experiments/recursive_reference/TESTING.md`.

Useful ready-made screening presets are grouped into a small matrix below so each branch stays interpretable.

## Main env vars

- `DYN_QK_POS_BUCKETS=8`
- `MODEL_DIM=512`
- `MLP_MULT=2.0`
- `LATE_MLP_MULT=0`
- `LATE_MLP_START_LAYER=11`
- `LATE_NUM_KV_HEADS=0`
- `LATE_KV_START_LAYER=11`
- `ENABLE_RECUR=1`
- `ENABLE_DUAL_LANES=1`
- `ENABLE_DYNAMIC_QK=1`
- `ENABLE_XSA=1`
- `ENABLE_DELTA_SUBSPACE=1`
- `RECUR_START_LAYER=4`
- `RECUR_NUM_LAYERS=2`
- `RECUR_EXTRA_PASSES=1`
- `RECUR_START_STEP=3000`
- `RECUR_WARMUP_STEPS=500`
- `DUAL_LANE_START_LAYER=7`
- `RECUR_LORA_RANK=2`
- `RECUR_LORA_ALPHA=0.6`
- `XSA_LAMBDA_INIT=1.0`
- `DELTA_RANK=16`
- `INT8_FORCE_PASSTHROUGH_PATTERNS=`
- `INT8_FORCE_INT8_PATTERNS=`

## Example

```bash
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

## Notes

- The experiment keeps the baseline `resid_mix`, `attn_scale`, and `mlp_scale` controls active after the dual-lane path turns on, so later blocks do not silently bypass those learned controls.
- `RECUR_START_STEP` now controls a real compute-saving switch: before the start step, the model uses a no-recurrence execution plan rather than executing zero-gated replay passes.
- Zero-rank ablations are supported: `RECUR_LORA_RANK=0` removes recurrent LoRA adapters and `DELTA_RANK=0` disables the delta subspace when `ENABLE_DELTA_SUBSPACE=1`.
- `MLP_MULT` now supports fractional values, and `LATE_MLP_MULT` plus `LATE_MLP_START_LAYER` can taper only the last few layers instead of shrinking every block uniformly.
- `LATE_NUM_KV_HEADS` plus `LATE_KV_START_LAYER` can taper KV capacity only in the last few layers, which is a cleaner byte trade than shrinking the whole model width.
- Corrective matrices for `q_lora`, `o_lora`, `delta_basis`, and `state_proj` are treated as control-style parameters, so they route through Adam and stay fp32 during training.
- Export policy can now be steered more surgically with `INT8_FORCE_PASSTHROUGH_PATTERNS` and `INT8_FORCE_INT8_PATTERNS`, which override the small-tensor passthrough threshold on matching tensor names.
- A recurrence window that crosses the encoder/decoder hinge is defensible, but the current execution plan makes the replayed first decoder block operate on a different kind of state than its main pass because the replay does not re-consume the decoder skip. That is an interpretation issue more than an obvious bug.
- The delta subspace now initializes with a random basis and zero output scale so it starts as an exact no-op while still giving `delta_scale` a learning signal immediately.
- Lightweight parseable `telemetry:` lines are emitted at `TRAIN_LOG_EVERY` for recurrence state, representative QK gains, XSA strength, lane-mix magnitudes, and delta ratio.
- Export now also logs `int8_diag:*` lines for payload mode/family breakdown and top payload tensors.
- This is an experiment path, not a tuned or challenge-ready record submission.

## Screening Preset Matrix

### Anchors

- `recursive_reference_backbone_only_screening`
  Best legal stripped-backbone anchor from the last batch. Use this when judging whether a new legal candidate is actually worth keeping.
- `recursive_reference_backbone_q15_screening`
  Best raw stripped-backbone anchor from the last batch. Use this for “preserve q15 quality while recovering bytes” sweeps.
- `recursive_reference_screening`
  Original full recursive-stack candidate. Keep this only as the high-complexity reference point.

### Late FFN / KV byte recovery

- `recursive_reference_backbone_q15_late_mlp1875_screening`
  Mild late MLP taper: last four MLP blocks go from `1024` hidden units to `960`.
- `recursive_reference_backbone_q15_late_mlp175_screening`
  Stronger late MLP taper: last four MLP blocks go from `1024` to `896`.
- `recursive_reference_backbone_q15_late_kv2_screening`
  Late KV taper only: last three layers go from `4` KV heads to `2`.
- `recursive_reference_backbone_q15_late_combo_screening`
  Combines the mild late MLP taper with late KV taper.

### Low-overhead quality probes that are already supported

- `recursive_reference_backbone_q15_dynamic_qk_screening`
  Nearest supported approximation to the requested QK-shaping path: global dynamic QK only.
- `recursive_reference_backbone_q15_dynamic_qk_xsa_screening`
  Nearest supported approximation to “QK plus deep-only XSA”: global dynamic QK plus global XSA.

### Targeted export probes that are already supported

- `recursive_reference_backbone_q15_export_clip99990_screening`
  Narrow global clip-percentile export probe on top of the raw-best `q15` backbone.
- `recursive_reference_backbone_q15_export_control_int8_screening`
  Targeted control-tensor export probe on top of the raw-best `q15` backbone.

### Recurrence 2.0 probes

- `recursive_reference_backbone_q15_recur4_r2_delayed_screening`
  Single delayed replay block at layer 4, with recurrence-specific LoRA enabled and all other auxiliary features off.
- `recursive_reference_backbone_q15_recur34_r2_delayed_screening`
  Delayed two-layer pre-hinge replay window for the cleanest recurrence semantics.
- `recursive_reference_recur2_l45_delayed_screening`
  Delayed hinge-adjacent replay window at layers 4-5. Supported, but harder to interpret because replayed decoder passes do not re-consume the decoder skip.

## Unsupported Paths

These ideas are not presetized because the current trainer does not expose a clean knob for them yet:

- late-layer-only or band-limited `ENABLE_XSA`
- explicit layer-wise or head-wise QK schedules beyond the existing per-layer learned dynamic-QK modules
- cross-layer or adjacent-layer KV sharing
- automatic greedy mixed-bit export search across selected matrices
- hidden ladder / ladder-fusion branches
- fiber-bank side-state geometry
