# RESULTS.md

## Purpose

This file tracks the live baseline-first screening evidence so new work starts from the strongest current signal instead of revisiting already-pruned branches.

## Current Matched 1xH100 Baseline

- Baseline run family: root [train_gpt.py](/Users/zemaj/www/parameter-golf/train_gpt.py) on `sp1024`
- Fresh matched screening baseline:
  - exact round-trip `val_bpb`: `1.34248996`
  - total size int8+zlib: `13,171,907`
  - final `step_avg`: `488.66 ms`

## Active Winner

- `last_layer_q30`
  - parent: root baseline
  - change: initialize only the final block's existing learned `q_gain` to `3.0`
  - implementation: [last_layer_qgain.py](/Users/zemaj/www/parameter-golf/experiments/last_layer_qgain/last_layer_qgain.py)
  - exact round-trip `val_bpb`: `1.32490829`
  - total size int8+zlib: `13,616,806`
  - final `step_avg`: `443.93 ms`

## Q-Gain Family Results

### Fixed Global Q-Gain

- `baseline_q20`
  - exact round-trip `val_bpb`: `1.33705309`
  - total size int8+zlib: `12,966,949`
  - final `step_avg`: `504.50 ms`

- `baseline_q30`
  - exact round-trip `val_bpb`: `1.33262099`
  - total size int8+zlib: `13,046,201`
  - final `step_avg`: `497.27 ms`

- `baseline_q40`
  - exact round-trip `val_bpb`: `1.34041343`
  - total size int8+zlib: `13,103,829`
  - final `step_avg`: `496.41 ms`

Read:

- global Q-gain matters
- the curve is not monotonic
- `3.0` is clearly better than `2.0` or `4.0`
- but global sharpening is still much weaker than a final-layer-only change

### Fixed Late Q-Gain

- `last_layer_q30`
  - exact round-trip `val_bpb`: `1.32490829`
  - total size int8+zlib: `13,616,806`
  - final `step_avg`: `443.93 ms`

- `last_layer_q40`
  - exact round-trip `val_bpb`: `1.34041343`
  - total size int8+zlib: `13,103,829`
  - final `step_avg`: `496.41 ms`

- `late_qgain_ramp_l5_t35`
  - exact round-trip `val_bpb`: `1.33695889`
  - total size int8+zlib: `13,274,114`
  - final `step_avg`: `478.30 ms`

Read:

- the win is not "make late layers sharper in general"
- the best signal is concentrated in the final layer
- spreading the gain backward across layers `5..8` hurts versus a single strong final-layer prior

### Adaptive Q-Gain

- `adaptive_qk_rms_pos`
  - exact round-trip `val_bpb`: `1.34423650`
  - status: negative

- `adaptive_qk_late_headgroup`
  - exact round-trip `val_bpb`: `1.33712560`

- `adaptive_qk_late_headgroup_late7`
  - exact round-trip `val_bpb`: `1.33682574`

- `adaptive_qk_late_headgroup_late8`
  - exact round-trip `val_bpb`: `1.33649155`
  - total size int8+zlib: `12,989,925`
  - final `step_avg`: `500.62 ms`

- `adaptive_qk_late_headgroup_late8_clamp10`
  - exact round-trip `val_bpb`: `1.34238080`
  - total size int8+zlib: `12,672,202`
  - final `step_avg`: `512.73 ms`

- `adaptive_qk_late_headgroup_late8_q30cap`
  - exact round-trip `val_bpb`: `1.34303670`
  - total size int8+zlib: `12,909,340`
  - final `step_avg`: `522.77 ms`

Read:

- adaptive Q-gain can beat the baseline
- but every adaptive version lost to simple fixed `last_layer_q30`
- stronger adaptive headroom did not help
- capping the adaptive branch at the fixed winner's `q=3.0` also did not help
- current read: on this line, adaptation itself is the losing ingredient

## Main Takeaways

- The strongest current idea is a very small one: give the final attention block a stronger `q_gain` initialization and let training refine it.
- The model seems to want more selective attention right before logits, not a more expressive attention controller throughout the stack.
- Broadening that change across more layers, or making it dynamic, consistently made results worse.
- The best current parent for new work is the root baseline plus `last_layer_q30`, not the adaptive-QK branch.

## Most Likely Interpretation

- `q_gain` in the baseline acts like a per-head attention temperature because it is applied after Q/K normalization and RoPE, immediately before SDPA.
- The final layer likely benefits from a much sharper retrieval / cleanup step than the earlier layers do.
- Earlier layers still seem to benefit from softer, more mixing-friendly attention.
- A fixed final-layer prior works better than an adaptive controller because it does not spend budget learning when or how much to sharpen.

## Next Branch: Hidden Ladder

This is the next novel branch worth trying.

Minimal baseline-first version:

- keep the root baseline stack unchanged
- save the hidden states from the last `2-3` blocks
- after the decoder stack and before `final_norm`, fuse those late hidden states into the final hidden state through a tiny low-rank correction
- use zero-init fusion gates so step `0` remains exactly the baseline

Clean insertion point:

- in [train_gpt.py](/Users/zemaj/www/parameter-golf/train_gpt.py), after the decoder loop in `GPT.forward`
- fuse into `x`
- then continue with the existing `final_norm` and tied-head path

First ladder sweep to try:

- taps: last `2` layers, then last `3` layers
- rank: `2`, then `4`
- shared output basis
- zero-init gates

## Hidden Ladder Results

- `hidden_ladder_last2_r2`
  - parent: root baseline
  - change: late-state low-rank fusion from layers `7,8` back into the final hidden state before `final_norm`
  - implementation: [hidden_ladder.py](/Users/zemaj/www/parameter-golf/experiments/hidden_ladder/hidden_ladder.py)
  - exact round-trip `val_bpb`: `1.34026714`
  - total size int8+zlib: `12,998,882`
  - final `step_avg`: `506.26 ms`

Read:

- hidden ladder is a real positive versus the fresh baseline (`-0.00222`)
- but this first `last2, rank2` version is only mildly positive
- it is still well behind `last_layer_q30`

- `hidden_ladder_last3_r2`
  - parent: root baseline
  - change: same rank-2 fusion, but broaden late taps from `7,8` to `6,7,8`
  - exact round-trip `val_bpb`: `1.34258512`
  - total size int8+zlib: `13,085,006`
  - final `step_avg`: `514.31 ms`

Updated read:

- `last2, rank2` is the best hidden-ladder result so far
- broadening the ladder to three late taps made the final result worse, despite slightly better early checkpoints
- current hidden-ladder evidence is mixed: there is a mild positive local signal, but not a strong or stable one yet
- hidden ladder does not currently threaten `last_layer_q30`

## How To Use This File

- Treat this as directional `1xH100` screening evidence, not leaderboard evidence.
- Prefer the freshest matched baseline when comparing a new branch.
- Before adding complexity, check whether the same effect was already tested in simpler fixed form.

## Heavy Stack Caveat

- The `records/track_non_record_16mb/2026-04-02_ScoredPosSLOT_PerSampleGPTQ_LastLayerQ30` branch should not be judged by the same `1xH100` pruning heuristics used for baseline-parent ideas.
- It is a much heavier `8xH100`-style stack with scored-position SLOT, per-sample delta/logit-bias eval machinery, full GPTQ, and XSA across all `11` layers.
- Current validation plan:
  - run a paired `8xH100` A/B on the same script and seed
  - control: `QK_GAIN_INIT=4.0`, `LAST_LAYER_QGAIN_ENABLED=0`
  - variant: `QK_GAIN_INIT=1.5`, `LAST_LAYER_QGAIN_ENABLED=1`, `LAST_LAYER_QK_GAIN_INIT=3.0`
  - stop the `8xH100` pod immediately after the comparison

## 8x Heavy-Stack Results

The heavyweight scored-position SLOT / per-sample GPTQ stack was finally tested in its intended `8xH100` regime. That changed the interpretation a lot: the branch is real and competitive on `8x`, but the late-layer q-gain tweaks did not beat the upstream-style control.

- `8x control (upstream-style q4)`
  - config: `QK_GAIN_INIT=4.0`, `LAST_LAYER_QGAIN_ENABLED=0`
  - exact round-trip: `0.92910946`
  - size: `15,569,649` bytes
  - intermediate round-trip: `1.13760890`
  - post-EMA float diagnostic: `1.1342`

- `8x last_layer_q30`
  - config: `QK_GAIN_INIT=1.5`, `LAST_LAYER_QGAIN_ENABLED=1`, `LAST_LAYER_QK_GAIN_INIT=3.0`
  - exact round-trip: `0.92992728`
  - size: `15,563,777` bytes
  - intermediate round-trip: `1.13818377`
  - post-EMA float diagnostic: `1.1348`

Read:

- On the real heavy stack, `last_layer_q30` is slightly worse than the upstream-style global `q4` control.
- The byte savings are tiny, about `5.9 KB`, so this is not a good trade.
- This confirms the earlier `1x` result was a bad proxy for the stack as a whole, but it does not rescue the last-layer q-gain idea on this branch.

Checkpoint-gated `8x` late-layer follow-ups:

- `last_layer_q40`
  - `step 4000 val_bpb: 1.2083`
  - slightly better than `q30` at the checkpoint, but still worse than the control `1.2073`
  - pruned before the expensive GPTQ/export tail

- `adaptive last layer`
  - first attempt exposed a real dtype bug with FlashAttention under compile
  - after the dtype fix, `step 4000 val_bpb: 1.2081`
  - best of the late-layer variants at the training checkpoint
  - still worse than the control `1.2073`
  - pruned before the expensive GPTQ/export tail

Takeaway:

- For this heavy `8x` stack, the upstream-style global `QK_GAIN_INIT=4.0` remains the best of the tested q-gain variants.
- The late-layer family was worth checking, but it does not currently improve the frontier stack.
