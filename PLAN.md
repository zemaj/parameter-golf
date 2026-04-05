# Baseline-First Exploration Plan

This plan replaces the previous `11L`-centered iteration loop.

## Goal

Find a novel, low-overhead idea that improves on the current root baseline without depending on the already-familiar `11L` basin. Use the root baseline at `train_gpt.py` as the parent, add one small idea at a time, and only promote ideas that show real screening signal.

## Operating Rules

- Start from the root baseline, not the `recursive_reference` stack.
- Keep stack changes small and isolate one idea per experiment.
- Screen on matched `1xH100` runs first.
- Promote to `8xH100` only when a branch shows a real signal or clear frontier relevance.
- Kill branches quickly when they add overhead without meaningful `final_int8_zlib_roundtrip_exact val_bpb` improvement.

## Immediate Run Order

Start every fresh `adaptive_qk` batch with a matched baseline re-anchor.

1. `baseline_screening`
2. `adaptive_qk_rms_pos_screening`
3. `baseline_screening` again only if the adaptive result is close enough to matter

Why:

- we only have one matched baseline point for this new branch
- screening variance is real
- the first adaptive-QK probe should be judged against a fresh anchor, not stale memory

Stop / go rule:

- if `adaptive_qk_rms_pos_screening` is clearly worse than the fresh baseline, pivot immediately to the structured late-layer adaptive-QK variant
- if it is roughly tied, still continue to the structured late-layer adaptive-QK variant
- if it shows a real gain, keep adaptive-QK as the active mainline and refine it before exploring other branches

## Branch Order

### 1. Adaptive QK

This is the active mainline branch.

Why:

- the baseline already learns per-head Q gain
- prior repo evidence suggests QK gain matters
- a dynamic attention thermostat is plausible and low-overhead

Initial test:

- preserve the baseline architecture and training loop
- replace static per-head Q gain with a multiplicative controller that depends on:
  - position bucket
  - token hidden-state RMS
- keep the controller zero-init so the model starts exactly at the baseline behavior

First refinement order:

1. `adaptive_qk_rms_pos_screening`
2. structured late-layer head-group gain
3. structured late-layer head-group gain + position buckets
4. only then compare whether the richer tokenwise RMS-driven controller is actually worth keeping

Success criteria:

- improves matched `1xH100` exact round-trip `val_bpb` with little or no step-time penalty

### 2. Hidden Ladder

Only pursue this after the adaptive-QK structured variants have been checked.

Add tiny late hidden-state taps that write low-rank corrections into the final hidden state before the tied output head.

Constraints:

- no extra attention passes
- zero-init fusion gates
- rank-first sweep (`2`, `4`, `8`)

### 3. Eval / Compression Side

Only pursue this after the adaptive-QK branch is either clearly positive or clearly exhausted.

Run a separate branch for score-first causal compression helpers.

Initial candidates:

- tiny entropy-gated unigram / bigram residual cache
- tiny hashed causal cache blended into logits

These should stay separate from train-side architecture work.

### 4. Clean Recurrence

Only after the lighter branches, and only if adaptive-QK and hidden-ladder do not produce a stronger mainline.

Constraints:

- baseline parent
- repeat one middle layer or one middle pair once
- delayed activation only
- no dual lanes, no delta branch, no broad recurrence bundle

### 5. Fiber Bank

Only if hidden ladder shows signal.

Goal:

- a tiny side-state that every layer can write to and late layers can read from
- no full-width multi-lane branches

## Immediate Next Step

Re-run the baseline, then run the first adaptive-QK screening preset:

1. `baseline_screening`
2. `adaptive_qk_rms_pos_screening`

Then implement the structured late-layer adaptive-QK variants before moving to hidden-ladder, recurrence, or eval-side branches.

## Progress Log

- `2026-04-02`: Created `experiments/adaptive_qk/` as the first baseline-first branch.
- `2026-04-02`: Added `adaptive_qk_rms_pos_screening` as the first isolation test.
- `2026-04-02`: Local validation passed:
  - preset resolution via `pg_harness show`
  - Python compile checks
  - CPU construction and int8 quantize/dequantize smoke
  - exact init-equivalence check against the root baseline (`loss_delta 0.0`)
- `2026-04-02`: Next active step is a matched `1xH100` batch:
  1. `baseline_screening`
  2. `adaptive_qk_rms_pos_screening`
- `2026-04-02`: First remote launch attempt failed before training started because the screening harness tried to create a fresh pod and hit a Runpod capacity error (`There are no instances currently available`).
- `2026-04-02`: Next retry is to force the harness onto the existing pod `z6l3ylqkwcbhu9` so the batch does not depend on new-capacity allocation.
- `2026-04-02`: The old pod `z6l3ylqkwcbhu9` was no longer resumable on the API side. Remote testing is now migrating to the new pod `xrerq1wjjbmiwy`.
- `2026-04-02`: The matched remote batch was relaunched on `xrerq1wjjbmiwy`.
- `2026-04-02`: Current remote blocker is dataset bootstrap on the new pod: `python3 data/cached_challenge_fineweb.py --variant sp1024` is still running, so baseline training has not started yet.
- `2026-04-02`: Added the next two structured adaptive-QK variants while waiting on the remote batch:
  - `adaptive_qk_late_headgroup_screening`
  - `adaptive_qk_late_headgroup_pos_screening`
- `2026-04-02`: Local validation passed for both structured variants:
  - preset resolution via `pg_harness show`
  - Python compile checks
  - CPU construction and int8 quantize/dequantize smoke
  - exact init-equivalence check against the root baseline (`loss_delta 0.0`)
- `2026-04-02`: The new pod finished bootstrapping the `sp1024` dataset cache (`80` train shards, `1` val shard), and the first `baseline_screening` leg is now actively training on `xrerq1wjjbmiwy`.
- `2026-04-02`: Fresh matched baseline completed on `xrerq1wjjbmiwy`:
  - exact round-trip `val_bpb`: `1.34248996`
  - total size int8+zlib: `13,171,907`
  - final `step_avg`: `488.66 ms`
  - stopped at wall clock step `1228`
- `2026-04-02`: The `adaptive_qk_rms_pos_screening` leg has started on the same pod and is now the active run under observation.
- `2026-04-02`: `adaptive_qk_rms_pos_screening` completed and is a negative result against the fresh matched baseline:
  - exact round-trip `val_bpb`: `1.34423650`
  - delta vs baseline: `+0.00174654` worse
  - total size int8+zlib: `13,093,438`
  - final `step_avg`: `497.18 ms`
- `2026-04-02`: Decision: drop the RMS+position tokenwise adaptive-QK variant and move to the structured late-layer adaptive-QK variants next.
- `2026-04-02`: Launched the next structured batch on the same pod:
  1. `adaptive_qk_late_headgroup_screening`
  2. `adaptive_qk_late_headgroup_pos_screening`
- `2026-04-02`: First `adaptive_qk_late_headgroup_screening` launch failed during warmup with a DDP unused-parameter error because late-only gain offsets were inactive outside the late layers.
- `2026-04-02`: Patched the structured controller so the late-only parameters remain part of the computation graph every step while still being zero-effect before their activation layers.
- `2026-04-02`: Re-validated both structured variants locally after the patch:
  - Python compile checks
  - exact init-equivalence check against the root baseline (`loss_delta 0.0`)
- `2026-04-02`: `adaptive_qk_late_headgroup_screening` completed and is a clear positive result against the fresh matched baseline:
  - exact round-trip `val_bpb`: `1.33712560`
  - delta vs baseline `1.34248996`: `-0.00536436`
  - total size int8+zlib: `13,343,911`
  - final `step_avg`: `475.05 ms`
  - wall-clock stop step: `1264`
- `2026-04-02`: Decision: keep the structured late-headgroup adaptive-QK branch alive as the new mainline and evaluate whether the queued `late_headgroup_pos` variant improves on it.
- `2026-04-02`: Prepared and locally validated two immediate follow-ups around the `late_headgroup` winner:
  - `adaptive_qk_late_headgroup_g2_screening`
  - `adaptive_qk_late_headgroup_late7_screening`
- `2026-04-02`: Both follow-ups resolve through the harness and remain exact no-ops at init relative to the baseline.
- `2026-04-02`: `adaptive_qk_late_headgroup_pos_screening` completed and is a negative final result despite mid-run promise:
  - exact round-trip `val_bpb`: `1.34307370`
  - fresh baseline: `1.34248996`
  - plain `late_headgroup`: `1.33712560`
  - total size int8+zlib: `13,239,616`
  - final `step_avg`: `501.93 ms`
- `2026-04-02`: Decision: prune `late_headgroup_pos`; continue from the plain `late_headgroup` winner and test the prepared `late7` and `g2` follow-ups next.
- `2026-04-02`: `adaptive_qk_late_headgroup_late7_screening` is now the active run on pod `xrerq1wjjbmiwy`.
- `2026-04-02`: Early checkpoint read for `late7` is encouraging versus both the fresh baseline and the current `late_headgroup` winner:
  - step `200` val_bpb: `1.6634`
  - current `late_headgroup` step `200` val_bpb: `1.6667`
  - fresh baseline step `200` val_bpb: `1.6841`
  - step `200` avg time: `470.78 ms`
- `2026-04-02`: Decision: let `late7` run through to completion before interpreting the branch further; if it holds this direction at round-trip time, keep the follow-up queue on the tighter late-layer side.
- `2026-04-02`: `adaptive_qk_late_headgroup_late7_screening` completed and is the new branch winner:
  - exact round-trip `val_bpb`: `1.33682574`
  - delta vs fresh baseline `1.34248996`: `-0.00566422`
  - delta vs prior `late_headgroup` winner `1.33712560`: `-0.00029986`
  - total size int8+zlib: `13,291,168`
  - final `step_avg`: `491.45 ms`
  - wall-clock stop step: `1221`
- `2026-04-02`: Decision: promote `late7` to the new adaptive-QK mainline. The `g2` follow-up is now running as the next structural simplification test on the same pod.
- `2026-04-02`: The first `adaptive_qk_late_headgroup_g2_screening` attempt materialized and reached step `350`, but then stalled without producing a final export or submission record.
- `2026-04-02`: Decision: discard the partial `g2` attempt as infrastructure noise and rerun the exact same preset cleanly before interpreting the grouped-head simplification direction.
- `2026-04-02`: The clean `g2` rerun also failed to become a credible candidate:
  - reached only step `100`
  - step `100` avg time ballooned to `951.35 ms`
  - process exited without a final export result (`returncode 255`)
- `2026-04-02`: Decision: prune the `g2` simplification direction. Even before the crash, its overhead was far outside the intended low-cost regime for this branch.
- `2026-04-02`: Prepared the next three low-overhead follow-ups around the `late7` winner:
  - `adaptive_qk_late_headgroup_late8_screening`
  - `adaptive_qk_late_headgroup_late7_clamp025_screening`
  - `adaptive_qk_late_headgroup_late7_g8_screening`
- `2026-04-02`: Local validation passed for the new follow-ups via `pg_harness show`.
- `2026-04-02`: Next remote run: `adaptive_qk_late_headgroup_late8_screening`.
- `2026-04-02`: `late8` exposed a screening-harness stability issue rather than a modeling result:
  - the SSH-coupled wrapper sometimes exits with `returncode 255` during or just after warmup
  - on retry, the harness can leave duplicate `torchrun` trees running on the pod at the same time
  - those duplicate trees materially contaminate step-time readings
- `2026-04-02`: Decision: for the active adaptive-QK branch, continue execution on the same pod via a detached remote runner and direct log monitoring until the pod is stable again.
- `2026-04-02`: Cleaned the pod back to a single detached `late8` trainer and removed duplicate `torchrun` trees from failed wrapper retries.
- `2026-04-02`: The detached `late8` attempts never produced a trustworthy branch readout:
  - one run died after step `150`
  - later attempts were contaminated by duplicate detached jobs and stale manual variants
  - no clean `late8` checkpoint was strong enough to justify prioritizing it over the other follow-ups
- `2026-04-02`: Verified a clean detached follow-up for `adaptive_qk_late_headgroup_late7_g8` on the same pod.
- `2026-04-02`: `adaptive_qk_late_headgroup_late7_g8` is a meaningful live candidate:
  - step `200` val_bpb: `1.6402`
  - current `late7` winner step `200` val_bpb: `1.6634`
  - fresh baseline step `200` val_bpb: `1.6841`
  - step `200` avg time: `527.91 ms`
- `2026-04-02`: Decision: keep `g8_late7` running through the full screening window. The early quality gain is large enough to justify the extra step time.
- `2026-04-02`: `g8_late7` kept its advantage at later matched checkpoints:
  - step `400` val_bpb: `1.4944` vs `late7` `1.5005`
  - step `600` val_bpb: `1.4269` vs `late7` `1.4324`
  - running step avg through step `600`: `531.64 ms`
- `2026-04-02`: Decision: the `g8_late7` quality lead has persisted long enough to justify waiting for the final round-trip result before starting the clamp follow-up.
- `2026-04-02`: `g8_late7` completed and is an interesting but negative final result:
  - exact round-trip `val_bpb`: `1.34150133`
  - total size int8+zlib: `12,832,162`
  - final `step_avg`: `523.06 ms`
  - last pre-roundtrip `val_bpb`: `1.3400`
- `2026-04-02`: Interpretation: `g8_late7` looked better than `late7` at the matched mid-run checkpoints, but the slower wall-clock pace left it short of updates by the end of the 600-second window.
- `2026-04-02`: Decision: prune `g8_late7` as a final candidate and move to `adaptive_qk_late_headgroup_late7_clamp025_screening` next.
- `2026-04-02`: Launched a clean detached run for `adaptive_qk_late_headgroup_late7_clamp025`, keeping the same `late7` structure and only tightening the adaptive log-delta clamp from `0.4` to `0.25`.
- `2026-04-02`: `late7_clamp025` is a clear early negative on throughput:
  - step `50` avg time: `661.84 ms`
  - step `100` avg time: `606.93 ms`
- `2026-04-02`: Decision: prune `late7_clamp025` early rather than spending the full 600-second budget on a variant that is already far outside the low-overhead target.
- `2026-04-02`: The clean detached `late8` retry completed and is the new branch winner:
  - exact round-trip `val_bpb`: `1.33649155`
  - delta vs fresh baseline `1.34248996`: `-0.00599841`
  - delta vs prior `late7` winner `1.33682574`: `-0.00033419`
  - total size int8+zlib: `12,989,925`
  - final `step_avg`: `500.62 ms`
  - wall-clock stop step: `1199`
- `2026-04-02`: Interpretation: pushing the late adaptive bias one layer deeper outperformed the original `late7` winner on the final round-trip metric, while staying substantially faster than the more expressive but ultimately losing `g8_late7` variant.
- `2026-04-02`: Current adaptive-QK ranking:
  1. `late8`: `1.33649155`
  2. `late7`: `1.33682574`
  3. `late_headgroup`: `1.33712560`
  4. `g8_late7`: `1.34150133`
  5. fresh baseline: `1.34248996`
- `2026-04-02`: Launched `late8_clamp03` as the next refinement:
  - keep the winning `late8` structure
  - reduce `ADAPTIVE_QK_LOG_DELTA_CLAMP` from `0.4` to `0.3`
  - goal: keep the `late8` quality while trimming the chance of over-sharpening without repeating the severe throughput regression from clamp `0.25`
- `2026-04-02`: Added a clean root-baseline QK-gain comparison sweep:
  - `baseline_q20_screening`
  - `baseline_q30_screening`
  - `baseline_q40_screening`
- `2026-04-02`: Goal: verify that the adaptive-QK winner is actually better than simply increasing the global baseline `QK_GAIN_INIT`.
- `2026-04-02`: Paused the `late8_clamp03` refinement so the next pod budget goes to the cleaner control experiment first.
- `2026-04-02`: Launched a detached clean run for root `baseline_q20` on the same pod.
- `2026-04-02`: `baseline_q20` completed and is a strong but slightly worse control than adaptive `late8`:
  - exact round-trip `val_bpb`: `1.33705309`
  - total size int8+zlib: `12,966,949`
  - final `step_avg`: `504.50 ms`
  - wall-clock stop step: `1190`
- `2026-04-02`: Interpretation: a plain global `QK_GAIN_INIT=2.0` nearly matches the adaptive winner, but `late8` still leads by about `0.00056` exact round-trip `val_bpb`.
- `2026-04-02`: Next control run: `baseline_q30`.
- `2026-04-02`: `baseline_q30` completed and is clearly better than the adaptive-QK winner:
  - exact round-trip `val_bpb`: `1.33262099`
  - total size int8+zlib: `13,046,201`
  - final `step_avg`: `497.27 ms`
  - wall-clock stop step: `1207`
- `2026-04-02`: Interpretation: at least on this screening setup, a plain global `QK_GAIN_INIT=3.0` beats adaptive `late8` by about `0.00387` exact round-trip `val_bpb`.
- `2026-04-02`: Next control run: `baseline_q40`.
- `2026-04-02`: `baseline_q40` is an early negative and does not need the full wall-clock budget:
  - step `200` val_bpb: `1.6608`
  - step `400` val_bpb: `1.5040`
- `2026-04-02`: Interpretation: the global Q-gain curve is not monotonic here; `4.0` is worse than `3.0`, so `3.0` is the relevant fixed global control.
- `2026-04-02`: Next comparison branch: `last_layer_q30` against adaptive `late8`.
- `2026-04-02`: `baseline_q40` completed and confirmed the early read:
  - exact round-trip `val_bpb`: `1.34041343`
  - total size int8+zlib: `13,103,829`
  - final `step_avg`: `496.41 ms`
  - wall-clock stop step: `1209`
- `2026-04-02`: Interpretation:
  - the global fixed-gain curve peaks around `QK_GAIN_INIT=3.0` on this setup
  - `QK_GAIN_INIT=4.0` is worse than `3.0` and only modestly better than the fresh baseline
  - future comparisons should treat `global q30` as the real fixed global control, not `q40`
- `2026-04-02`: Added the fixed late-layer ramp branch `late_qgain_ramp` and locally validated the intended layer schedule:
  - `[1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 3.5]`
- `2026-04-02`: Launched the first late-ramp screening run:
  - `late_qgain_ramp_l5_t35`
- `2026-04-02`: `late_qgain_ramp_l5_t35` completed and is not competitive with the simpler fixed controls:
  - exact round-trip `val_bpb`: `1.33695889`
  - total size int8+zlib: `13,274,114`
  - final `step_avg`: `478.30 ms`
  - wall-clock stop step: `1255`
- `2026-04-02`: Interpretation:
  - the broad late-stack ramp is better than the original fresh baseline
  - but it is worse than `global q30`
  - and far worse than `last_layer_q30`
  - the current evidence points to concentrating Q-gain in the final layer rather than diffusing it backward across layers `5..8`
- `2026-04-02`: `last_layer_q30` completed and is the new best result across the Q-gain comparison family:
  - exact round-trip `val_bpb`: `1.32490829`
  - total size int8+zlib: `13,616,806`
  - final `step_avg`: `443.93 ms`
  - wall-clock stop step: `1352`
- `2026-04-02`: Interpretation:
  - fixed last-layer-only `q_gain=3.0` beats global `QK_GAIN_INIT=3.0` by about `0.00771`
  - fixed last-layer-only `q_gain=3.0` beats adaptive `late8` by about `0.01158`
  - the late-layer focus was directionally right, but the simple fixed last-layer control is currently much stronger than the adaptive branch
- `2026-04-02`: Next comparison branch: `last_layer_q40`.
- `2026-04-02`: `last_layer_q40` completed and is a clear regression from `last_layer_q30`:
  - exact round-trip `val_bpb`: `1.34041343`
  - interpretation: pushing the fixed final-layer gain to `4.0` overshoots badly on this branch, so the meaningful fixed late-layer anchor is still `last_layer_q30`
- `2026-04-02`: Added a much stronger adaptive final-layer control:
  - `adaptive_qk_late_headgroup_late8_clamp10_screening`
  - same `late8` final-layer-only adaptive structure
  - increase `ADAPTIVE_QK_LOG_DELTA_CLAMP` from `0.4` to `1.0`
  - this expands the adaptive multiplier range enough to exceed the successful fixed `q_gain=3.0` and even the tested `3.5` ramp target if learning wants it
- `2026-04-02`: Local validation passed for the stronger adaptive preset:
  - `pg_harness show adaptive_qk_late_headgroup_late8_clamp10_screening`
  - Python compile checks for `experiments/adaptive_qk/*.py`
- `2026-04-02`: Launched a clean detached run for `adaptive_qk_late_headgroup_late8_clamp10_screening` on pod `xrerq1wjjbmiwy`:
  - remote log: `/workspace/parameter-golf/logs/20260402T043803Z_adaptive_qk_late8_clamp10_detached_clean.log`
  - goal: check whether the adaptive late8 branch was simply underpowered before, versus the simpler fixed `last_layer_q30` control
- `2026-04-02`: `adaptive_qk_late_headgroup_late8_clamp10_screening` completed and is a negative answer to the stronger-adaptive question:
  - exact round-trip `val_bpb`: `1.34238080`
  - total size int8+zlib: `12,672,202`
  - final `step_avg`: `512.73 ms`
  - wall-clock stop step: `1171`
- `2026-04-02`: Interpretation:
  - increasing the final-layer adaptive clamp from `0.4` to `1.0` made the branch more expressive, but not better
  - it finished slightly better than the fresh baseline (`1.34248996`) only by noise-scale margin
  - it was much worse than the original `late8` adaptive winner (`1.33649155`)
  - it was far worse than the simple fixed `last_layer_q30` control (`1.32490829`)
  - current read: the adaptive branch was not merely underpowered; the simple fixed final-layer Q-gain change is the stronger idea on this line
- `2026-04-02`: Added a capped adaptive control to test the cleaner hypothesis:
  - `adaptive_qk_late_headgroup_late8_q30cap_screening`
  - same strong `late8` adaptive controller (`log_delta_clamp=1.0`)
  - hard-cap the effective final-layer `q_gain` at `3.0`
  - goal: test whether adaptation helps once it is prevented from overshooting the fixed winner
- `2026-04-02`: Local validation passed for the capped adaptive preset:
  - `pg_harness show adaptive_qk_late_headgroup_late8_q30cap_screening`
  - Python compile checks for `experiments/adaptive_qk/*.py`
  - exact init-equivalence check against the root baseline (`loss_delta 0.0`)
- `2026-04-02`: Launched a clean detached run for `adaptive_qk_late_headgroup_late8_q30cap_screening` on pod `xrerq1wjjbmiwy`:
  - remote log: `/workspace/parameter-golf/logs/20260402T052116Z_adaptive_qk_late8_q30cap_detached_clean.log`
- `2026-04-02`: `adaptive_qk_late_headgroup_late8_q30cap_screening` completed and is also negative against the fixed winner:
  - exact round-trip `val_bpb`: `1.34303670`
  - total size int8+zlib: `12,909,340`
  - final `step_avg`: `522.77 ms`
  - wall-clock stop step: `1148`
- `2026-04-02`: Interpretation:
  - capping the effective adaptive final-layer `q_gain` at `3.0` did not rescue the adaptive branch
  - it was slightly slower than the original adaptive `late8`, and worse on the final round-trip metric
  - it remained far behind the simple fixed `last_layer_q30` control
  - current read: on this line, adaptation itself looks like the losing ingredient, not just overshoot
- `2026-04-02`: Built the first hidden-ladder branch at `experiments/hidden_ladder/`:
  - baseline parent
  - taps from the last two layers (`7,8`)
  - rank-2 low-rank fusion
  - zero-init gates so step `0` is exactly the baseline
- `2026-04-02`: Local validation passed for `hidden_ladder_last2_r2_screening`:
  - `pg_harness show hidden_ladder_last2_r2_screening`
  - Python compile checks
  - exact init-equivalence check against the root baseline (`loss_delta 0.0`)
- `2026-04-02`: Launched a clean detached run for `hidden_ladder_last2_r2_screening` on pod `xrerq1wjjbmiwy`:
  - remote log: `/workspace/parameter-golf/logs/20260402T062519Z_hidden_ladder_last2_r2_detached_clean.log`
- `2026-04-02`: `hidden_ladder_last2_r2_screening` completed and is a real but modest positive:
  - exact round-trip `val_bpb`: `1.34026714`
  - total size int8+zlib: `12,998,882`
  - final `step_avg`: `506.26 ms`
  - wall-clock stop step: `1186`
- `2026-04-02`: Interpretation:
  - the hidden-ladder idea is alive on the baseline parent
  - `last2, rank2` beats the fresh baseline by about `0.00222`
  - but it is still far behind `last_layer_q30`
  - best next follow-up is `last3, rank2` before increasing rank
- `2026-04-02`: Added and launched `hidden_ladder_last3_r2_screening`:
  - broaden late taps from `7,8` to `6,7,8`
  - keep rank fixed at `2`
  - remote log: `/workspace/parameter-golf/logs/20260402T063915Z_hidden_ladder_last3_r2_detached_clean.log`
- `2026-04-02`: `hidden_ladder_last3_r2_screening` completed and did not hold its early checkpoint edge:
  - exact round-trip `val_bpb`: `1.34258512`
  - total size int8+zlib: `13,085,006`
  - final `step_avg`: `514.31 ms`
  - wall-clock stop step: `1167`
- `2026-04-02`: Interpretation:
  - `last3, rank2` looked slightly better early than `last2, rank2`
  - but it finished slightly worse than the fresh baseline and clearly worse than `last2, rank2`
  - current hidden-ladder mainline remains `last2, rank2`, but the branch is not yet strong enough to challenge `last_layer_q30`
- `2026-04-02`: The first two `g2` launch attempts were contaminated by pod contention: both remote runs were left active at once, which doubled step time and made the artifacts uninterpretable.
- `2026-04-02`: Cleaned the pod by terminating both stale `g2` remote processes and relaunched a single fresh `adaptive_qk_late_headgroup_g2_screening` run. Only the clean rerun should be used for comparison.
- `2026-04-02`: The clean `g2` rerun is also underperforming at the first matched checkpoint:
  - step `200` val_bpb: `1.6683`
  - current `late_headgroup` winner step `200` val_bpb: `1.6667`
  - current `late7` winner step `200` val_bpb: `1.6634`
  - step `200` avg time: `534.05 ms`
- `2026-04-02`: Decision: prune `g2` early rather than burn the full 600s on a branch that is slower and already behind on quality.
- `2026-04-02`: Prepared the next two follow-up presets around the `late7` winner:
  - `adaptive_qk_late_headgroup_late8_screening`
  - `adaptive_qk_late_headgroup_g8_late7_screening`
- `2026-04-02`: Local validation passed for both new presets via `pg_harness show` and Python compile checks. Next remote order is `late8` first, then `g8_late7`.
- `2026-04-02`: The heavyweight `records/track_non_record_16mb/2026-04-02_ScoredPosSLOT_PerSampleGPTQ_LastLayerQ30/` stack is being moved off the `1xH100` screening lane.
  - Reason: this branch is a frontier-style `8xH100` stack with scored-position SLOT, per-sample delta/logit-bias eval logic, full GPTQ, `11` layers, and a much heavier optimization/eval path than the baseline-parent ideas.
  - Current read: `1xH100` pruning checkpoints are not a trustworthy proxy for this class of experiment.
  - New validation plan: run a paired `8xH100` A/B on the same script and same seed:
    - control: emulate the upstream `#1229` style setting with `QK_GAIN_INIT=4.0` and `LAST_LAYER_QGAIN_ENABLED=0`
    - variant: local `LastLayerQ30` defaults with `QK_GAIN_INIT=1.5`, `LAST_LAYER_QGAIN_ENABLED=1`, `LAST_LAYER_QK_GAIN_INIT=3.0`
  - Pod: `xiglpo53k8gjaa`
  - Constraint: shut the pod down immediately after the A/B completes.
- `2026-04-02`: Brought the fresh `8xH100` pod `xiglpo53k8gjaa` into a runnable state for the heavyweight A/B:
  - synced the repo workspace
  - bootstrapped the `sp1024` FineWeb cache onto the new volume (`80` train shards + `1` val shard)
  - copied the tokenizer model into `/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
  - installed the record script's missing compression dependencies (`brotli`, `zstandard`)
- `2026-04-02`: Added one extra comparison branch to the record folder before the `8x` run:
  - file: `records/track_non_record_16mb/2026-04-02_ScoredPosSLOT_PerSampleGPTQ_LastLayerQ30/adaptive_last_layer_qgain.py`
  - purpose: minimal adaptive final-layer head-group multiplier, zero-init so it starts as a no-op
  - planned extra comparisons after the core A/B:
    - `last_layer_q40`
    - adaptive final-layer Q-gain
- `2026-04-02`: The first `8x` launch failed immediately because the record folder's default relative `./data/...` paths do not point at the shared pod cache when run from inside the record directory.
  - fix: relaunch with explicit absolute `DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024`
  - fix: relaunch with explicit absolute `TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
- `2026-04-02`: The corrected `8x` control run is now genuinely training:
  - config: upstream-style control (`QK_GAIN_INIT=4.0`, `LAST_LAYER_QGAIN_ENABLED=0`, adaptive off)
  - seed: `1337`
  - inner log: `logs/f48a8d35-ae48-487a-86f3-ff4689b19fb1.txt`
  - startup signal:
    - `step:0/20000 val_bpb:4.1049`
    - `step:500/20000 train_loss:2.3490 step_avg:87.02ms`
    - `step:1000/20000 train_loss:2.2440 step_avg:87.41ms`
  - interpretation: this branch behaves like a real `8xH100` training job, not a broken `1x` proxy.
- `2026-04-02`: The `8x` control completed with a real frontier-style result:
  - config: upstream-style control (`QK_GAIN_INIT=4.0`, `LAST_LAYER_QGAIN_ENABLED=0`, adaptive off)
  - exact final metric: `final_int8_zlib_roundtrip_exact val_bpb: 0.92910946`
  - legal artifact size: `15,569,649` bytes
  - intermediate quant metric: `final_int6_roundtrip_exact val_bpb: 1.13760890`
  - post-EMA float diagnostic: `1.1342` bpb
  - interpretation:
    - the heavyweight stack is behaving sensibly on `8x`
    - the earlier `1x` read was indeed not the right proxy
    - this is now the correct control for comparing `last_layer_q30`
- `2026-04-02`: Launched the paired `8x` `last_layer_q30` run on the exact same setup:
  - config: `QK_GAIN_INIT=1.5`, `LAST_LAYER_QGAIN_ENABLED=1`, `LAST_LAYER_QK_GAIN_INIT=3.0`, adaptive off
  - seed: `1337`
  - outer log: `logs/20260402T081354Z_8x_last_layer_q30.log`
  - goal: measure whether the local final-layer q-gain prior helps, hurts, or is neutral on the scored-position SLOT / per-sample GPTQ stack when tested in its intended `8xH100` regime.
- `2026-04-02`: The paired `8x` `last_layer_q30` run completed:
  - exact final metric: `final_int8_zlib_roundtrip_exact val_bpb: 0.92992728`
  - legal artifact size: `15,563,777` bytes
  - intermediate quant metric: `final_int6_roundtrip_exact val_bpb: 1.13818377`
  - post-EMA float diagnostic: `1.1348` bpb
  - matched comparison vs control:
    - control: `0.92910946`, `15,569,649` bytes
    - `last_layer_q30`: `0.92992728`, `15,563,777` bytes
  - interpretation:
    - `last_layer_q30` is slightly worse on this heavy `8x` stack
    - it only saves about `5.9 KB`, which is not enough to justify the quality loss
- `2026-04-02`: Ran `last_layer_q40` in checkpoint-gated mode instead of paying a full GPTQ/export tail:
  - config: `QK_GAIN_INIT=1.5`, `LAST_LAYER_QGAIN_ENABLED=1`, `LAST_LAYER_QK_GAIN_INIT=4.0`, adaptive off
  - matched `step 4000` checkpoint: `val_bpb: 1.2083`
  - reference checkpoints:
    - control `step 4000`: `1.2073`
    - `last_layer_q30` `step 4000`: `1.2087`
  - interpretation:
    - `q40` is slightly better than `q30` at the checkpoint
    - but it still trails the control, so it was pruned before the long GPTQ/export tail
- `2026-04-02`: Ran the adaptive final-layer probe in checkpoint-gated mode:
  - first attempt failed with a real dtype bug in the adaptive multiplier path under FlashAttention compile
  - fix: force the adaptive multiplier and effective `q_gain` back to the attention dtype before use
  - clean fixed run `step 4000` checkpoint: `val_bpb: 1.2081`
  - reference checkpoints:
    - control `step 4000`: `1.2073`
    - `last_layer_q40` `step 4000`: `1.2083`
    - `last_layer_q30` `step 4000`: `1.2087`
  - interpretation:
    - adaptive final-layer is the best of the late-layer variants at the training checkpoint
    - but it still does not beat the upstream-style control on this `8x` stack
    - result: pruned before the long GPTQ/export tail
- `2026-04-02`: Finished the `8x` validation batch and stopped pod `xiglpo53k8gjaa`.
