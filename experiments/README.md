# Experiments

This directory holds lightweight experiment tooling and presets for Parameter Golf work.

The harness is intentionally thin:

- presets live in `experiments/presets/*.json`
- run artifacts live in `experiments/runs/`
- `tools/pg_harness.py` materializes commands, executes runs, sweeps override grids, and parses final metrics from logs

## Common commands

List presets:

```bash
python3 tools/pg_harness.py list
```

Show built-in leaderboard target snapshots:

```bash
python3 tools/pg_harness.py targets
```

Show a preset:

```bash
python3 tools/pg_harness.py show baseline_screening
```

Materialize a run directory without executing:

```bash
python3 tools/pg_harness.py materialize baseline_screening --set MAX_WALLCLOCK_SECONDS=300
```

Run a preset and capture parsed metrics:

```bash
python3 tools/pg_harness.py run baseline_screening
```

Parse an existing log from a record or run:

```bash
python3 tools/pg_harness.py parse-log records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log
```

Compare several existing logs or harness run directories:

```bash
python3 tools/pg_harness.py compare \
  records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log \
  records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train.log
```

The compare command ranks rows by best parsed `val_bpb`. You can compare against a fixed baseline with `--baseline <name>`, and you can opt into or disable the built-in snapshot target with `--target 1.1200` or `--target none`.

Materialize or run a sweep across override values:

```bash
python3 tools/pg_harness.py sweep baseline_screening \
  --grid TRAIN_BATCH_TOKENS=131072,262144,524288 \
  --summary-json experiments/runs/baseline_screening_sweep.json
```

```bash
python3 tools/pg_harness.py sweep hash_aux_prior_screening \
  --grid AUX_PRIOR_BUCKETS=2048,4096 \
  --grid AUX_PRIOR_SCALE_INIT=0.02,0.03 \
  --summary-json experiments/runs/hash_aux_prior_sweep_summary.json
```

Run a screening study across several algorithm presets on the same Runpod pod:

```bash
python3 tools/pg_harness.py study \
  baseline_screening \
  hash_aux_prior_screening \
  low_risk_11l_screening \
  --set MAX_WALLCLOCK_SECONDS=300 \
  --summary-json experiments/runs/screening_study.json
```

`study` defaults to comparing everything against the first preset as the baseline, and `study` / `sweep` default to `--target none` so local screening stays local-by-local unless you explicitly request a snapshot comparison.

The screening presets launch or resume the default Runpod screening pod automatically, sync the current workspace, ensure the `sp1024` FineWeb cache exists on the remote volume, run the command remotely, and stop the pod when finished unless you pass `--keep-pod-running`.

If you plan to launch several studies back-to-back, prefer `--keep-pod-running` during the active screening session and stop the pod after the batch. That avoids paying repeated cold-start time and reduces the chance of hitting transient Runpod resume-capacity failures between runs.

To use the screening executor, put `RUNPOD_API_KEY` in a local `.env` file, add your SSH public key to Runpod, and keep the helper defaults aligned with your account:

```bash
python3 tools/runpod_pod.py list-pods
python3 tools/runpod_pod.py list-gpus --match H100
```

Single-H100 screening runs are directional only. Compare them against other single-H100 screening runs, then promote promising ideas to the real 8xH100 setup.

Recommended screening flow:

```bash
python3 tools/pg_harness.py run baseline_screening
python3 tools/pg_harness.py study \
  baseline_screening \
  hash_aux_prior_screening \
  low_risk_11l_screening \
  --summary-json experiments/runs/screening_study.json
```

That gives you one remote baseline plus candidate variants on the same Runpod pod shape, with per-run artifacts under `experiments/runs/` and baseline-relative deltas in the study table.

## Notes

- The parser is intentionally generic and looks for `final* ... val_loss:... val_bpb:...` lines, plus benchmark-specific summary lines such as `benchmark_exploit_candidate_exact ...`.
- Different candidate scripts can print different `final_*` labels; the harness keeps all of them and also reports the best final metric it sees.
- Built-in target snapshots are convenience references, not live leaderboard queries. The current default snapshot in this branch is `main-track-sota = 1.1147` as of `2026-04-01`.
- Presets are easy to clone for new candidate scripts added under experiment-focused paths.
- The default screening executor is meant to answer “does this idea help on 1xH100?” before you spend money on full 8xH100 confirmation runs.
