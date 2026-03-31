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
python3 tools/pg_harness.py show baseline_smoke
```

Materialize a run directory without executing:

```bash
python3 tools/pg_harness.py materialize baseline_smoke --set MAX_WALLCLOCK_SECONDS=30
```

Run a preset and capture parsed metrics:

```bash
python3 tools/pg_harness.py run baseline_smoke
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

The compare command ranks rows by best parsed `val_bpb` and, by default, shows delta versus the built-in `main-track-sota` snapshot. Override or disable the target with `--target 1.1200` or `--target none`.

Materialize or run a sweep across override values:

```bash
python3 tools/pg_harness.py sweep low_risk_11l_reference \
  --grid EVAL_TEMPERATURE=0.94,0.98,1.00,1.02 \
  --materialize-only
```

```bash
python3 tools/pg_harness.py sweep hash_aux_prior_reference \
  --grid AUX_PRIOR_BUCKETS=2048,4096 \
  --grid AUX_PRIOR_SCALE_INIT=0.02,0.03 \
  --summary-json experiments/runs/hash_aux_prior_sweep_summary.json
```

Mac-friendly MLX smoke and screening presets:

```bash
python3 tools/pg_harness.py show mlx_mac_smoke
python3 tools/pg_harness.py materialize mlx_mac_smoke
python3 tools/pg_harness.py run mlx_mac_smoke
python3 tools/pg_harness.py run mlx_mac_screen --set ITERATIONS=400
```

For Apple Silicon, install the MLX path described in the root README and download the `sp1024` FineWeb cache first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Your Mac runs are useful for directional local screening, but they are not leaderboard-comparable with the 8xH100 track.

## Notes

- The parser is intentionally generic and looks for `final* ... val_loss:... val_bpb:...` lines, plus benchmark-specific summary lines such as `benchmark_exploit_candidate_exact ...`.
- Different candidate scripts can print different `final_*` labels; the harness keeps all of them and also reports the best final metric it sees.
- Built-in target snapshots are convenience references, not live leaderboard queries. The current default snapshot in this branch is `main-track-sota = 1.1147` as of `2026-04-01`.
- Presets are easy to clone for new candidate scripts added under experiment-focused paths.
