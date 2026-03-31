# AGENTS.md

## Purpose

This repository supports the OpenAI Model Craft Challenge: Parameter Golf.
The goal is to train the best language model that:

- fits in a `16,000,000` byte artifact,
- trains in under `10 minutes` on `8xH100` GPUs for the main track,
- is evaluated on FineWeb validation using tokenizer-agnostic `val_bpb`.

Agents working in this repo should preserve those constraints and optimize for reproducible, challenge-legal results.

## Repo Map

- `README.md`: challenge rules, leaderboard, submission requirements, and getting-started docs.
- `train_gpt.py`: baseline CUDA trainer for new participants.
- `train_gpt_mlx.py`: baseline MLX trainer for Apple Silicon.
- `data/`: dataset download/export helpers and tokenizer/data workflow docs.
- `records/track_10min_16mb/`: accepted and example main-track submissions.
- `records/track_non_record_16mb/`: non-record or unlimited-compute submissions.

## Working Rules

### 1. Preserve challenge legality

- Treat `val_bpb` as the primary metric, not plain token loss.
- Do not introduce workflows that rely on validation leakage.
- Do not assume evaluation may access external downloads, network resources, or training data unless those bits are explicitly paid for under the artifact limit.
- Remember that the counted artifact is `code bytes + compressed model bytes`, with the code expected to live in `train_gpt.py` for submissions.

### 2. Keep baseline code beginner-friendly

- `train_gpt.py` and `train_gpt_mlx.py` are starter baselines, not the place for every SOTA idea.
- Preserve readability for newcomers.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under the stated `1500` line soft limit unless the user explicitly wants to revisit that policy.
- Competitive or experimental variants usually belong in a new folder under `records/`, not by turning the root trainer into a research dump.

### 3. Put work in the right place

- Use the root trainers for general improvements, simplifications, bug fixes, and baseline-quality tuning.
- Use `records/track_10min_16mb/<run_name>/` for main-track submissions.
- Use `records/track_non_record_16mb/<run_name>/` for unlimited-compute or non-record ideas.
- When adding a new record folder, mirror the structure used by existing records.

### 4. Respect submission structure

Submission folders should generally include:

- `README.md` explaining the method and setup,
- `submission.json` with metadata and measured results,
- `train.log` or equivalent training evidence,
- `train_gpt.py` and any required dependencies needed to reproduce the run from inside that folder.

### 5. Be careful with tokenizer and dataset changes

- The benchmark is tokenizer-agnostic, so tokenizer changes are allowed.
- Tokenizer and dataset modifications are high-risk because incorrect `val_bpb` accounting can create fake wins.
- If changing tokenizer or dataset handling, add clear evidence that the `val_bpb` calculation remains correct.
- Prefer preserving the published FineWeb export conventions unless the task is explicitly about changing them.

### 6. Optimize with the real bottlenecks in mind

Successful submissions usually trade off among three constraints:

- model quality (`val_bpb`),
- compressed artifact size,
- wall-clock training and evaluation time.

Be aware that ideas that improve pre-quantization loss may still fail if they:

- compress poorly,
- exceed the byte cap,
- slow training or evaluation too much,
- depend on illegal post-training data access.

## Practical Guidance For Agents

### When modifying training or evaluation

- Check how final artifact bytes are computed and logged.
- Check how `final_int8_zlib_roundtrip` metrics are produced.
- Avoid changing score definitions unless the task is specifically about benchmark mechanics.
- If a change affects training speed, artifact size, tokenizer accounting, or evaluation legality, mention that explicitly in your summary.

### When creating a submission

- Follow the conventions in nearby `records/.../README.md` files.
- Include exact commands, environment assumptions, artifact size, and final `val_bpb`.
- Keep claims conservative unless supported by logs.
- For a new SOTA-style claim, note that the repo rules require at least a `0.005` nat improvement with sufficient statistical evidence unless it is a pure systems-speed improvement.

### When reviewing PRs or patches

Prioritize findings about:

- benchmark legality,
- incorrect `val_bpb` computation,
- hidden artifact-size regressions,
- reproducibility gaps,
- train/eval time regressions,
- accidental complexity added to the root baseline scripts.

## Useful Mental Model

This is not a generic LLM training repo.
It is a constrained-optimization competition repo where bytes, milliseconds, and legality matter as much as modeling quality.

When in doubt, prefer changes that are:

- reproducible,
- challenge-legal,
- compression-aware,
- easy for future participants to understand.
