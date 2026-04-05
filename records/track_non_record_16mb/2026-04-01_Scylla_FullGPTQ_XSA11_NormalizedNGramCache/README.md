# Experiment: Scylla + Full GPTQ + XSA-all + Normalized N-gram Cache

This folder is a non-record experiment based on the Scylla + Full GPTQ + XSA-all stack from PR `#1184`, with an exact incremental multi-order backoff cache adapted from PR `#1185`.

This experiment has not been benchmarked yet in this folder. The metadata below is intentionally conservative until fresh runs exist.

## Goal

Test whether the `#1184` Scylla backbone benefits from a normalized, causal eval-time cache without relying on hashed bucket shortcuts or hardcoded `1024`-token assumptions.

## What Changed

- Base model and tokenizer path come from the Scylla `#1184` stack.
- Added `NgramBackoffCache`, an exact order-2..N backoff cache with:
  - incremental score-first updates
  - Laplace smoothing over the actual configured vocabulary size
  - entropy-adaptive mixing with model log-probabilities
- Enabled cache use in sliding-window eval by default.
- Kept legal score-first TTT off by default so the cache effect can be measured in isolation.
- Default tokenizer config now points at the local `candidate.vocab` and `candidate.meta.npz` assets in this folder.

## Important Caveats

- The cache path currently evaluates redundantly on every rank to preserve identical causal cache state across distributed eval. That keeps the implementation simple and reviewable, but may increase eval cost.
- This experiment still needs fresh timing and BPB measurements.
- The legality of normalized eval-time caches is stricter than plain neural eval and should be reviewed carefully before any leaderboard use.

## Suggested First Run

```bash
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_scylla \
TOKENIZER_PATH=./candidate.vocab \
TOKENIZER_META_PATH=./candidate.meta.npz \
VOCAB_SIZE=998 \
USE_GPTQ=1 \
TTT_ENABLED=0 \
NGRAM_CACHE_ENABLED=1 \
NGRAM_ORDER=9 \
NGRAM_ALPHA_BASE=0.08 \
NGRAM_ALPHA_RANGE=0.65 \
NGRAM_ENTROPY_CENTER=3.5 \
XSA_LAST_N=11 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py`: Scylla `#1184` backbone plus normalized cache integration.
- `candidate.vocab`: Scylla tokenizer vocabulary.
- `candidate.meta.npz`: metadata-driven byte accounting for Scylla tokens.
- `submission.json`: placeholder metadata for this experiment until fresh runs exist.
