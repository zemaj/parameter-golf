# Non-Record Experiment: Loader + Full GPTQ + QK Gain 4.0 + SLOT

This folder is a runnable experiment derived from the [#1060](https://github.com/openai/parameter-golf/pull/1060) pure-neural stack, with two additions borrowed from the later eval-time frontier:

- `QK_GAIN_INIT=4.0` from [#1176](https://github.com/openai/parameter-golf/pull/1176)
- a `SLOT`-only eval pass adapted from [#1176](https://github.com/openai/parameter-golf/pull/1176)

The intent is to test a clean standard-tokenizer variant of:

- coprime-stride loader
- Full Hessian GPTQ
- XSA on all 11 layers
- BigramHash(2816x112)
- `QK_GAIN_INIT=4.0`
- `SLOT_ENABLED=1`
- `TTT_ENABLED=0` by default

This experiment has not been run yet, so the bundled seed logs are inherited from the copied [#1060](https://github.com/openai/parameter-golf/pull/1060) base and should be treated as historical reference only, not as results for this variant.

## What Changed vs #1060

### 1. QK Gain Default Raised to 4.0
The copied base already supports query/key gain scaling. This experiment promotes the setting to the default so the folder runs the intended configuration without extra env overrides.

### 2. SLOT Eval Path Added
After the usual sliding-window evaluation, the script can now run a `SLOT` pass that:

- freezes model weights
- optimizes a small per-batch delta vector at the final hidden state
- rescored each batch under that adapted delta

This keeps the implementation separate from TTT and makes it easy to measure `sliding` vs `SLOT` deltas directly.

### 3. TTT Left Off by Default
The goal here is to test the most transferable parts of [#1176](https://github.com/openai/parameter-golf/pull/1176) on top of the stronger [#1060](https://github.com/openai/parameter-golf/pull/1060) backbone. Since stronger Full-GPTQ stacks have often made TTT neutral, `TTT_ENABLED` remains `0` by default.

## Expected Configuration

- 11L, 512d, 8H/4KV (GQA), MLP 3x LeakyReLU(0.5)^2
- XSA on all 11 layers
- BigramHash(2816x112)
- Partial RoPE (16d), LN Scale, EMA
- Full Hessian GPTQ int6 + LZMA
- Parallel Muon + parameter banking
- QK gain 4.0
- SLOT enabled

## Reproduction

From this folder:

```bash
uv pip install -r requirements.txt

SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
USE_GPTQ=1 \
GPTQ_RESERVE_MS=14000 \
TTT_ENABLED=0 \
SLOT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional overrides for SLOT:

```bash
SLOT_LR=0.005 SLOT_STEPS=8
```

## Notes

- This is a non-record experiment folder intended for follow-up runs.
- `submission.json` has been downgraded to an experiment manifest instead of a measured result claim.
- The current defaults are chosen so the folder runs the intended `QK_GAIN=4.0 + SLOT` experiment with minimal extra flags.
