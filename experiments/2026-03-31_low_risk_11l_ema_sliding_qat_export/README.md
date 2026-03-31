# Low-Risk 11L EMA Sliding Candidate

This experiment is a self-contained fork for harness-driven competitive runs. It stays out of `records/` on purpose and keeps the method close to the proven March 21-22 meta:

- 11-layer GPT stack at `512d`, `8` heads, `4` KV heads
- `seq_len=2048`, `train_batch_tokens=786432`
- EMA enabled by default
- XSA on the last 4 layers by default
- late QAT hook enabled via `LATE_QAT_THRESHOLD`
- mixed int6 export with sliding-window final evaluation
- manual final-eval temperature hook for simple calibration sweeps

## Main env vars

- `DATA_PATH`, `TOKENIZER_PATH`: dataset/tokenizer inputs, same contract as the repo trainers
- `NUM_LAYERS=11`, `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`: core model/eval shape
- `TRAIN_BATCH_TOKENS=786432`, `ITERATIONS=20000`, `WARMDOWN_ITERS=3500`
- `EMA_ENABLED=1`, `EMA_DECAY=0.997`
- `XSA_LAST_N=4`, `ROPE_DIMS=16`, `LN_SCALE=1`
- `LATE_QAT_THRESHOLD=0.15`: enables fake quant during the LR warmdown tail when the schedule multiplier drops below this value
- `EVAL_STRIDE=64`: sliding-window stride for the final score-oriented eval
- `EVAL_TEMPERATURE=1.0`: manual temperature hook for final roundtrip and sliding eval. Keep at `1.0` by default; a shared harness can sweep values like `0.95` or `1.05` later without changing code.

## Example

```bash
torchrun --standalone --nproc_per_node=8 \
  experiments/2026-03-31_low_risk_11l_ema_sliding_qat_export/train_gpt.py
```

With a manual temperature sweep:

```bash
EVAL_TEMPERATURE=0.95 \
torchrun --standalone --nproc_per_node=8 \
  experiments/2026-03-31_low_risk_11l_ema_sliding_qat_export/train_gpt.py
```

## Expected output lines

Look for these log prefixes when parsing runs:

- `candidate:low_risk_11l_ema_sliding_qat_export`
- `final_eval_hook temperature:... mode:...`
- `late_qat:enabled step:... scale:...`
- `ema:applying EMA weights`
- `Serialized model int6+zstd:` or `Serialized model int6+zlib:`
- `Total submission size int6+...:`
- `final_int6_roundtrip_exact val_loss:... val_bpb:... temperature:...`
- `final_int6_sliding_window_exact val_loss:... val_bpb:... temperature:...`
- `final_int6_sliding_window_s64_exact val_loss:... val_bpb:... temperature:...`

## Notes

- This is an experiment path, not an accepted record folder.
- The temperature hook is intentionally manual to avoid baking in a potentially legality-sensitive auto-tuning loop. If a shared harness wants to search temperatures, it can do so from the outside by setting `EVAL_TEMPERATURE`.
