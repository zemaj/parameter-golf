# Experiment: Scylla + Full GPTQ + QK-Gain 4.0 + SLOT

This folder is an unrun experiment scaffold derived from the Scylla Full-GPTQ stack in PR `#1184`, with the eval-time SLOT path taken from the later Full-GPTQ+SLOT implementation in PR `#1209`.

## Intent

- Keep the Scylla tokenizer and metadata-driven byte accounting from PR `#1184`
- Keep the stronger Full Hessian GPTQ / XSA-all style stack as the backbone
- Raise `QK_GAIN_INIT` to `4.0`
- Enable `SLOT` by default
- Keep `TTT` off by default so the first run isolates the cheaper eval-time delta path

## Default Runtime Assumptions

- `DATA_PATH=./data/datasets/fineweb10B_scylla`
- `TOKENIZER_PATH` defaults to the local `candidate.vocab`
- `TOKENIZER_META_PATH` defaults to the local `candidate.meta.npz`
- `QK_GAIN_INIT=4.0`
- `SLOT_ENABLED=1`
- `TTT_ENABLED=0`

## Suggested First Run

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-01_Scylla_FullGPTQ_XSA11_FA3_QKGain4_SLOT/train_gpt.py
```

## Notes

- No claims are made yet about `val_bpb`, artifact bytes, or wallclock.
- This is meant to answer one narrow question first: does `QK_GAIN_INIT=4.0 + SLOT` improve the Scylla `#1184` family without pulling in full TTT?
