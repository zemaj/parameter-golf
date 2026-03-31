# Hash Aux Prior

This is a weird-but-plausible experiment wrapper around the root `train_gpt.py` baseline.

It injects a tiny hashed lexical prior directly into logits:

- previous-token ids are hashed into a small bucket table,
- multiple hashes are summed count-sketch style,
- the result is projected into vocab logits,
- the auxiliary branch is blended before the normal logit softcap.

The goal is to cheaply capture local lexical regularities that the benchmark rewards.

## Main env vars

- `AUX_PRIOR_ENABLE=1`
- `AUX_PRIOR_BUCKETS=4096`
- `AUX_PRIOR_DIM=16`
- `AUX_PRIOR_HASHES=2`
- `AUX_PRIOR_SCALE_INIT=0.03`

## Example

```bash
python3 experiments/hash_aux_prior/train_gpt.py
```

Or with a different prior size:

```bash
AUX_PRIOR_BUCKETS=8192 AUX_PRIOR_DIM=24 python3 experiments/hash_aux_prior/train_gpt.py
```

## Expected output lines

- `candidate:hash_aux_prior ...`
- `aux_hash_prior:enabled est_params=...`
- the normal baseline final metric lines such as `final_int8_zlib_roundtrip_exact ...`
