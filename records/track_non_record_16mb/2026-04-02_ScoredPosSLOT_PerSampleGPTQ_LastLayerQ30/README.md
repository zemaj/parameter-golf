# Derived Variant: PR #1229 with Last-Layer Q-Gain

This folder is a local derived variant of upstream [PR #1229](https://github.com/openai/parameter-golf/pull/1229), imported for follow-up experiments.

What changed:

- imported the `train_gpt.py` and `requirements.txt` from PR `#1229`
- replaced the global `QK_GAIN_INIT=4.0` default with the repo's local `last_layer_q30` approach
- the script now keeps the baseline/global `q_gain` init at `1.5`
- after model construction, it overrides only the final block's learned `q_gain` to `3.0`
- the same override is applied in the training model, the Hessian collection model, and the post-quant eval model

Environment knobs:

- `QK_GAIN_INIT` defaults to `1.5`
- `LAST_LAYER_QGAIN_ENABLED` defaults to `1`
- `LAST_LAYER_QK_GAIN_INIT` defaults to `3.0`

Important note:

- this is not a validated submission yet
- the upstream `#1229` logs, metrics, and submission metadata were intentionally not copied, because they would no longer match this modified code path

Reproduction:

```bash
torchrun --nproc_per_node=8 train_gpt.py
```
