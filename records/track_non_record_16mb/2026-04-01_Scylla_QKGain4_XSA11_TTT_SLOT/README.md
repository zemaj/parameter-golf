# Experiment: Scylla Port of QK-Gain + XSA11 + SLOT/TTT Stack

This folder is an unrun experiment scaffold derived from the PR `#1176` eval stack, but retargeted at the Scylla tokenizer flow from PRs `#1143` and `#1184`.

## Intent

- Keep the `#1176` ideas that looked most transferable: `QK_GAIN_INIT=4.0`, XSA-all, SLOT, optional TTT
- Swap the tokenizer and byte-accounting path over to the local Scylla assets in this folder
- Start with `SLOT_ENABLED=1` and `TTT_ENABLED=0`
- Re-enable TTT only if Scylla + SLOT looks promising enough to justify the extra eval budget

## Default Runtime Assumptions

- `DATA_PATH=./data/datasets/fineweb10B_scylla`
- `TOKENIZER_PATH` defaults to the local `candidate.vocab`
- `TOKENIZER_META_PATH` defaults to the local `candidate.meta.npz`
- `VOCAB_SIZE=998`
- `QK_GAIN_INIT=4.0`
- `SLOT_ENABLED=1`
- `TTT_ENABLED=0`

## Suggested First Run

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-01_Scylla_QKGain4_XSA11_TTT_SLOT/train_gpt.py
```

## Notes

- The historical seed logs in this folder came from the copied `#1176` base and are not claims for the Scylla variant.
- This scaffold is intentionally biased toward the lower-risk `SLOT-first` version of the experiment.

## Acknowledgments

- **PR #1135** (@barneywohl) — base architecture, Parallel Muon, GPTQ, TTT implementation
- **PR #1125** — QK_GAIN systematic sweep (45 experiments)
- **PR #1128** (@AnubhavBharadwaaj) — SLOT technique reference
- **Hu et al. (arXiv:2505.12392v2)** — SLOT paper
- **PR #549** (@abaybektursun) — legal score-first TTT pattern
- **Issue #140** (@notapplica) — comprehensive community analysis

## Reproduction

```bash
# On 8×H100 SXM with RunPod parameter-golf template:
cd /workspace/parameter-golf
QK_GAIN_INIT=4.0 TTT_ENABLED=0 SLOT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training: ~600s. Start with sliding + SLOT only, then re-enable TTT only if the Scylla transfer is promising.
