# Hidden Ladder

This experiment keeps the root baseline stack and training loop, but adds a tiny late-state fusion module right before the final norm and logits.

## Idea

Save the hidden states from a small set of late blocks, then let those states write low-rank corrections into the final hidden state.

The first version is intentionally minimal:

- baseline parent
- no extra attention passes
- no recurrence
- zero-init fusion gates so step `0` is exactly the baseline

## Env Knobs

- `HIDDEN_LADDER_ENABLE`
- `HIDDEN_LADDER_RANK`
- `HIDDEN_LADDER_TAP_LAYERS`

The wrapper also routes `hidden_ladder` parameters through the control-tensor path so they stay in the small corrective-structure bucket during training and export.

## First Screening Preset

- [hidden_ladder_last2_r2_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/hidden_ladder_last2_r2_screening.json)

This starts with:

- taps: layers `7,8`
- rank: `2`

on the default 9-layer baseline.
