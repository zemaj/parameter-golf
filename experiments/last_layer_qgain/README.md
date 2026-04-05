# Last-Layer Q Gain

This experiment keeps the root baseline trainer and model intact except for one initialization change:

- all layers use the normal baseline `QK_GAIN_INIT`
- the final attention block has its learned `q_gain` vector overridden to a different initializer

## Purpose

This is the clean control for the adaptive-QK branch.

It answers:

- is the adaptive late-layer result better than just increasing Q gain everywhere?
- if a larger global `QK_GAIN_INIT` helps, is the real win simply "more Q gain in the last layer"?

## Env Knobs

- `LAST_LAYER_QGAIN_ENABLE`
- `LAST_LAYER_QK_GAIN_INIT`
- regular baseline knobs such as `QK_GAIN_INIT`

## Suggested Comparison

Use these in order:

- root baseline `QK_GAIN_INIT` sweeps (`baseline_q20`, `baseline_q30`, `baseline_q40`)
- then matching last-layer-only fixed sweeps (`last_layer_q20`, `last_layer_q30`, `last_layer_q40`)
- compare those against the adaptive-QK `late8` winner
