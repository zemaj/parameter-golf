# Late Q Gain Ramp

This experiment keeps the root baseline trainer and model, but changes the `q_gain`
initializer only in the late layers.

## Idea

Instead of:

- increasing `QK_GAIN_INIT` globally for every layer, or
- changing only the final layer,

this branch linearly ramps `q_gain` from the baseline value to a larger target
across the final layers.

For a 9-layer baseline with:

- `QK_GAIN_INIT=1.5`
- `LATE_QGAIN_RAMP_START_LAYER=5`
- `LATE_QGAIN_RAMP_TARGET_QK_GAIN_INIT=3.5`

the schedule becomes:

- layer 5: `2.0`
- layer 6: `2.5`
- layer 7: `3.0`
- layer 8: `3.5`

## Purpose

This is the clean fixed-control version of the “ramp up late-layer Q gain”
idea. It helps answer whether the adaptive-QK signal is really about:

- sharper attention only at the end of the stack, or
- a more distributed late-layer sharpening schedule.

## Env Knobs

- `LATE_QGAIN_RAMP_ENABLE`
- `LATE_QGAIN_RAMP_START_LAYER`
- `LATE_QGAIN_RAMP_TARGET_QK_GAIN_INIT`
- regular baseline knobs such as `QK_GAIN_INIT`
