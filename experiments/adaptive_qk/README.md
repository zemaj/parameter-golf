# Adaptive QK

This experiment keeps the root baseline stack and training loop, but replaces the static per-head Q gain with a small adaptive controller.

## Idea

The baseline already learns one `q_gain` value per attention head. This experiment tests whether that attention temperature should also depend on cheap token-local signals, while keeping the rest of the model unchanged.

The first variant adds a multiplicative Q-gain modifier driven by:

- position bucket
- token hidden-state RMS relative to the sequence average

The controller is zero-initialized, so step `0` is exactly the baseline behavior.

Structured follow-up variants are also supported:

- late-layer head-group bias
- late-layer head-group bias plus deeper-layer position buckets

## Scope

- same backbone as the root `train_gpt.py` baseline
- same training loop and export path
- no recurrence
- no dual lanes
- no extra hidden branches

## Env Knobs

- `ADAPTIVE_QK_ENABLE`
- `ADAPTIVE_QK_MODE`
- `ADAPTIVE_QK_NUM_POS_BUCKETS`
- `ADAPTIVE_QK_USE_TOKEN_RMS`
- `ADAPTIVE_QK_LOG_DELTA_CLAMP`
- `ADAPTIVE_QK_EFFECTIVE_Q_GAIN_MAX`
- `ADAPTIVE_QK_HEAD_GROUPS`
- `ADAPTIVE_QK_LATE_START_LAYER`
- `ADAPTIVE_QK_POS_START_LAYER`

## First Screening Preset

Use [adaptive_qk_rms_pos_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_rms_pos_screening.json).

## Structured Follow-Up Presets

- [adaptive_qk_late_headgroup_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_screening.json)
- [adaptive_qk_late_headgroup_pos_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_pos_screening.json)
- [adaptive_qk_late_headgroup_g2_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_g2_screening.json)
- [adaptive_qk_late_headgroup_late7_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late7_screening.json)
- [adaptive_qk_late_headgroup_late8_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late8_screening.json)
- [adaptive_qk_late_headgroup_late8_q30cap_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late8_q30cap_screening.json)
- [adaptive_qk_late_headgroup_late8_clamp10_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late8_clamp10_screening.json)
- [adaptive_qk_late_headgroup_late8_clamp03_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late8_clamp03_screening.json)
- [adaptive_qk_late_headgroup_late7_clamp025_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late7_clamp025_screening.json)
- [adaptive_qk_late_headgroup_late7_g8_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late7_g8_screening.json)
- [adaptive_qk_late_headgroup_late8_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_late8_screening.json)
- [adaptive_qk_late_headgroup_g8_late7_screening.json](/Users/zemaj/www/parameter-golf/experiments/presets/adaptive_qk_late_headgroup_g8_late7_screening.json)
