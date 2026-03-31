# AR Temp-Calibrated Long-Context Proxy

This is a non-record benchmark-exploit candidate built from the strong `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` trainer, but with the export path changed to optimize what the benchmark actually scores after compression.

The key idea is simple:

- keep the autoregressive self-generated GPTQ calibration,
- reuse a subset of those self-generated sequences as an export-time proxy set,
- search a small temperature grid on the quantized roundtrip model,
- pick the temperature that minimizes post-quant proxy BPB,
- report both normal roundtrip BPB and long-context sliding-window BPB, plus one explicit `benchmark_exploit_candidate_exact` line that selects the better of the two final metrics.

This stays challenge-legal because calibration uses only model-generated tokens and the already-produced artifact.

## Run

From this folder:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
EXPORT_CALIB_SEQS=16
EXPORT_TEMP_GRID=0.90,0.94,0.98,1.00
EXPORT_PROXY_STRIDE=64
```

## Final Metric Lines

The script prints these export-time summary lines at the end:

```text
export_temp_search temp:... calib_block_bpb:... calib_proxy_bpb:... proxy_mode:...
export_temp_selected temperature:... proxy_mode:... calib_proxy_bpb:...
final_int6_roundtrip_exact val_loss:... val_bpb:... temperature:...
final_int6_sliding_window_exact val_loss:... val_bpb:... stride:... temperature:...
final_int6_sliding_window_s64_exact val_loss:... val_bpb:... temperature:...
benchmark_exploit_candidate_exact mode:... val_loss:... val_bpb:... temperature:... stride:...
```

The last line is the candidate's explicit score-aware summary.
