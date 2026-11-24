# Results

Every number here comes from a file in this repository.

## Machine learning: grouped 5-fold, recording-disjoint

Source: `outputs/recordwise_topk3_stride32_cv5_capped8/cross_validation_metrics.json`

| Fold | Window acc | Window macro-F1 | Recording acc | Recording macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 84.93 % | 75.39 % | 91.67 % | 73.53 % |
| 2 | 81.47 % | 62.65 % | 75.00 % | 62.25 % |
| 3 | 79.96 % | 65.82 % | 91.67 % | 73.53 % |
| 4 | 76.74 % | 62.32 % | 75.00 % | 60.90 % |
| 5 | 79.78 % | 66.07 % | 100.0 % | 100.0 % |
| **Mean ± std** | **80.57 ± 2.98** | **66.45 ± 5.29** | **86.67 ± 11.18** | **74.04 ± 15.70** |

Weighted F1: window 83.24 ± 2.49 %, recording 83.39 ± 12.67 %. Source-recording
overlap is 0 in every fold. A recording's prediction is the argmax of the mean
of its window logits.

Final checkpoint: `outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth`
(SHA-256 `458bc509…a9738`). It was chosen by validation macro-F1 (0.784 at
epoch 35); test metrics were not used for selection.

### Why the split matters

A random window-level split of the earlier `Radar2DCNN` / max-bin pipeline
scored 96.93 % accuracy (`outputs/real/train_max_stride32_4class_dpu/`). The same
max-bin pipeline under a row-disjoint split scored 43.74 %
(`outputs/recordwise/comparison.md`). Overlapping windows from one recording
make a random split far too optimistic.

## Hardware

| Quantity | Streaming baseline | Streaming optimized |
|---|---:|---:|
| C-synth latency estimate | 160,051 cycles | 80,056 cycles |
| C/RTL cosim latency (measured) | 224,411 cycles | 112,020 cycles |
| Est. Fmax (C-synth) | — | 205.47 MHz |
| Standalone IP post-route period | — | 7.021 ns (150 MHz target not met) |
| System PL clock | 100 MHz | ~97 MHz |
| Setup WNS / hold WHS | +0.068 / +0.010 ns | +1.149 / +0.010 ns |
| LUT | 97,365 (83.13 %) | 98,707 (84.28 %) |
| FF | 71,696 (30.61 %) | 69,227 (29.55 %) |
| BRAM tiles | 25.5 (17.71 %) | 57.5 (39.93 %) |
| DSP | 681 (54.57 %) | 599 (48.00 %) |
| URAM | 0 | 0 |
| Vivado power estimate | 4.196 W | 3.712 W |
| Accelerator frame time | 2.24 ms | ≈ 1.16 ms |

Sources: `6_hardware/hls/radar_cnn_hls_accel/reports/*/` and `6_hardware/vivado/*/`.

The frame-buffered design (`reports/frame_buffered_vitis2025_1/csynth.rpt`) needs
1,625 BRAM_18K (564 % of the device) because it stores full feature maps. That
is why the design was rebuilt as a streaming pipeline.

## Verification chain

| Step | Result |
|---|---|
| Host-float kernel vs PyTorch | max error ≤ 8.94e-7 on one sample per class (`outputs/evidence/`) |
| Fixed-point C-sim | every stage bit-exact against the V1 reference |
| C/RTL co-simulation | PASS, logits identical to C-sim |
| Board execution | not yet run |
