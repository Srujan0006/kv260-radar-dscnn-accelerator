# HLS reports

| Folder | Build | Contents |
|---|---|---|
| `frame_buffered_vitis2025_1/` | V1 frame-buffered (`radar_cnn_accel`) | `csynth.rpt`: 1,625 BRAM_18K (564 %), does not fit |
| `baseline_vitis2025_1/` | Streaming, baseline unroll factors | `csynth.rpt` (160,051-cycle estimate), `cosim.rpt` (PASS, 224,411 cycles) |
| `optimized_vitis2022_2/` | Streaming, final unroll factors (`solution3_full`) | `csynth.rpt` (80,056 cycles, est. 205.47 MHz), `cosim.rpt` (PASS, 112,020 cycles), `export_impl.rpt` and `standalone_*_routed.rpt` (standalone IP place and route: 95,735 LUT, 599 DSP, 115 BRAM_18K, 7.021 ns) |
| `fixed_point_validation/` | — | Real input samples and PyTorch reference logits used by the testbenches |
| `final_dscnn_host_float_equivalence.json` | — | Host-float kernel vs PyTorch on one sample |

Integrated-system reports are in [`../../../vivado/`](../../../vivado/).

## Regenerating the fixed-point validation set

```bash
python 6_hardware/hls/radar_cnn_hls_accel/tools/compare_fixed_point.py prepare \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --data outputs/recordwise_topk3_stride32/test/spectrogram_cache.npz \
  --output-dir 6_hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation
```
