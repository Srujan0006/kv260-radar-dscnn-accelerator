# HLS Validation Reports

This folder stores fixed-point HLS validation artifacts generated outside the
source tree of the accelerator.

## Verified evidence on the full streaming kernel

`v2c_full_vitis2022_2/` holds the Vitis HLS 2022.2 reports for
`radar_cnn_stream_full` (solution3_full):

| File | Content |
|---|---|
| `csynth.rpt` | Synthesis: 80,056-cycle estimate, per-stage latency/resource breakdown |
| `cosim.rpt` | C/RTL co-simulation: **PASS**, measured 112,020 cycles |
| `export_impl.rpt` | Vivado place & route of the standalone IP: 95,735 LUT, 599 DSP, 115 BRAM, achieved 7.021 ns |

The full-system implementation evidence (block design with the Zynq PS,
timing met at ~97 MHz) lives in the Vivado project's `impl_1` run, outside
this repo.

Recommended workflow:

1. Prepare real spectrogram samples and PyTorch reference logits:

```bash
python hardware/hls/radar_cnn_hls_accel/tools/compare_fixed_point.py prepare \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --data outputs/recordwise_topk3_stride32/test/spectrogram_cache.npz \
  --output-dir hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation
```

2. Run Vitis HLS C simulation with the real-sample testbench:

```bash
cd hardware/hls/radar_cnn_hls_accel/v1_frame_buffered
RADAR_CNN_HLS_TESTBENCH=tb_real_samples.cpp \
  vitis-run.bat --mode hls --tcl run_hls.tcl | tee ../reports/fixed_point_validation/csim_output.txt
```

3. Compare HLS fixed-point output against PyTorch:

```bash
python hardware/hls/radar_cnn_hls_accel/tools/compare_fixed_point.py compare \
  --reference hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/pytorch_reference.json \
  --hls-output hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/csim_output.txt \
  --output hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/hls_compare_report.json
```

Expected deliverables:

- `fixed_point_validation/input_samples.txt`
- `fixed_point_validation/pytorch_reference.json`
- `fixed_point_validation/pytorch_reference.csv`
- `fixed_point_validation/csim_output.txt`
- `fixed_point_validation/hls_vs_pytorch_logits.csv`
- `fixed_point_validation/hls_compare_report.json`
- `synthesis/radar_cnn_accel_csynth.rpt`
- `synthesis/radar_cnn_accel_export.rpt`
- `synthesis/exported_ip/`
