# HLS Radar CNN Accelerator

## Purpose
Custom Vitis HLS accelerator for the radar 2D CNN project. It accelerates the trained 100x100 spectrogram classifier directly in programmable logic as an alternative/educational path to the Vitis AI DPU flow.

## Input / Output
- Input: flattened 100x100 grayscale spectrogram, one channel, row-major, `data_t` fixed point.
- Output: 4 logits: `Drone`, `Bird`, `Human`, `CornerReflector`.

## CNN Architecture Implemented
The HLS kernel implements the selected depthwise-separable DS-CNN:

```text
Input [1,100,100]
Conv3x3 1->16 + BN folded + ReLU
DepthwiseConv3x3 16 + BN folded + ReLU
PointwiseConv1x1 16->32 + BN folded + ReLU
MaxPool2x2 -> [32,50,50]
DepthwiseConv3x3 32 + BN folded + ReLU
PointwiseConv1x1 32->64 + BN folded + ReLU
MaxPool2x2 -> [64,25,25]
DepthwiseConv3x3 64 + BN folded + ReLU
PointwiseConv1x1 64->96 + BN folded + ReLU
MaxPool2x2 -> [96,12,12]
DepthwiseConv3x3 96 + BN folded + ReLU
PointwiseConv1x1 96->128 + BN folded + ReLU
AvgPool12x12 -> [128]
1x1 Conv classifier -> [4]
```

## Files
- `v1_frame_buffered/` — validated fixed-point frame-buffered accelerator and testbenches. This is the numerical reference.
- `v2_streaming_v2b/v2_streaming/` — **ACTIVE**: complete streaming (DATAFLOW) implementation of the full CNN, layer templates in `stream_layers.h`, three milestone kernels (stem / stem+DW1 / full pipeline). See its README for verified results.
  (The old stem-only `v2_streaming/` snapshot was quarantined to `../../deprecated/hls_v2a_stem_only/` on 2026-07-11.)
- `config/hardware_config.tcl` — shared K26 part and target clock.
- `reports/v1_resource_summary.md` — V1 synthesis summary template/status.
- `reports/v2_resource_summary.md` — V2 synthesis summary template/status.
- `reports/v1_vs_v2_comparison.md` — factual comparison without estimated results.
- `tools/export_weights_to_hls.py` — folds Conv+BN and writes trained weights to `weights.h`.
- `tools/compare_fixed_point.py` — exports real samples, stores PyTorch logits, and compares C-simulation output.

## Toolchain: Vitis 2025.1 (installed at `C:\Xilinx\2025.1`)

Vitis 2025.1 has **no `vitis_hls` executable** — the classic launcher was
retired. All Tcl scripts in this repo still work; launch them through
`vitis-run` instead. From Git Bash on this machine:

```bash
export PATH="/c/Xilinx/2025.1/Vitis/bin:$PATH"
# then, in any folder that has a run_hls*.tcl:
vitis-run.bat --mode hls --tcl <script>.tcl
```

From Windows CMD the equivalent is:

```bat
call C:\Xilinx\2025.1\Vitis\settings64.bat
vitis-run --mode hls --tcl <script>.tcl
```

Env vars (`RADAR_CNN_HLS_MODE`, `RADAR_CNN_HLS_TESTBENCH`) work exactly as
before — set them in the shell before invoking `vitis-run`. The HLS headers
(`ap_fixed.h`, `hls_stream.h`) live at `C:/Xilinx/2025.1/Vitis/include`;
`.vscode/c_cpp_properties.json` points IntelliSense there, which resolves
the red-squiggle include errors in the editor.

## Generate Real Weights
From the main project root. Note: the committed `.venv/` is an empty
bootstrap (pip/setuptools only) — install `requirements_host.txt` into it
first, or use whichever environment actually has torch installed:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python hardware/hls/radar_cnn_hls_accel/tools/export_weights_to_hls.py \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --output hardware/hls/radar_cnn_hls_accel/v1_frame_buffered/weights.h
```

## Host Equivalence Check

This compiles the kernel with host floating-point types and compares its logits
with PyTorch on a cached spectrogram. It validates layer ordering, pooling,
padding, folded BatchNorm parameters, and weight layout without Vitis:

```bash
python hardware/hls/radar_cnn_hls_accel/tools/compare_hls_float_python.py \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --cache outputs/recordwise_topk3_stride32/test/spectrogram_cache.npz
```

It is not bit-accurate fixed-point validation. Vitis HLS C simulation and
cosimulation are still required before treating the IP as hardware-validated.

## Fixed-Point Validation Path

Prepare a balanced set of real spectrogram samples and PyTorch reference logits:

```bash
python hardware/hls/radar_cnn_hls_accel/tools/compare_fixed_point.py prepare \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --data outputs/recordwise_topk3_stride32/test/spectrogram_cache.npz \
  --output-dir hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation
```

Then run Vitis HLS C simulation with the real-sample testbench:

```bash
cd hardware/hls/radar_cnn_hls_accel/v1_frame_buffered
RADAR_CNN_HLS_TESTBENCH=tb_real_samples.cpp \
  vitis-run.bat --mode hls --tcl run_hls.tcl | tee ../reports/fixed_point_validation/csim_output.txt
```

Finally compare the HLS fixed-point logits against PyTorch:

```bash
python hardware/hls/radar_cnn_hls_accel/tools/compare_fixed_point.py compare \
  --reference hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/pytorch_reference.json \
  --hls-output hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/csim_output.txt \
  --output hardware/hls/radar_cnn_hls_accel/reports/fixed_point_validation/hls_compare_report.json
```

Target acceptance:

- logit absolute error below `0.10`
- prediction match ratio above `95%`

## Status

- **Selected DS-CNN host-float PyTorch equivalence:** complete (maximum absolute
  error `1.67e-6` on the checked real sample).
- **V1 fixed-point C simulation:** previously reported as passing, but its exact
  output log must be recaptured.
- **V2 streaming, COMPLETE CNN pipeline (V2C, 2026-07-11):** host-float verified.
  Every stage and the end-to-end `radar_cnn_stream_full` kernel are **bit-exact**
  against the V1-order reference, and all 8 golden radar samples match the
  PyTorch logits within `4.3e-6` (tolerance `2e-4`), 8/8 predictions correct.
  Along the way DW1 was corrected from an unverified stride-2 draft to the
  stride-1 form the trained checkpoint actually uses (downsampling happens only
  in the max pools).
- **V2 fixed-point C simulation:** PASSED 2026-07-12 (Vitis 2025.1 csim,
  ap_fixed types): every stage and the full `radar_cnn_stream_full` kernel
  bit-exact vs the V1-order reference, 0 errors.
- **V2 C synthesis:** first full-kernel synthesis captured 2026-07-12; it did
  NOT fit the K26 (BRAM 816 vs ~288, LUT ~202k vs ~117k) and triggered the
  optimization round documented in
  `v2_streaming_v2b/v2_streaming/README.md` (depthwise pipeline rewrite,
  SRL FIFOs, cyclic weight banking, wider AXI). Re-synthesis pending.
- **V1 C synthesis:** pending; resource summaries contain no invented values.
- **Timing closure:** pending.
- **Board execution:** pending.

## Run the V2 streaming accelerator

Host-float verification (no Vitis needed; checks every stage plus the golden
PyTorch samples):

```bash
cd hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming
g++ -std=c++17 -O2 -DRADAR_CNN_SOFTWARE_FLOAT -Wno-unknown-pragmas \
  -o tb_stream tb_stream.cpp radar_cnn_stream.cpp
./tb_stream ../../golden/input_samples.txt \
            ../../reports/fixed_point_validation/pytorch_reference.csv
```

Vitis HLS (per milestone, each in its own solution):

```bash
export PATH="/c/Xilinx/2025.1/Vitis/bin:$PATH"
cd hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_stream.tcl   # V2A stem regression
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_dw1.tcl      # V2B stem+DW1 regression
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_full.tcl     # V2C full CNN
```

Replace `csim` with `synth` after C simulation passes.

## Run Vitis HLS (V1 frame-buffered)
From the project root:

```bash
export PATH="/c/Xilinx/2025.1/Vitis/bin:$PATH"
cd hardware/hls/radar_cnn_hls_accel/v1_frame_buffered
vitis-run.bat --mode hls --tcl run_hls.tcl
```

After synthesis and IP export complete, collect the C-simulation, synthesis,
latency/resource, export, and packaged-IP evidence:

```bash
bash scripts/collect_hls_reports.sh
```

## Integration Notes
There are two independent top-level kernels — do not look for one in the
other's source file:

```cpp
// V1 frame-buffered reference (v1_frame_buffered/radar_cnn_hls.cpp):
void radar_cnn_accel(const data_t *input, data_t *logits);

// V2 streaming, complete CNN (v2_streaming_v2b/v2_streaming/radar_cnn_stream.cpp;
// the same file also defines the radar_cnn_stream_stem / _dw1 milestone kernels):
void radar_cnn_stream_full(const data_t *input, data_t *logits);
```

Both use the same AXI4 interface pattern:

In a block design, connect:
- `s_axi_control` to AXI-Lite control interconnect.
- `m_axi_gmem0` and `m_axi_gmem1` to PS DDR through HP/HPC port.

The PS should write one flattened spectrogram of 10,000 fixed-point samples to DDR, start the IP, then read 4 logits.

## Important Limitations
- This is a straightforward HLS implementation, not yet deeply optimized.
- It is intended for KV260-sized resources, not PYNQ-Z2.
- For best latency, later optimize with line buffers, streaming, tiling, channel parallelism, and weight partitioning.
- The existing Vitis AI DPU model is still the recommended fastest deployment path. This HLS implementation is useful for learning, customization, and hardware-architecture demonstration.
