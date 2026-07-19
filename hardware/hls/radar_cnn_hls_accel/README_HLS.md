# Custom HLS Accelerator — Streaming Radar CNN

Fixed-point Vitis HLS implementation of the trained `RadarDSCNN2D` classifier
(100×100 spectrogram → 4 logits), deployed as a single streaming DATAFLOW
pipeline with no intermediate feature-map storage.

**Status: fully verified through implementation.** csim bit-exact ✓ ·
synthesis ✓ · C/RTL cosim PASS ✓ · Vivado place-and-route ✓ (system timing
met at ~97 MHz). Only the physical board benchmark remains
(see [../../../deployment/kv260_baremetal/](../../../deployment/kv260_baremetal/)).

## Layout

| Path | Role |
|---|---|
| `v1_frame_buffered/` | Frame-buffered implementation — the **numerical reference**. Too big to fit the K26; kept because every streaming stage is verified bit-exact against it. |
| `v2_streaming_v2b/v2_streaming/` | **ACTIVE** streaming accelerator: `radar_cnn_stream.cpp` (top kernels), `stream_layers.h` (layer templates), `weights.h` (generated), `tb_stream.cpp` (stage-by-stage testbench). |
| `config/hardware_config.tcl` | Single source of truth: part `xck26-sfvc784-2LV-c`, 6.667 ns target clock, top-function names. |
| `reports/` | Validation and synthesis evidence, including `v2c_full_vitis2022_2/` (csynth + cosim + place-and-route reports). |
| `tools/` | `export_weights_to_hls.py` (folds Conv+BN from the checkpoint into `weights.h` — never hand-edit), fixed-point/float comparators. |
| `golden/` | 8 real radar samples + PyTorch reference logits for end-to-end checks. |

Top-level kernels (both AXI: `s_axi_control` + `m_axi_gmem0/1`):

```cpp
void radar_cnn_accel(const data_t *input, data_t *logits);        // V1 reference
void radar_cnn_stream_full(const data_t *input, data_t *logits);  // ACTIVE (also _stem/_dw1 milestones)
```

## Numerical contract

`data_t = ap_fixed<16,6>` (Q6.10), `acc_t = ap_fixed<32,12>`. Every stage
accumulates **starting from the bias, in ascending channel/kernel order**, in
`acc_t`, then casts to `data_t`, then applies ReLU — exactly the V1 order.
Because the order is identical, host-float builds compare against the
reference **bit-exactly**, and any failure localizes to a single stage.
Do not reorder accumulation.

## Verify on the host (seconds, no Vitis)

`-DRADAR_CNN_SOFTWARE_FLOAT` switches `data_t` to `float` and mocks
`hls::stream`:

```bash
cd v2_streaming_v2b/v2_streaming
g++ -std=c++17 -O2 -DRADAR_CNN_SOFTWARE_FLOAT -Wno-unknown-pragmas \
    -o tb_stream tb_stream.cpp radar_cnn_stream.cpp
./tb_stream ../../golden/input_samples.txt \
            ../../reports/fixed_point_validation/pytorch_reference.csv
```

Expected: `MAX_ERROR 0` on every stage, golden samples within 2e-4 of PyTorch
(measured ≤ 4.3e-6), `ALL CHECKS PASSED`.

## Run the full Vitis HLS flow

Classic Vitis HLS (2022.2 or similar — `vitis_hls` executable):

```bash
cd v2_streaming_v2b/v2_streaming
RADAR_CNN_HLS_MODE=csim  vitis_hls -f run_hls_full.tcl   # modes: csim|synth|cosim|all
```

Vitis 2025.1 (no `vitis_hls` binary — use the runner):

```bash
RADAR_CNN_HLS_MODE=csim vitis-run --mode hls --tcl run_hls_full.tcl
```

Milestone scripts `run_hls_stream.tcl` (stem) and `run_hls_dw1.tcl`
(stem+DW1) use separate solutions so reports never overwrite each other.

> **Windows note:** synthesize from a short working directory (e.g.
> `C:\hlswork`). Generated RTL filenames are long enough to break MAX_PATH
> inside a deeply nested repo, even with long paths enabled.

## Regenerate weights after retraining

```bash
python tools/export_weights_to_hls.py \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --output v2_streaming_v2b/v2_streaming/weights.h
```

## Verified results (Vitis HLS 2022.2, solution3_full)

| Stage | Result |
|---|---|
| Fixed-point csim | PASS — every stage bit-exact vs V1, logits exact |
| Synthesis estimate | 80,056 cycles · est. Fmax 205 MHz · 1,018 DSP · 109k LUT |
| C/RTL cosim (measured) | **PASS — 112,020 cycles** (real FIFO back-pressure adds ~40% over the estimate) |
| Place & route (standalone IP) | 95,735 LUT · 599 DSP · 115 BRAM · achieved 7.021 ns (~142 MHz) |
| Full system (Vivado BD @ ~97 MHz) | **timing met** (WNS +1.137 ns), LUT 82%, XSA exported |

Frame time at the 97 MHz system clock: **≈ 1.16 ms (~865 inferences/s)**.

Key design facts worth knowing before editing:

- Inter-stage FIFOs are forced to SRL (`BIND_STORAGE impl=srl`) — a BRAM FIFO
  costs ceil(width/36) BRAM36 *regardless of depth*, and tokens here are up to
  2,048 bits wide.
- Depthwise stages are one flattened `(row, col, channel-block)` pipeline at
  II=1 — reverting to nested pipelines costs up to 55× in cycles.
- UltraScale+ URAM cannot hold initialized ROMs; weights stay in BRAM/LUT-RAM
  (a URAM-bound weight ROM would silently come up all-zero on the board).
- The throughput bottleneck is the early pointwise stages (~80k cycles each,
  II=1); widening their unroll factors is the first lever for more speed.
