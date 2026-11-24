# Hardware flow

Target: `xck26-sfvc784-2LV-c` (Kria K26 on the KV260 carrier).

## 1. Host check (no Vitis needed)

```bash
cd 6_hardware/hls/radar_cnn_hls_accel/v2_streaming
g++ -std=c++17 -O2 -DRADAR_CNN_SOFTWARE_FLOAT -Wno-unknown-pragmas -o tb_stream tb_stream.cpp radar_cnn_stream.cpp
./tb_stream ../golden/input_samples.txt ../reports/fixed_point_validation/pytorch_reference.csv
```

Expected result: every stage reports `MAX_ERROR 0`, and the run ends with `ALL CHECKS PASSED`.

## 2. Vitis HLS 2022.2

```bash
cd 6_hardware/hls/radar_cnn_hls_accel/v2_streaming
RADAR_CNN_HLS_MODE=all vitis_hls -f run_hls_full.tcl   # csim | synth | cosim | all
vitis_hls -f run_export.tcl                            # IP export + standalone place and route
```

On Windows, copy the folder to a short path (for example `C:\hlswork\v2`)
first. The generated RTL file names exceed MAX_PATH inside a deep checkout.
`run_cosim.ps1` and `run_export.ps1` run from their own folder.

Part, clock and top-function names live in `config/hardware_config.tcl`.

## 3. Vivado system

The block design for both builds is in `6_hardware/vivado/<build>/*.bd`:

- Zynq UltraScale+ PS with the KV260 SOM preset and UART1 on MIO 36/37.
- Control path: `M_AXI_HPM0_FPD` → AXI SmartConnect → `s_axi_control`
  (base `0xA0000000`, 64 KB).
- Data path: `m_axi_gmem0` and `m_axi_gmem1` → `S_AXI_HPC0_FPD` → DDR.
- Clock and reset: `pl_clk0`, with a processor system reset driving `ap_rst_n`.
- Interrupt: routed to `pl_ps_irq0`.

Each build folder also holds its routed timing, utilization and power reports
and the exported `.xsa`.

## Report index

| Build | HLS reports | Vivado system |
|---|---|---|
| Frame-buffered | `reports/frame_buffered_vitis2025_1/` | did not fit |
| Streaming baseline | `reports/baseline_vitis2025_1/` | `vivado/baseline_100mhz/` |
| Streaming optimized | `reports/optimized_vitis2022_2/` | `vivado/optimized/` |
