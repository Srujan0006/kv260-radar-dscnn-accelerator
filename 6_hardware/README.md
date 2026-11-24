# Hardware

| Path | Contents |
|---|---|
| [`hls/radar_cnn_hls_accel/`](hls/radar_cnn_hls_accel/README_HLS.md) | Vitis HLS sources (frame-buffered reference and streaming accelerator), golden vectors, weight exporter, and HLS reports for every build |
| [`vivado/baseline_100mhz/`](vivado/baseline_100mhz/) | First integrated system: block design, routed timing, utilization and power reports, `radar_bd_wrapper.xsa` |
| [`vivado/optimized/`](vivado/optimized/) | Final integrated system (the board target): block design, routed reports, `v3_wrapper.xsa` |

Vivado and Vitis project folders are generated and are not committed. Rebuild
them by following [../docs/hardware.md](../docs/hardware.md).
