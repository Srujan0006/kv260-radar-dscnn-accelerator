# V2 streaming accelerator

The validated frame-buffered implementation remains isolated in
`../v1_frame_buffered/` and is the numerical reference.

## Current milestone: V2A

Implemented dataflow:

```text
DDR input (10,000 values)
  -> InputPixel stream (10,000 tokens, FIFO depth 32)
  -> 2-row plus sliding-window streaming stem
  -> StemPixel<16> stream (10,000 tokens, FIFO depth 16)
  -> DDR capture (160,000 values, pixel-major/channel-vector order)
```

The stem uses only `2 * 100` activation values for its line buffer; the current
row neighborhood is held in a partitioned `3 x 3` register window. It emits
exactly `100 * 100` pixels with explicit zero padding on all four borders. The
testbench compares every channel of every output pixel against the validated V1
operation order: accumulation, bias, cast to `data_t`, then ReLU.

## Run Vitis HLS

```bash
cd hardware/hls/radar_cnn_hls_accel/v2_streaming
RADAR_CNN_HLS_MODE=csim vitis_hls -f run_hls_stream.tcl
```

Target part: `xck26-sfvc784-2LV-c` (KV260/K26).

Do not add DW1 until V2A C simulation and synthesis pass and their reports are
saved. The next milestone is V2B: streaming stem followed by streaming DW1.
