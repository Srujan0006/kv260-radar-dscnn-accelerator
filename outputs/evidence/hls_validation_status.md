# HLS validation status

The host-float HLS kernel was compared with the current record-wise PyTorch
checkpoint on one held-out sample from every class.

| Class | Maximum absolute logit error | Status |
|---|---:|---|
| Drone | 4.17e-7 | Passed |
| Bird | 8.94e-7 | Passed |
| Human | 3.58e-7 | Passed |
| CornerReflector | 3.13e-7 | Passed |

Detailed JSON results are stored as `hls_host_float_class0.json` through
`hls_host_float_class3.json` in this directory.

This validates layer order, padding, pooling, folded BatchNorm weights, and the
1×1 classifier in floating point. It is not fixed-point HLS validation.

Fixed-point C simulation, C synthesis, C/RTL co-simulation and implementation were
completed afterwards. See `6_hardware/hls/radar_cnn_hls_accel/reports/` and
`6_hardware/vivado/`.
