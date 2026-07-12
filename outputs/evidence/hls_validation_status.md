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

Update 2026-07-11: Vitis 2025.1 is now installed at C:/Xilinx/2025.1 (use `vitis-run --mode hls --tcl ...`; the classic `vitis_hls` launcher no longer exists). These steps remain pending:

- Vitis HLS C simulation using `ap_fixed`
- C synthesis
- resource estimate
- latency estimate
- IP export
