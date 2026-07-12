# Version 2 resource summary

> **Outdated (predates 2026-07-11):** V2 is no longer stem-only -- the complete streaming CNN now lives in `v2_streaming_v2b/v2_streaming/` and is host-float verified. This file is kept until real synthesis numbers replace it.

Current scope: Stage 1 proof of concept, input reader and streaming stem only.
This is not the complete CNN.

| Item | Result |
|---|---:|
| Target part | `xck26-sfvc784-2LV-c` |
| Target clock | 6.667 ns (150 MHz) |
| Host compilation | Pass |
| Host-float stem comparison | Pass: 160,000/160,000 equal, maximum error 0 |
| Vitis fixed-point C simulation | Not run yet; Vitis 2025.1 now installed at C:/Xilinx/2025.1 (launch with `vitis-run --mode hls --tcl ...`) |
| BRAM | Pending synthesis report |
| URAM | Pending synthesis report |
| DSP | Pending synthesis report |
| LUT | Pending synthesis report |
| FF | Pending synthesis report |
| Estimated latency | Pending synthesis report |
| Estimated initiation interval | Pending synthesis report |
| Fits K26 | Not established |

The stem stores two previous rows (`2 x 100` values), a local `3 x 3` window,
and bounded streams. It does not allocate a `100 x 100 x 16` feature map.
