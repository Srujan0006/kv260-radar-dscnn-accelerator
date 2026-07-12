# Version 1 versus Version 2

> **Outdated (predates 2026-07-11):** V2 is no longer stem-only -- the complete streaming CNN now lives in `v2_streaming_v2b/v2_streaming/` and is host-float verified. This file is kept until real synthesis numbers replace it.

V2 currently covers only the Stage 1 stem milestone. Full-CNN comparisons stay
pending until the remaining stages are implemented and validated incrementally.

| Item | Version 1 | Version 2 |
|---|---:|---:|
| Architecture | Frame-buffered full CNN | Streaming Stage 1 stem |
| C simulation | Previously reported pass; log recapture required | Vitis C simulation not run |
| Available numerical comparison | Host-float full logits: pass, max error `1.67e-6` | Host-float stem: pass, max error `0` |
| Prediction agreement | Pending fixed-point report | Not applicable before full classifier |
| BRAM | Pending synthesis | Pending synthesis |
| URAM | Pending synthesis | Pending synthesis |
| DSP | Pending synthesis | Pending synthesis |
| LUT | Pending synthesis | Pending synthesis |
| FF | Pending synthesis | Pending synthesis |
| Estimated latency | Pending synthesis | Pending synthesis |
| Fits K26 | Not established | Not established |

Resource fields must be populated from generated HLS reports, never estimates.
