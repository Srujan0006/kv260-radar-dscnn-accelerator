# V2 streaming accelerator

The validated frame-buffered implementation remains isolated in
`../../v1_frame_buffered/` and is the numerical reference.

## Status

| Milestone | Top function | Scope | Solution | Status |
|---|---|---|---|---|
| V2A | `radar_cnn_stream_stem` | stem only | `solution1` | verified vs V1 order, bit-exact (host float) |
| V2B | `radar_cnn_stream_dw1` | stem + DW1 | `solution2_dw1` | verified vs V1 order, bit-exact (host float) |
| V2C | `radar_cnn_stream_full` | **complete CNN** | `solution3_full` | **fully verified**: ap_fixed csim bit-exact · C/RTL cosim PASS (112,020 cycles measured) · placed & routed (95,735 LUT, 599 DSP); full system timing met at ~97 MHz |

History: the first synthesis of `radar_cnn_stream_full` did not fit the K26
(BRAM 816/~288, LUT ~202k/~117k) and drove the optimization round documented
below; the optimized design fits with headroom. Reports live in
`../../reports/v2c_full_vitis2022_2/`. The flow runs under classic
`vitis_hls` (2022.2) or `vitis-run --mode hls` (2025.1, which has no
`vitis_hls` executable).

## DW1 stride correction (resolves the old V2B open item)

An earlier V2B draft implemented DW1 as a fused stride-2 depthwise
convolution, assumed "by analogy with the stem" and flagged as
unverified. Cross-checking against `../../v1_frame_buffered/radar_cnn_hls.cpp`
(the validated reference) shows the trained checkpoint uses:

* DW1 = 3x3 depthwise, **stride 1**, padding 1 (output stays 100 x 100);
* all spatial downsampling happens in the three 2x2 max pools
  (100 -> 50 -> 25 -> 12, where 25/2 floors to 12).

DW1 was therefore rewritten as stride 1. `radar_cnn_stream_dw1` now
captures 100 x 100 x 16 = 160,000 values and matches the V1 operation
order bit-for-bit in host-float mode.

## V2C: complete streaming pipeline

```text
DDR -> InputPixel(1) -> stem 3x3 (1->16)            100x100
    -> DW1 3x3 s1 (16)   -> PW1 1x1 (16->32)        100x100
    -> maxpool 2x2                                   50x50
    -> DW2 3x3 s1 (32)   -> PW2 1x1 (32->64)         50x50
    -> maxpool 2x2                                   25x25
    -> DW3 3x3 s1 (64)   -> PW3 1x1 (64->96)         25x25
    -> maxpool 2x2 (floor)                           12x12
    -> DW4 3x3 s1 (96)   -> PW4 1x1 (96->128)        12x12
    -> global average pool -> classifier 1x1 (128->4, no ReLU)
    -> DDR (4 logits)
```

All stages run concurrently in one `DATAFLOW` region; every stage-to-stage
hop is an `hls::stream<Pixel<C>>` FIFO of pixel-major channel vectors. No
intermediate feature map is ever stored (in DDR or on-chip): only the 3x3
depthwise stages keep 2-row line buffers, pool keeps one row of pair
maxima, GAP keeps 128 accumulators.

Channel counts follow the trained checkpoint (`weights.h`): PW3 is
64 -> 96 and PW4 is 96 -> 128 (not 64 -> 128 / 128 -> 256 as some older
planning notes said).

### Layer library (`stream_layers.h`)

* `depthwise_conv_stream<C, H, W, UF>` -- stride-1 "same" 3x3 depthwise;
  per-channel generalization of the verified streaming stem (2-row line
  buffer + 3x3 windows, one flush row/column). One flattened
  `(py, px, cb)` pipeline at II=1; `UF` channels are convolved per cycle
  (`UF * 9` multipliers), all per-channel state banked cyclic by `UF`.
* `pointwise_conv_stream<CIN, COUT, TOKENS, UF_IN, UF_OUT, RELU>` -- one
  pixel vector in, one out; flat pipelined loop over UF_OUT x UF_IN weight
  tiles at II=1. Accumulation is chained from the bias in ascending
  input-channel order, so it is numerically identical to V1's linear loop
  (bit-exact in ap_fixed, same rounding order in float). Also instantiated
  for the classifier (TOKENS=1, RELU=false).
* `maxpool_stream<C, H_IN, W_IN>` -- 2x2 stride-2, floor semantics for odd
  sizes (trailing row/column consumed but unused, exactly like V1); II=1
  per input token, comparators only.
* `gap_stream<C, TOKENS, UF>` -- running sums, divide by TOKENS, emit one
  token.
* Weight/bias arrays are `ARRAY_PARTITION`ed at their use sites, banked
  `cyclic` to match each stage's UF access tile (one read per bank per
  cycle — see the optimization round below). Never bind them to URAM:
  UltraScale+ URAM cannot hold initialized ROMs.

## 2026-07-12 optimization round (driven by the first real synthesis report)

The first Vitis 2025.1 synthesis of `radar_cnn_stream_full` (baseline
report, run at a 10 ns clock) showed the design did not fit the K26:
**BRAM 816** vs ~288 available, **LUT ~202k** vs ~117k, DSP 1,078/1,248.
Three root causes were identified and fixed. **No change alters the
numerical result** — the host-float testbench remains bit-exact vs the
V1-order reference and 8/8 golden PyTorch samples still match within
4.3e-6 after the rewrite.

### 1. Depthwise stages rewritten (latency AND LUT fix)

Baseline: DW2/DW3/DW4 did not pipeline their pixel loops — 228/453/781
cycles/pixel — and burned 24k/41k/57k LUTs, because the window and line
buffer were `complete`-partitioned on the channel dim while the MAC loop
indexed them with a *variable* channel (`cb*UF+u`): every lane got a C:1
multiplexer over all taps.

Fix (in `depthwise_conv_stream`): one PERFECT `(py, px, cb)` loop nest
with `PIPELINE II=1` on the innermost channel-block loop (Vitis flattens
it into a single pipeline — no per-pixel flush), and all per-channel
state (`window`, `line_buffer`, `weights`, `biases`) banked
`cyclic factor=UF` so each lane does one memory access per cycle at
address `cb` instead of muxing. The line buffer additionally gets
`BIND_STORAGE ram_s2p impl=lutram` (banks are at most ~200 words) and a
`DEPENDENCE inter false` waiver (its true reuse distance is one full row,
`(W+1)*C/UF` iterations, far beyond the pipeline depth).

Expected cycles at II=1 = `(H+1)*(W+1)*C/UF`:

| Stage | Baseline cycles | New (II=1) | New (worst case II=2) |
|---|---|---|---|
| DW1 (UF=16) | ~30k | 10,201 | (registers, II=1) |
| DW2 (UF=8) | 570k (228/px) | 10,404 | 20,808 |
| DW3 (UF=8) | 283k (453/px) | 5,408 | 10,816 |
| DW4 (UF=8) | 112k (781/px) | 2,028 | 4,056 |

Even at II=2 (possible if the tool won't prove the distance-`C/UF` window
recurrence; see the comment in `stream_layers.h` for the one-line
`DEPENDENCE` fix) every DW stage sits far below the 160k-cycle critical
path. Tradeoff: DSP count for DW is unchanged (UF×9 multipliers per
stage); the C:1 mux LUTs are simply gone.

### 2. FIFO storage forced to SRL (the ~660-BRAM fix)

A BRAM-built FIFO costs `ceil(width/36)` BRAM36 **regardless of depth**,
and these streams are up to `Pixel<128>` = 2,048 bits wide: 57 BRAMs
each for `pw4`/`gap`, 43 each for the three 1,536-bit streams, 29/15 for
the narrower ones — ~660 BRAM total in the baseline, on a part with ~288.
Every stream now has an explicit depth and
`BIND_STORAGE type=fifo impl=srl` (~1 LUT/bit at these depths, ~14k LUTs
total — paid for many times over by fix 1).

Depth math: the pipeline is one linear chain (no fork/join), so depth 2
(the ap_ctrl handshake minimum) already guarantees deadlock-freedom;
extra depth only smooths row-boundary jitter (a DW stage consumes nothing
during its flush column, a pool emits nothing on even rows). Chosen:
depth 4 for all inter-stage hops, depth 2 for the three widest/slowest
hops (`pw4`, `gap`, `logits` — the GAP consumer drains 48x faster than
PW4 produces, so backpressure never builds), and depth 64 for the 16-bit
`input_pixels` (eight 128-bit AXI beats of slack so the widened DDR burst
keeps streaming while the stem is in flush cycles).

### 3. Weight ROMs re-banked; why URAM is deliberately NOT used

Pointwise weights were `complete dim=2` (CIN banks of depth COUT — up to
96 mostly-empty BRAMs for PW4, ~114 BRAM total); depthwise weights were
`complete dim=0` (all registers, feeding the channel muxes of fix 1).
Both are now banked `cyclic` to exactly match the access tile: each of
the `UF_OUT*UF_IN` (pointwise) / `UF*9` (depthwise) banks serves exactly
one read per cycle — no muxes, no over-partitioning, and bank depths
small enough that most land in LUTRAM.

**URAM exclusion (task was "move weights to URAM"):** UltraScale+ URAM
cannot be initialized from the bitstream (UG573) — a `BIND_STORAGE ...
impl=uram` const ROM would come up **all-zero on the board** with no
tool error to warn about it. The BRAM pressure was therefore fixed by
partitioning instead. Bias vectors (16–128 words) are excluded from any
RAM mapping as well — they are trivially small and stay as
registers/LUTRAM.

### 4. Pointwise parallelism raised where it pays (UF_OUT 2 -> 4)

| Stage | Tokens | Baseline cycles (8x2) | Now | Tile |
|---|---|---|---|---|
| **PW1** | 10,000 | 320k | **160k** | 8x4 |
| **PW2** | 2,500 | 320k | **160k** | 8x4 |
| PW3 | 625 | 240k | 120k | 8x4 |
| PW4 | 144 | 110.6k | 110.6k | 8x2 (unchanged) |

PW4 stays at 8x2 on purpose: it is already below the new 160k critical
path, so widening it would spend 16 DSPs for zero frame-rate gain. The
justification vs the depthwise bottleneck: after fix 1 the DW stages are
~10k cycles, so the pointwise stages ARE the bottleneck and are the only
place extra DSPs buy throughput.

DSP budget (16-bit MACs, 1 DSP each): stem 144 + DW 144+72+72+72 + PW
32*3 + PW4 16 + CLS 16 = **632 MAC DSPs** — about half the K26's 1,248
(baseline report used 1,078). Frame rate: critical path 160k cycles =
**~1.07 ms/frame at 150 MHz (~930 fps)**, 2x the baseline architecture's
320k. Next knob if more is needed: raise UF on **both** PW1 and PW2
(they tie), then PW3.

### 5. AXI masters

`m_axi` input ports now carry `max_widen_bitwidth=128` (the baseline
report logged "Widen Fail ... threshold of 0"): 8 samples/beat, so the
input burst is 1,250 beats instead of 10,000. The milestone kernels'
capture ports are widened the same way. The full kernel's `logits` port
is left narrow deliberately — 4x16 bits fits in a single beat either way.

### Explicitly skipped (and why)

* **Weights in URAM** — impossible for initialized ROMs on UltraScale+,
  see fix 3. Doing it silently zeroes the network on the board.
* **Packing weights into wider words in `weights.h`** — `weights.h` is
  auto-generated and shared with the float host build and the V1
  reference; packing would fork the format for a benefit the cyclic
  banking already delivers (the ROMs are no longer a BRAM problem).
* **Deduplicating the milestone kernels** — `radar_cnn_stream_stem`/
  `_dw1`/`_full` already share every layer template and `read_input`;
  the only "duplication" is their interface pragmas. They are separate
  top functions built in separate solutions, and only one is ever
  exported to a bitstream, so there is no resource cost on hardware.

### Pragma portability note

The new pragmas use template parameters as factors
(`ARRAY_PARTITION cyclic factor=UF`). Vitis HLS has supported
constant-expression/template pragma arguments since 2021.1; 2025.1
handles them. If a future tool version complains, replace the factor
with the literal value at each instantiation.

## Verify on the host (no Vitis needed)

```bash
cd hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming
g++ -std=c++17 -O2 -DRADAR_CNN_SOFTWARE_FLOAT -Wno-unknown-pragmas \
    -o tb_stream tb_stream.cpp radar_cnn_stream.cpp
./tb_stream ../../golden/input_samples.txt \
            ../../reports/fixed_point_validation/pytorch_reference.csv
```

Latest recorded result (2026-07-11, g++ 14.2 float mode):

* stem / DW1 kernels and all 13 per-stage template checks: **max error 0,
  0 mismatches** vs the V1-order reference;
* `radar_cnn_stream_full` end-to-end: **bit-exact** vs the reference;
* 8/8 golden radar samples: logits match PyTorch within **4.3e-6**
  (recorded V1 equivalence tolerance: 2e-4), all predictions agree.

## Run Vitis HLS

Vitis 2025.1 (this machine: `C:\Xilinx\2025.1`) ships `vitis-run` instead
of the retired `vitis_hls` launcher. From Git Bash:

```bash
export PATH="/c/Xilinx/2025.1/Vitis/bin:$PATH"
cd hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_stream.tcl   # V2A regression
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_dw1.tcl      # V2B regression
RADAR_CNN_HLS_MODE=csim vitis-run.bat --mode hls --tcl run_hls_full.tcl     # V2C full CNN
```

(From Windows CMD: `call C:\Xilinx\2025.1\Vitis\settings64.bat`, set the
env var with `set`, then `vitis-run --mode hls --tcl <script>.tcl`.)

Target part: `xck26-sfvc784-2LV-c` (KV260/K26) at 150 MHz, from the shared
`config/hardware_config.tcl`. Each milestone uses its own solution so
earlier reports are never overwritten. `RADAR_CNN_HLS_MODE` accepts
`csim`, `synth`, `cosim` or `all`; the full script also exports the IP
catalog on `all`.

Next steps once a Vitis host is available: csim in ap_fixed mode
(expected bit-exact vs V1 fixed, since every stage preserves V1's
accumulate/cast order and ap_fixed addition is exact), then synthesis to
replace the throughput/DSP estimates above with real numbers, then cosim
to validate the DATAFLOW FIFO depths.
