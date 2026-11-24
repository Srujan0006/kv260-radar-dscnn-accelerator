# Architecture

## System partition

```mermaid
flowchart LR
    IQ[Raw complex IQ] --> WIN[256-sample windows, stride 32]
    WIN --> SPEC[PS: top-3 energy bins → STFT → 100×100 image]
    SPEC --> ACC[PL: streaming HLS accelerator]
    ACC --> LOG[4 logits]
    LOG --> DEC[PS: argmax / recording-level mean of logits]
```

Windowing and spectrogram generation run in software on the Arm processing
system. CNN inference runs in programmable logic.

## Spectrogram pipeline (`src/radar2dcnn/spectrogram.py`)

For each window of 5 range bins × 256 slow-time samples:

1. Rank the range bins by mean power and keep the top 3.
2. For each kept bin: remove DC, run an STFT (Hann 64, overlap 56, 128-point FFT,
   two-sided, fftshift), take power, log-scale, resize bilinearly to 100×100,
   then min-max normalize to [0, 1].
3. Average the three images.

## RadarDSCNN2D (`src/radar2dcnn/model.py`)

| Stage | Operation | Output | Params |
|---|---|---|---:|
| Stem | Conv 3×3, 1→16, BN, ReLU | 16×100×100 | 176 |
| Block 1 | DW 3×3 + PW 1×1, 16→32, MaxPool | 32×50×50 | 752 |
| Block 2 | DW 3×3 + PW 1×1, 32→64, MaxPool | 64×25×25 | 2,528 |
| Block 3 | DW 3×3 + PW 1×1, 64→96, MaxPool | 96×12×12 | 7,040 |
| Block 4 | DW 3×3 + PW 1×1, 96→128 | 128×12×12 | 13,600 |
| Head | Global average pool, Conv 1×1 128→4 | 4 | 516 |
| **Total** | | | **24,612** |

About 19.9 M MACs per image. The classifier is a 1×1 convolution rather than a
`Linear` layer, so the whole graph stays inside the operators the Vitis AI DPU
supports.

## Streaming accelerator (`6_hardware/hls/radar_cnn_hls_accel/v2_streaming/`)

The whole network is one `#pragma HLS DATAFLOW` region with 16 stages connected
by `hls::stream` FIFOs: AXI reader → stem → 4 × (DW → PW [→ pool]) → GAP →
classifier → AXI writer. No feature map is ever stored in full. Depthwise
stages keep two line-buffer rows, pools keep one row of running maxima, and GAP
keeps 128 accumulators.

- **Arithmetic:** `data_t = ap_fixed<16,6>` (Q6.10), `acc_t = ap_fixed<32,12>`
  (truncate/wrap defaults). Each stage starts accumulating from the bias and
  adds channels in ascending order, so a float build is bit-comparable with
  the reference.
- **Interfaces:** `s_axi_control` (control and buffer pointers), `m_axi_gmem0`
  (input, widened to 128-bit beats), `m_axi_gmem1` (4 logits), interrupt.
- **Weights:** BatchNorm is folded into compile-time ROMs in `weights.h`, kept
  in BRAM/LUT-RAM. URAM is avoided because it cannot be initialized from the
  bitstream.
- **Key optimizations:** flattened (row, col, channel-block) depthwise pipelines
  at II=1; inter-stage FIFOs forced to SRL; pointwise weights re-banked per
  output lane.

## Vitis AI DPU path

`outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel` targets
`DPUCZDX8G_ISA1_B4096` with subgraphs USER → DPU → CPU. It was compiled from
the earlier plain `Radar2DCNN` (4 standard 3×3 conv layers, max-energy-bin
input), **not** from RadarDSCNN2D.
