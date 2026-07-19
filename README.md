# Radar Target Classification on the Kria KV260 — Streaming 2D CNN Accelerator

![PyTorch](https://img.shields.io/badge/PyTorch-DS--CNN%2024.6k%20params-ee4c2c)
![Vitis HLS](https://img.shields.io/badge/Vitis%20HLS-C%2FRTL%20cosim%20PASS-blue)
![Vivado](https://img.shields.io/badge/Vivado-timing%20met%20%40%2097%20MHz-blue)
![Verification](https://img.shields.io/badge/streaming%20vs%20reference-bit--exact-brightgreen)
![Accuracy](https://img.shields.io/badge/5--fold%20accuracy-80.6%25-green)

Classifying radar targets — **Drone / Bird / Human / Corner Reflector** — from
micro-Doppler spectrograms, all the way from raw IQ recordings to a custom
streaming CNN accelerator that is placed, routed, and timing-clean on the AMD
Kria KV260.

Two deployment routes are implemented from one trained checkpoint:

```text
                        ┌──> Route A: ONNX ──> Vitis AI INT8 ──> KV260 DPU (.xmodel)
Raw IQ ─> spectrograms ─┤
  (1.45 GiB, 130 recs)  └──> Route B: fixed-point HLS ──> streaming DATAFLOW IP
                                     ──> Vivado system @ ~97 MHz ──> XSA (board run pending)
```

---

## Motivation

Detection tells you *something* is out there; classification tells you *what*.
A drone's spinning propellers, a bird's wingbeats, a walking human's limbs, and
a rigid corner reflector each modulate a radar echo in a characteristic,
time-varying way. Turning that modulation into a 100×100 time–frequency image
(a micro-Doppler spectrogram) turns target classification into an image
classification problem — one a small CNN handles well.

A CNN on an embedded CPU is slow and non-deterministic. An FPGA runs the same
multiply-accumulate work in deep fixed-point pipelines with *identical* latency
on every frame — exactly what a radar processing chain wants. The KV260 pairs
an Arm PS with K26 programmable logic, so preprocessing stays in software while
inference runs in fabric.

## The dataset

A single `radar_iq.npy` object array (1.45 GiB, kept **outside** the repo —
see [data/README.md](data/README.md)): 130 recordings, each an identifier plus
a complex IQ matrix of 5 range bins × slow-time samples. After a length check,
59 recordings survive. Each is cut into 256-sample windows at stride 32; the 3
highest-energy range bins are converted to spectrograms and averaged
(64-sample Hann STFT, hop 8, 128-point FFT, power → log → resize → min-max).

## The validation trap this project refused to fall into

Windows cut 32 samples apart from one recording are near-duplicates. Split
them randomly into train/test and the test set is full of near-copies of
training data: our first model scored an impressive-looking **96.9%** that
way. The number is a mirage.

The fix is **recording-disjoint splitting**: every window of a recording stays
in exactly one partition, verified programmatically (overlap = 0), with
grouped five-fold cross-validation as the primary evidence:

| Evaluation | Accuracy | Macro F1 | Verdict |
|---|---:|---:|---|
| Random window split | 96.93% | 92.54% | leaky — reference only |
| Strict single split | 43.74% | 52.79% | honest floor |
| Best grouped single split | 94.26% | 64.42% | sensitive to rare recordings |
| **Grouped 5-fold (primary)** | **80.57% ± 2.98%** | **66.45% ± 5.29%** | zero overlap in every fold |

Aggregating window logits per recording lifts accuracy to **86.7% ± 11.2%**.
The drop from 97% to 81% isn't a failure — it *is* the finding.

## The model

`RadarDSCNN2D` ([src/radar2dcnn/model.py](src/radar2dcnn/model.py)) — a
depthwise-separable CNN of only **24,612 parameters**, small enough that every
weight lives on-chip:

```text
Input 1×100×100
  Stem   Conv3×3 1→16            (100×100)
  Block1 DW3×3 + PW1×1 16→32  ─ pool → 50×50
  Block2 DW3×3 + PW1×1 32→64  ─ pool → 25×25
  Block3 DW3×3 + PW1×1 64→96  ─ pool → 12×12
  Block4 DW3×3 + PW1×1 96→128   (12×12, no pool)
  Global average pool → 128 → Conv1×1 → 4 logits
```

The classifier is a 1×1 convolution instead of `nn.Linear` — a deliberate
choice that lets the Vitis AI compiler map the *entire* graph onto the DPU
(`DPUCZDX8G_ISA1_B4096`, single DPU subgraph). ONNX export matches PyTorch to
2.6e-6.

## The hardware story: two designs, one lesson

**Attempt 1 — frame-buffered.** Each layer computes its full output feature
map before the next starts. Easy to write, easy to trust (it became the
numerical reference), impossible to fit: the stem output alone is 16×100×100
values, and the design needed ~1,600 BRAM against the K26's 288. No pragma
could fix it — the *data movement* was wrong.

**Attempt 2 — streaming (the shipped design).** One `#pragma HLS DATAFLOW`
region, 17 concurrent processes, every stage connected by `hls::stream` FIFOs
carrying one pixel's worth of channels per token. A 3×3 conv remembers just
two rows in a line buffer; pooling keeps one row; GAP keeps 128 accumulators.
**No feature map is ever stored.** Arithmetic is `ap_fixed<16,6>` (Q6.10) with
`ap_fixed<32,12>` accumulators, weights baked into BRAM/LUT-ROM (URAM is
deliberately avoided — it can't initialize from the bitstream).

A synthesis-report-driven optimization round then flattened the depthwise
pipelines (up to 55× fewer cycles per stage), forced the wide inter-stage
FIFOs into shift registers instead of BRAM, and re-banked the pointwise weight
ROMs — taking the design from *not fitting* (816 BRAM / 202k LUT) to fitting
with room to spare, without changing a single output bit.

## Verification: bit-exact or it didn't happen

The same kernel source compiles either as `ap_fixed` (synthesis) or as `float`
on plain g++ (`-DRADAR_CNN_SOFTWARE_FLOAT`) — so numerical bugs are caught on
the host in seconds. Every streaming stage follows a strict operation-order
contract (accumulate from bias, ascending channel/kernel order, cast, ReLU),
which makes comparisons against the frame-buffered reference **exact**, not
tolerance-based:

| Check | Result |
|---|---|
| PyTorch → ONNX | max err 2.6e-6 |
| PyTorch → host-float C++ (8 golden samples) | ≤ 4.3e-6, 8/8 classes correct |
| V1 reference → V2 streaming (every stage) | **bit-exact, error 0** |
| Fixed-point C simulation | PASS, 0 errors |
| C/RTL co-simulation (xsim) | **PASS**, logits bit-identical |
| Vivado implementation (full system) | **timing met**, 0 routing errors |

## Results

**Accelerator IP** (Vitis HLS, xck26-sfvc784-2LV-c):

| Metric | Value |
|---|---|
| RTL cosim latency (measured) | 112,020 cycles |
| Frame time @ 96.97 MHz system clock | **1.16 ms ≈ 865 inferences/s** |
| Routed resources (standalone IP) | 95,735 LUT · 599 DSP · 115 BRAM_18K · 0 URAM |

**Full system** (Vivado block design: Zynq PS + SmartConnect + accelerator):

| Metric | Value |
|---|---|
| Timing | **all constraints met** (setup WNS +1.137 ns, hold met) |
| Clock | PS `pl_clk0` ≈ 97 MHz |
| Utilization | LUT 82.4% · DSP 48% · BRAM 40% · FF 28% |
| Interfaces | AXI-Lite control @ `0xA000_0000` · 2× AXI-master → DDR via HPC0 · interrupt → `pl_ps_irq0` |
| Power (tool estimate) | ~3.5 W |
| Hardware handoff | XSA exported (PS + bitstream) |

Full reports: [hardware/hls/radar_cnn_hls_accel/reports/](hardware/hls/radar_cnn_hls_accel/reports/).
**Still pending:** the physical board benchmark (latency / power / accuracy
measured on the KV260) — the baremetal test app for it is ready in
[deployment/kv260_baremetal/](deployment/kv260_baremetal/).

## Repository structure

```text
├── preprocessing/        recording-disjoint windowing + spectrogram cache
├── training/             record-wise trainer + 5-fold cross-validation
├── evaluation/           metrics, confusion matrices, ONNX validation
├── quantization/         Vitis AI INT8 quantize + KV260 DPU compile
├── scripts/              pipeline utilities (ONNX export, DPU quantize/compile, board runner)
│   └── legacy/           historical run wrappers, kept for provenance
├── src/radar2dcnn/       model, dataset, spectrogram, imbalance strategies
├── hardware/hls/radar_cnn_hls_accel/
│   ├── v1_frame_buffered/        numerical reference implementation
│   ├── v2_streaming_v2b/v2_streaming/   ACTIVE streaming accelerator
│   ├── reports/                  synthesis / cosim / implementation evidence
│   └── tools/                    weight export (Conv+BN folding), comparators
├── deployment/kv260_baremetal/   Vitis baremetal board-test app + docs
├── outputs/              metrics, confusion matrices, evidence artifacts
├── tests/                pytest suite (model shapes, splits, spectrograms)
└── deprecated/           quarantined legacy code — do not use (see its README)
```

## How to reproduce

```bash
pip install -r requirements_host.txt

# 1. Data → windows → spectrogram caches (raw .npy lives outside the repo)
python preprocessing/01_make_recordwise_windows.py --input ../radar_data/radar_iq.npy

# 2. Train + evaluate (recording-disjoint)
python training/02_train_recordwise.py --imbalance-strategy capped --augmentation
python evaluation/03_eval_recordwise.py
python training/03_cross_validate_recordwise.py     # 5-fold primary result

# 3. HLS host verification — no Vitis needed, runs in seconds
cd hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming
g++ -std=c++17 -O2 -DRADAR_CNN_SOFTWARE_FLOAT -Wno-unknown-pragmas \
    -o tb_stream tb_stream.cpp radar_cnn_stream.cpp
./tb_stream ../../golden/input_samples.txt \
            ../../reports/fixed_point_validation/pytorch_reference.csv

# 4. Full HLS flow (csim / synth / cosim) — Vitis HLS required
RADAR_CNN_HLS_MODE=all vitis_hls -f run_hls_full.tcl
```

### Route A: DPU deployment (Vitis AI 3.5 docker)

```bash
# Export ONNX and validate against PyTorch
python scripts/04_export_onnx.py \
  --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
  --output outputs/real/radar_2dcnn.onnx

# INT8 quantization (inside the vitis-ai-pytorch conda env): calibrate, then test
python scripts/05_quantize_vai_pytorch.py --quant-mode calib \
  --checkpoint <ckpt> --data <spectrogram_cache.npz> --output-dir outputs/real/vai_quant
python scripts/05_quantize_vai_pytorch.py --quant-mode test \
  --checkpoint <ckpt> --data <spectrogram_cache.npz> --output-dir outputs/real/vai_quant

# Compile for the KV260 DPU and inspect the subgraph mapping
bash scripts/06_compile_vitis_ai.sh outputs/real/vai_quant/Radar2DCNN_int.xmodel outputs/real/compiled_kv260
python quantization/inspect_xmodel.py outputs/real/compiled_kv260/radar_2dcnn.xmodel

# On the board (Vitis AI runtime):
python3 scripts/08_run_kv260_vart.py --xmodel radar_2dcnn.xmodel \
  --data spectrogram_cache.npz --num-samples 20
```

### Extra experiments

```bash
# Compare imbalance strategies (one flag, not separate scripts)
python training/02_train_recordwise.py --imbalance-strategy inverse
python training/02_train_recordwise.py --imbalance-strategy sampler_only
python training/02_train_recordwise.py --imbalance-strategy capped --cap 8

# Sanity-check the windowed dataset / quick CPU demo / toy pipeline test
python scripts/00_check_dataset.py --input data/real/radar_iq_windows_stride32_4class.npz
python scripts/07_run_cpu_demo.py
python scripts/99_generate_toy_dataset.py --output data/toy_radar_iq.npz
```

The exact shell wrappers used during development are preserved unmodified in
[scripts/legacy/](scripts/legacy/) for provenance.

## Struggles and lessons learned

- **A great score can be a bug.** The 97%→81% correction after fixing window
  leakage was the most valuable result in the project.
- **Memory architecture beats pragmas.** The frame-buffered design could not
  be tuned into fitting; the fix was changing how data moves, not directives.
- **Make verification exact, then debugging is local.** Because every stage
  reproduces the reference operation order bit-for-bit, any failure points at
  one stage — never at an accumulated tolerance.
- **HLS estimates are pessimistic.** 93% estimated LUT routed at 82%; 1,018
  estimated DSPs became 599 after Vivado's multiplier packing.
- **Windows path length is real:** Vitis generates RTL filenames long enough
  to break the 260-char MAX_PATH inside a deep repo — synthesize from a short
  working directory.

## Future work

- Physical KV260 benchmark (the last unchecked box): latency, power, and
  held-out accuracy on the board, CPU vs DPU vs custom-HLS under one protocol.
- Move STFT/spectrogram generation into the PL for a camera-to-logits pipeline.
- Widen the early pointwise stages (the ~80k-cycle throughput bottleneck).
- Unknown-target rejection and confidence calibration.
- Runtime-loadable weights (currently compile-time constants).

## References

- V. C. Chen, *The Micro-Doppler Effect in Radar*, Artech House, 2019
- Howard et al., *MobileNets*, arXiv:1704.04861 — depthwise-separable convolutions
- AMD UG1399 (Vitis HLS), UG1414 (Vitis AI), UG1089 (Kria KV260)
