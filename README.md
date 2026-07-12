# KV260 Radar 2D CNN MVP

Radar target classification for the Kria KV260 using 100×100 micro-Doppler
spectrograms and a compact, DPU-friendly 2D CNN.

The existing DPU deployment path is:

```text
Raw IQ → CPU/PS spectrogram preprocessing → KV260 DPU CNN → CPU postprocessing
```

The model uses a `1×1 Conv2d` classifier rather than the older `Linear`
classifier. This change is what allowed the CNN inference path to compile to the
KV260 DPU.

The active custom-hardware path is:

```text
Raw IQ → CPU/PS top-3 spectrogram preprocessing → KV260 HLS DS-CNN → CPU postprocessing
```

---

## What changed on 2026-07-11 (cleanup + streaming HLS completion)

The repo was audited, cleaned and reorganized. Quick before/after:

| Area | Before | After |
|---|---|---|
| Repo size | ~1.5 GB (97% was one raw dataset file) | **~41 MB**; raw dataset moved to `../radar_data/radar_iq.npy` (outside the repo, see `data/README.md`) |
| Streaming HLS accelerator | Stem + DW1 only, and DW1 used a **wrong stride-2** assumption | **Complete 14-stage streaming CNN** (`radar_cnn_stream_full`), DW1 corrected to stride 1, verified bit-exact vs the V1 reference and 8/8 golden PyTorch samples (host float) |
| Legacy window-split trainers (`scripts/02_train*.py`) | Mixed in with live code; their random splits **leak recordings across train/test** | Quarantined in `deprecated/` with an explanatory README — do not use for reported results |
| Stale/dead files | `v2_streaming/` HLS snapshot, `model_old_linear.py`, `configs/default.yaml`, broken hardcoded path in `tb_real_samples.cpp` | Moved to `deprecated/` (or fixed: the testbench path is now an `argv` parameter) |
| Git | `.git` existed but was **empty** — no history, ignore rules inert | Initialized on `main`; `.gitignore` verified (dataset, `.venv`, output binaries excluded). **No commits yet** — run `git add . && git commit` to create the first one |
| Dependencies | `requirements_host.txt` loose `>=` bounds; training env versions never recorded | Still unpinned (nothing to pin from — see the note inside `requirements_host.txt` for the one-command fix) |

Where things live now:

```text
../radar_data/radar_iq.npy      raw 1.45 GiB dataset (OUTSIDE the repo)
data/real/*.npz                 small derived window caches (regenerable)
deprecated/                     quarantined legacy code — see deprecated/README.md
hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming/
                                ACTIVE streaming accelerator (full CNN)
hardware/hls/radar_cnn_hls_accel/v1_frame_buffered/
                                validated frame-buffered numerical reference
```

---

## Current status

- Selected preprocessing: stride32 windows with `topk_energy_mean`, `top_k=3`.
- Selected PyTorch model: capped-weight `RadarDSCNN2D`, chosen by validation
  macro F1 rather than test performance.
- ONNX export: complete.
- Vitis AI quantization and KV260 compilation: complete.
- Compiler mapping: `USER → DPU → CPU`.
- Fixed row-wise evaluation (`0→train`, `1→val`, `2→test`): implemented and evaluated.
- Grouped-stratified row-wise evaluation: implemented and evaluated with zero
  source-row overlap.
- Selected DS-CNN HLS weights and host-float equivalence: complete.
- Streaming HLS accelerator (V2C, complete CNN pipeline): host-float verified —
  bit-exact vs the V1 reference and within 4.3e-6 of PyTorch on all 8 golden
  samples. See `hardware/hls/radar_cnn_hls_accel/v2_streaming_v2b/v2_streaming/README.md`.
- Vitis HLS fixed-point C simulation and synthesis: pending. Vitis 2025.1 is
  installed at `C:\Xilinx\2025.1`; launch flows with
  `vitis-run --mode hls --tcl <script>.tcl` (the classic `vitis_hls`
  executable no longer exists in 2025.1). See `README_HLS.md` for commands.
- Physical KV260 VART execution and benchmarking: pending.

---

## Why KV260?

The Kria KV260 combines a quad-core Arm processing system with K26 programmable
logic, DSP resources, UltraRAM, and a supported Vitis AI DPU deployment flow.
That makes it suitable for the intended partition: CPU/PS spectrogram
preprocessing followed by accelerated CNN inference in programmable logic.

## Why DPU instead of custom HLS?

The DPU is the main deployment path because the trained CNN already quantized
and compiled successfully for `DPUCZDX8G`. It offers a shorter, supported route
to board execution than designing, integrating, and timing-closing a complete
custom CNN accelerator. The HLS implementation remains valuable for education,
architecture comparison, and future specialization.

## Selected deployment checkpoint

The selected deployment recipe is top-3 energy range-bin spectrogram
generation, DS-CNN inference, train-only augmentation, and capped class-weight
training. Grouped record-wise validation is used as the primary evidence
because it prevents source-recording leakage. The project now proceeds with
this selected checkpoint for custom HLS hardware acceleration.

The stable checkpoint is
`outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth`. It was selected from
completed capped-weight runs by best validation macro F1 (78.39% at epoch 35);
test metrics were not used for selection. Its complete recipe and SHA-256 are
recorded in `outputs/final_model/train_config.json`.

---

## Folder structure

```text
KV260_Radar_2D_CNN_MVP/
├── data/
│   └── real/
├── preprocessing/
│   └── 01_make_recordwise_windows.py
├── training/
│   └── 02_train_recordwise.py
├── evaluation/
│   ├── 03_eval_recordwise.py
│   ├── compare_results.py
│   ├── plot_training_curve.py
│   └── validate_onnx.py
├── quantization/
│   ├── quantize_vai.py
│   ├── compile_kv260.sh
│   └── inspect_xmodel.py
├── scripts/
│   ├── 00_check_dataset.py
│   ├── 01_make_spectrogram_cache.py
│   ├── 03_evaluate.py
│   ├── 04_export_onnx.py
│   ├── 05_quantize_vai_pytorch.py
│   ├── 06_compile_vitis_ai.sh
│   └── 08_run_kv260_vart.py
├── src/radar2dcnn/
│   ├── dataset.py
│   ├── labels.py
│   ├── metrics.py
│   ├── model.py
│   └── spectrogram.py
├── hardware/hls/radar_cnn_hls_accel/
│   ├── v1_frame_buffered/        (validated numerical reference)
│   └── v2_streaming_v2b/v2_streaming/   (ACTIVE streaming accelerator)
├── deprecated/                    (quarantined legacy code — see its README)
├── notebooks/
├── outputs/real/
├── outputs/recordwise/
├── tests/
├── requirements_host.txt
└── requirements_kv260.txt

../radar_data/radar_iq.npy         (raw dataset, kept OUTSIDE the repo)
```

---

## Classes

```text
0 = Drone
1 = Bird
2 = Human
3 = CornerReflector
```

---

## Host installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_host.txt
```

The root automation scripts reproduce each major stage:

```text
run_preprocessing.sh
run_train_recordwise.sh
run_quantization.sh
run_compile_kv260.sh
inspect_xmodel.sh
```

---

## Step 1: Check the windowed dataset

```bash
python scripts/00_check_dataset.py \
  --input data/real/radar_iq_windows_stride32_4class.npz
```

The window archive uses complex IQ tensors and integer labels:

```text
X: [N, 5, 256] complex64
y: [N] integer class IDs
```

---

## Step 2: Create the winning spectrogram cache

The current engineering result uses stride32 windows and the strongest selected
range bin (`range_mode=max`), not the earlier center-bin default.

```bash
python scripts/01_make_spectrogram_cache.py \
  --input data/real/radar_iq_windows_stride32_4class.npz \
  --output outputs/real/spectrogram_cache_max_stride32.npz \
  --image-size 100 \
  --range-mode max
```

---

## Step 3: Train the DPU-friendly CNN

This training path imports `src/radar2dcnn/model.py`, whose classifier is a
`1×1 Conv2d`. It does not use the old Linear-classifier model.

> **Deprecated:** this window-split trainer now lives in
> `deprecated/scripts/` (its random splits leak windows of the same
> recording across train/test). Prefer `training/02_train_recordwise.py`;
> the command below is kept only to document how the old DPU checkpoint
> was produced.

```bash
python deprecated/scripts/02_train_sampler_only_dpu.py \
  --data outputs/real/spectrogram_cache_max_stride32.npz \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.0005 \
  --output-dir outputs/real/train_max_stride32_4class_dpu
```

Deployable checkpoint:

```text
outputs/real/train_max_stride32_4class_dpu/best_model.pth
```

---

## Step 4: Evaluate

The saved window-level metrics and confusion matrix are available in:

```text
outputs/real/train_max_stride32_4class_dpu/test_metrics.json
outputs/real/train_max_stride32_4class_dpu/confusion_matrix.png
```

For the stricter record-wise pipeline:

```bash
python preprocessing/01_make_recordwise_windows.py \
  --input ../radar_data/radar_iq.npy \
  --output-dir outputs/recordwise \
  --stride 32 \
  --range-mode max \
  --split-strategy row_modulo

python training/02_train_recordwise.py
python evaluation/03_eval_recordwise.py
```

This applies source row `0→train`, row `1→val`, and row `2→test` cyclically,
then creates windows. The saved metadata verifies that the three row sets are
disjoint. Split caches are stored under `outputs/recordwise/train/`, `val/`, and
`test/`.

### Grouped top-k DSCNN result and next imbalance experiments

The improved pipeline is implemented separately from the historical row-modulo
result. It uses a grouped-stratified 70/15/15 source-row split, averages the
spectrograms of the three highest-energy range bins, applies augmentation only
to the training dataset, uses inverse-frequency weighted cross entropy, and
selects the best depthwise-separable CNN checkpoint by validation macro F1.

Run each stage explicitly:

```bash
bash scripts/run_preprocessing_recordwise_topk3.sh
bash scripts/run_train_recordwise_dscnn_topk3_aug.sh
bash scripts/run_eval_recordwise_dscnn_topk3_aug.sh
```

Outputs are written to `outputs/recordwise_topk3_stride32/`. Evaluation reports
accuracy, macro and weighted F1, per-class precision/recall/F1, the confusion
matrix, and recording-row overlap. Existing `outputs/recordwise/` results are
not overwritten.

The completed strong-weight run achieved 75.31% window accuracy, 54.81% macro
F1, and 84.51% weighted F1 with zero row overlap. Its test set contains 392
Drone windows but only 2 Bird, 5 Human, and 2 CornerReflector windows. Therefore
the accuracy and weighted F1 improvement is meaningful, while the single-split
macro F1 remains statistically unstable. The predicted distribution (293
Drone, 76 Bird, 30 Human, 2 CornerReflector) exposes minority-class
overprediction caused by the 34.84 CornerReflector loss weight.

Evaluation now also averages logits over all windows from each source row. The
current checkpoint classifies all nine held-out rows correctly, but 9/9 is too
small to present as a stable generalization estimate; grouped cross-validation
is required before making that claim.

Sampler-only and cap-8 runs are complete. Cap 8 currently gives the best window
result: 94.01% accuracy, 64.03% macro F1, and 95.75% weighted F1. Run the
remaining cap sweep without overwriting completed runs:

```bash
bash scripts/run_capped_weight_sweep.sh
```

| Weight strategy | Window accuracy | Window macro F1 | Recording accuracy | Recording macro F1 |
|---|---:|---:|---:|---:|
| Strong inverse weights | 75.31% | 54.81% | 100.00% | 100.00% |
| Sampler only | 88.53% | 59.55% | 100.00% | 100.00% |
| Capped weights, cap 4 | TBD | TBD | TBD | TBD |
| Capped weights, cap 6 | TBD | TBD | TBD | TBD |
| Capped weights, cap 8 | 94.01% | 64.03% | 88.89% | 73.08% |

The evaluator compares six recording aggregation methods. On the cap-8
checkpoint they all produce the same result, so none recovers the Bird row:

| Aggregation method | Recording accuracy | Macro F1 |
|---|---:|---:|
| Mean logits | 88.89% | 73.08% |
| Mean softmax probability | 88.89% | 73.08% |
| Majority vote | 88.89% | 73.08% |
| Median probability | 88.89% | 73.08% |
| Max class confidence | 88.89% | 73.08% |
| Top-30% confidence average | 88.89% | 73.08% |

The Bird row contains two windows split between Drone and Bird; all confidence
tie-breaks favor Drone. The aggregation comparison is therefore negative but
useful evidence. Results are saved in
`recording_aggregation_comparison.{json,md}` inside each run directory.

Run nested five-fold grouped validation with capped weights:

```bash
bash scripts/run_cross_validation_recordwise_topk3_capped.sh
```

Each recording row appears in an outer test fold exactly once. An inner grouped
split supplies validation data for early stopping. The final report contains
mean and sample standard deviation for window- and recording-level accuracy,
macro F1, and weighted F1.

The controlled comparisons change one pipeline component at a time:

| Experiment | Model | Range mode | Train augmentation |
|---|---|---|---|
| 1 | `radar2dcnn` | `max` | no |
| 2 | `radar2dcnn` | `max` | yes |
| 3 | `radar2dcnn` | `topk_energy_mean --top-k 3` | yes |
| 4 | `radar_dscnn2d` | `topk_energy_mean --top-k 3` | yes |

---

## Step 5: Export ONNX

```bash
python scripts/04_export_onnx.py \
  --checkpoint outputs/real/train_max_stride32_4class_dpu/best_model.pth \
  --output outputs/real/radar_2dcnn_dpu_stride32.onnx
```

---

## Step 6: Vitis AI quantization

Run this inside the Vitis AI PyTorch environment:

```bash
conda activate vitis-ai-pytorch

python scripts/05_quantize_vai_pytorch.py \
  --data outputs/real/spectrogram_cache_max_stride32.npz \
  --checkpoint outputs/real/train_max_stride32_4class_dpu/best_model.pth \
  --quant-mode calib \
  --output-dir outputs/real/vai_quant_dpu_stride32

python scripts/05_quantize_vai_pytorch.py \
  --data outputs/real/spectrogram_cache_max_stride32.npz \
  --checkpoint outputs/real/train_max_stride32_4class_dpu/best_model.pth \
  --quant-mode test \
  --output-dir outputs/real/vai_quant_dpu_stride32
```

Quantized model:

```text
outputs/real/vai_quant_dpu_stride32/Radar2DCNN_int.xmodel
```

---

## Step 7: Compile for KV260 DPU

```bash
bash scripts/06_compile_vitis_ai.sh \
  outputs/real/vai_quant_dpu_stride32/Radar2DCNN_int.xmodel \
  outputs/real/compiled_kv260_dpu
```

Compiled model:

```text
outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel
```

---

## DPU compatibility result

### Linear classifier problem

The first CNN used a `Linear` classifier after feature extraction. Although it
trained successfully, the Vitis AI compiler mapped that classifier path to CPU:

```text
USER → CPU
```

### 1×1 Conv2d DPU workaround

Replacing the final `Linear` classifier with a `1×1 Conv2d` produced the
operator pattern needed by the KV260 DPU compiler.

### Vitis AI compiler subgraph result

The successful compile reported:

```text
0 subgraph_Radar2DCNN__input_0                                      USER 1
1 subgraph_Radar2DCNN__Radar2DCNN_Conv2d_classifier_conv__ret_27   DPU  22
2 subgraph_Radar2DCNN__Radar2DCNN_ret_fix_                         CPU  1
```

In short:

```text
USER → DPU → CPU
```

This confirms that the main CNN inference path is mapped to the KV260 DPU.

---

## Step 8: Run on KV260

Copy these files to the board:

```text
outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel
outputs/real/spectrogram_cache_max_stride32.npz
scripts/08_run_kv260_vart.py
src/radar2dcnn/
```

Then run in the KV260 Vitis AI environment:

```bash
python3 scripts/08_run_kv260_vart.py \
  --xmodel radar_2dcnn.xmodel \
  --data spectrogram_cache_max_stride32.npz \
  --num-samples 20
```

---

## Reproduce results

### Reproduce the current window-level DPU result

Run quantization inside the Vitis AI PyTorch environment and compilation inside
a Vitis AI compiler environment:

```bash
./run_preprocessing.sh
./deprecated/run_train_window_split.sh   # legacy window-split trainer (leaky splits; see deprecated/README.md)
./run_quantization.sh
./run_compile_kv260.sh
./inspect_xmodel.sh
```

### Reproduce the record-wise validation result

`run_preprocessing.sh` creates the required row-wise caches as well as the
window-level cache. Then run:

```bash
./run_train_recordwise.sh
```

That script trains, evaluates, and regenerates the validation comparison. To
reevaluate an existing record-wise checkpoint without retraining:

```bash
python evaluation/03_eval_recordwise.py
```

---

## Current best artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Window-level DPU checkpoint | `outputs/real/train_max_stride32_4class_dpu/best_model.pth` | Best engineering/demo model and checkpoint corresponding to the compiled DPU path |
| ONNX model | `outputs/real/radar_2dcnn_dpu_stride32.onnx` | Sanity check and portability |
| Quantized Vitis AI xmodel | `outputs/real/vai_quant_dpu_stride32/Radar2DCNN_int.xmodel` | Quantized model before KV260 compilation |
| Compiled KV260 xmodel | `outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel` | Model to execute on the KV260 DPU |
| Winning spectrogram cache | `outputs/real/spectrogram_cache_max_stride32.npz` | Stride32/max input used for window-level training and evaluation |
| Record-wise checkpoint | `outputs/recordwise/best_model.pth` | Stricter validation model trained on fixed, disjoint recording rows |
| Record-wise metrics | `outputs/recordwise/test_metrics.json` | Stricter validation result and confusion matrix data |
| Selected DS-CNN checkpoint | `outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth` | Stable model selected by validation macro F1 |
| Selected training recipe | `outputs/final_model/train_config.json` | Model, preprocessing, split, weighting, and checkpoint provenance |

The existing compiled DPU artifact remains the board-demo model. The selected
capped-weight DS-CNN is now the primary ML checkpoint and the source for the
custom HLS path.

---

## Custom HLS accelerator

The compiled KV260 Vitis AI DPU model remains the proven board-deployment
baseline:

```text
outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel
```

An optional functional-first HLS implementation is available in:

```text
hardware/hls/radar_cnn_hls_accel/
```

The accelerator implements spectrogram-to-logits CNN inference. It does not
accelerate raw IQ-to-spectrogram processing; that remains on the PS/CPU.

Export HLS weights from the selected validation checkpoint:

```text
outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth
```

The checked-in HLS kernel implements the selected depthwise/pointwise DS-CNN,
and `weights.h` was generated from that checkpoint. Host-float comparison on a
real cached sample matches PyTorch with maximum absolute logit error `1.67e-6`.
Fixed-point Vitis HLS simulation and synthesis are still required.

HLS status:

- **Selected DS-CNN host-float PyTorch equivalence:** complete (`1.67e-6` max
  absolute logit error on the checked real sample).
- **Fixed-point C simulation:** pending.
- **C synthesis:** pending.
- **Timing closure:** pending.
- **Board execution:** pending.

---

## Important engineering note

This project has three validation views.

- **Window-level evaluation:** completed using overlapping stride32 windows.
- **Window evaluation on held-out recordings:** reports window predictions while
  keeping source rows disjoint.
- **Recording-level evaluation:** averages the logits of all windows belonging
  to one held-out source row.

### Window-level results

The DPU-friendly window-level model achieved:

```text
Window-level test accuracy : 96.93%
Macro F1-score             : 92.54%
Weighted F1-score          : 96.98%
```

Overlapping windows from the same original recording may appear in both the
training and test partitions for this result. It is therefore an
engineering/MVP result, not the final scientific claim.

### Record-wise results

The stricter record-wise run achieved:

```text
Record-wise test accuracy  : 43.74%
Macro F1-score             : 52.79%
Weighted F1-score          : 51.58%
Recording-row overlap      : 0
```

| Validation type | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| Window-level stride32 | 96.93% | 92.54% | 96.98% | Engineering/MVP result |
| Historical row-modulo strict split | 43.74% | 52.79% | 51.58% | Row-disjoint baseline |
| Grouped top-k DSCNN, window predictions | 75.31% | 54.81% | 84.51% | Strong inverse weights; row overlap 0 |
| Grouped top-k DSCNN, recording predictions | 100.00% | 100.00% | 100.00% | Only nine held-out rows; not yet stable |

The record-wise test split contains 580 Drone, 56 Bird, 55 Human, and 4
CornerReflector windows. It is more defensible than the window-level split, but
additional independent CornerReflector recordings are still required.

The grouped top-k result is the current strictest single-split result because no
source recording row overlaps between train, validation, and test. Five-fold
grouped validation is the next required evidence because minority-class support
is too small for a trustworthy single-split macro F1.

---

## Evidence artifacts

| Evidence | Path |
|---|---|
| Spectrogram image grid | `outputs/real/day2_real_spectrogram_grid.png` |
| Window-level confusion matrix | `outputs/real/train_max_stride32_4class_dpu/confusion_matrix.png` |
| Record-wise confusion matrix | `outputs/recordwise/confusion_matrix.png` |
| Window-level training curve | `outputs/evidence/training_curve_window.png` |
| Record-wise training curve | `outputs/evidence/training_curve_recordwise.png` |
| ONNX validation report | `outputs/evidence/onnx_validation.json` |
| Xmodel subgraph output | `outputs/evidence/xmodel_subgraph_mapping.txt` |
| Quantization evidence | `outputs/evidence/quantization_status.md` |
| HLS equivalence evidence | `outputs/evidence/hls_validation_status.md` |
| Architecture diagram | `docs/architecture.md` |
| Validation comparison | `outputs/recordwise/comparison.md` |
| Recording aggregation comparison | `outputs/recordwise_topk3_stride32_capped8/recording_aggregation_comparison.md` |
| Imbalance strategy comparison | `outputs/imbalance_strategy_comparison/comparison.md` |

---

## Known limitations

- Physical KV260 VART execution and latency benchmarking are pending.
- Window-level accuracy may be optimistic because of overlapping stride32
  windows from shared recordings.
- The grouped test split has only one Bird, one Human, and one CornerReflector
  source row, so its perfect recording-level result is not a stable estimate.
- Five-fold grouped validation is implemented but has not yet been trained.
- IQ-to-spectrogram preprocessing currently runs on CPU/PS, not in PL.
- The custom HLS path has passed host-float equivalence testing, but fixed-point
  C simulation, synthesis, timing closure, and board execution remain pending.

---

## TODO

- [x] Implement fixed recording-row train/validation/test splitting.
- [x] Retrain the DPU-friendly CNN with the record-wise split.
- [x] Generate the record-wise confusion matrix and metrics.
- [x] Add grouped-stratified source-row splitting.
- [x] Add top-k energy spectrogram averaging and train-only augmentation.
- [x] Add a DPU-friendly depthwise-separable CNN experiment.
- [x] Report predicted class distributions and averaged-logit recording metrics.
- [x] Add sampler-only, capped, square-root, and effective-number weighting.
- [x] Add nested five-fold grouped validation.
- [x] Train sampler-only and cap-8 controlled experiments.
- [x] Compare mean-logit, probability, vote, maximum, and top-confidence aggregation.
- [ ] Train capped-weight experiments with caps 4 and 6.
- [ ] Run five-fold grouped validation and report mean ± standard deviation.
- [ ] Run and compare all four controlled record-wise experiments.
- [ ] Run `radar_2dcnn.xmodel` on a physical KV260 using VART.
- [ ] Benchmark DPU-only and end-to-end latency.
- [ ] Add a KV260 live/demo notebook.
- [x] Add the final architecture diagram in `docs/architecture.md`.

---

## Dry run with synthetic data

Use this only to test the software pipeline:

```bash
python scripts/99_generate_toy_dataset.py \
  --output data/toy_radar_iq.npz \
  --samples-per-class 64

python scripts/01_make_spectrogram_cache.py \
  --input data/toy_radar_iq.npz \
  --output outputs/toy_cache.npz \
  --range-mode max

python deprecated/scripts/02_train_sampler_only_dpu.py \
  --data outputs/toy_cache.npz \
  --epochs 3 \
  --batch-size 16 \
  --output-dir outputs/toy_train
```

Synthetic data is for pipeline debugging only and is not scientifically valid.
