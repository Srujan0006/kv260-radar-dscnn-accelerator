# Training and evaluation

## Environment

```bash
pip install -r requirements_host.txt
python -m pytest
```

`requirements_host.txt` gives lower bounds only. The exact library versions
used for the published checkpoint were not recorded.

## Data split

Windows from the same recording are almost identical (87.5 % overlap), so all
splits group by `source_row_index`.

- Single split: `grouped_stratified_recording_rows`, ratios 0.70 / 0.15 / 0.15,
  seed 42.
- Cross-validation: 5 outer folds × 4 inner folds, seed 42.
- Every manifest records `recording_row_overlap: 0`.

## Pipeline

```bash
# windows, split and spectrogram caches
python 2_preprocessing/make_recordwise_windows.py --input ../radar_data/radar_iq.npy \
       --range-mode topk_energy_mean --top-k 3 --stride 32
# train (AdamW 1e-3, wd 1e-4, batch 32, ReduceLROnPlateau, early stop on val macro-F1)
python 3_training/train_recordwise.py --model radar_dscnn2d --augmentation \
       --imbalance-strategy capped --max-class-weight 8
python 4_evaluation/eval_recordwise.py
python 3_training/cross_validate_recordwise.py --cache-root outputs/recordwise
```

## Augmentation (training split only, `src/radar2dcnn/dataset.py`)

| Operation | Range | Probability |
|---|---|---|
| Amplitude scaling | U[0.85, 1.15] | always |
| Gaussian noise | σ = 0.02 | 0.5 |
| Circular time shift | ±5 columns | 0.5 |
| Circular Doppler shift | ±4 rows | 0.5 |
| Time mask | 1–10 columns | 0.5 |
| Frequency mask | 1–8 rows | 0.5 |
| Crop and resize | ≤ 5 % per edge | 0.25 |

## Exports

```bash
python scripts/export_onnx.py --help
python 5_quantization/quantize_vai.py --help     # inside the Vitis AI 3.5 container
bash 5_quantization/compile_kv260.sh
python 6_hardware/hls/radar_cnn_hls_accel/tools/export_weights_to_hls.py \
       --checkpoint outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth \
       --output 6_hardware/hls/radar_cnn_hls_accel/v2_streaming/weights.h
```
