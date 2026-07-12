# One-Week Execution Plan

> **Historical document (kept as-is).** Commands referencing `scripts/02_train*.py` now live in `deprecated/scripts/` and use leaky window-level splits -- use `training/02_train_recordwise.py` instead. The raw dataset now lives at `../radar_data/radar_iq.npy`.

## Day 1: Dataset and baseline

- Download/prepare the radar IQ dataset.
- Convert it into `data/radar_iq.npz` with keys `X` and `y`.
- Run:

```bash
python scripts/00_check_dataset.py --input data/radar_iq.npz
```

Exit criteria:

- Dataset loads.
- Label distribution is correct.
- First 5 spectrograms are generated without error.

## Day 2: Spectrogram cache

Run:

```bash
python scripts/01_make_spectrogram_cache.py \
  --input data/radar_iq.npz \
  --output outputs/spectrogram_cache.npz \
  --image-size 100 \
  --range-mode center
```

Try `--range-mode mean` and `--range-mode max` if needed.

Exit criteria:

- `outputs/spectrogram_cache.npz` created.
- Shape is `[N, 1, 100, 100]`.

## Day 3: Train first 2D CNN

Run debug training first:

```bash
python scripts/02_train.py --data outputs/spectrogram_cache.npz --epochs 3 --batch-size 32 --output-dir outputs/debug_train
```

Then full training:

```bash
python scripts/02_train.py --data outputs/spectrogram_cache.npz --epochs 50 --batch-size 64 --output-dir outputs/train_run
```

Exit criteria:

- `outputs/train_run/best_model.pth` exists.
- Confusion matrix generated.
- Accuracy is acceptable for first demo.

## Day 4: Tuning

Try combinations:

```bash
# Rebuild cache using mean range mode
python scripts/01_make_spectrogram_cache.py --input data/radar_iq.npz --output outputs/spectrogram_mean.npz --range-mode mean
python scripts/02_train.py --data outputs/spectrogram_mean.npz --epochs 50 --output-dir outputs/train_mean

# Rebuild cache using max-energy range mode
python scripts/01_make_spectrogram_cache.py --input data/radar_iq.npz --output outputs/spectrogram_max.npz --range-mode max
python scripts/02_train.py --data outputs/spectrogram_max.npz --epochs 50 --output-dir outputs/train_max
```

Pick the best validation/test accuracy.

## Day 5: Export and quantize

Export ONNX:

```bash
python scripts/04_export_onnx.py --checkpoint outputs/train_run/best_model.pth --output outputs/radar_2dcnn.onnx
```

Inside Vitis AI container:

```bash
conda activate vitis-ai-pytorch
python scripts/05_quantize_vai_pytorch.py --data outputs/spectrogram_cache.npz --checkpoint outputs/train_run/best_model.pth --quant-mode calib --output-dir outputs/vai_quant
python scripts/05_quantize_vai_pytorch.py --data outputs/spectrogram_cache.npz --checkpoint outputs/train_run/best_model.pth --quant-mode test --output-dir outputs/vai_quant
```

Exit criteria:

- Quantized xmodel generated.

## Day 6: Compile and run on KV260

Compile:

```bash
bash scripts/06_compile_vitis_ai.sh outputs/vai_quant/Radar2DCNN_int.xmodel outputs/compiled_kv260
```

Copy compiled model and data to KV260, then run:

```bash
python3 scripts/08_run_kv260_vart.py --xmodel radar_2dcnn.xmodel --data spectrogram_cache.npz --num-samples 20
```

Exit criteria:

- KV260 inference prints predictions.

## Day 7: Prepare demo and report

Prepare these outputs:

- Dataset summary.
- Spectrogram sample images.
- Training curve/history.
- Confusion matrix.
- CPU inference demo.
- DPU inference demo if available.
- Explain fallback plan if DPU environment is not ready.

