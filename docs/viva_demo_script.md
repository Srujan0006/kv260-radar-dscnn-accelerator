# Viva / Demo Script

> **Historical document (kept as-is).** Commands referencing `scripts/02_train*.py` now live in `deprecated/scripts/` and use leaky window-level splits -- use `training/02_train_recordwise.py` instead. The raw dataset now lives at `../radar_data/radar_iq.npy`.

## Opening

This project upgrades a radar target classifier from a compact PYNQ-Z2 1D CNN design to a Kria KV260-ready 2D CNN design. The main reason for using KV260 is that the 2D CNN requires significantly more FPGA acceleration resources than the PYNQ-Z2 can provide.

## Dataset explanation

The radar input is a complex IQ segment. Each segment has 1280 complex samples, interpreted as five range cells and 256 sweeps. These IQ samples are converted into a 100x100 micro-Doppler spectrogram, which captures time-frequency signatures useful for separating drones, birds, humans, and corner reflectors.

## Algorithm explanation

The algorithm has three stages:

1. IQ preprocessing and spectrogram generation.
2. 2D CNN classification.
3. Deployment preparation using Vitis AI for KV260 DPU.

## Why 2D CNN

The 1D CNN uses handcrafted features and is smaller, but it loses some spatial time-frequency information. The 2D CNN operates directly on spectrogram images and can learn richer micro-Doppler patterns.

## Why KV260

KV260 has much stronger DSP and UltraRAM resources than PYNQ-Z2. This makes it more suitable for CNN acceleration. The MVP uses the DPU route because it is faster and safer for a one-week deadline than writing a full custom HLS CNN accelerator.

## Demonstration commands

```bash
python scripts/00_check_dataset.py --input data/radar_iq.npz
python scripts/01_make_spectrogram_cache.py --input data/radar_iq.npz --output outputs/spectrogram_cache.npz
python scripts/02_train.py --data outputs/spectrogram_cache.npz --epochs 3 --output-dir outputs/debug_train
python scripts/07_run_cpu_demo.py --data outputs/spectrogram_cache.npz --checkpoint outputs/debug_train/best_model.pth --num-samples 10
```

## Closing

The final deliverable is an end-to-end demonstrator. If the KV260 DPU environment is ready, the compiled xmodel can be run directly on the board. If not, the CPU demo still proves the algorithmic flow, while the Vitis AI scripts show the path to hardware deployment.
