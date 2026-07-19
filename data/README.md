# Data folder

The raw dataset is intentionally kept OUTSIDE the repository to keep it
compact (the raw file is ~1.45 GiB):

```text
../radar_data/radar_iq.npy        (sibling folder of the repo root)
```

i.e. on this machine:

```text
Downloads/KV260_Radar_2D_CNN_MVP/radar_data/radar_iq.npy   <- raw IQ recordings
Downloads/KV260_Radar_2D_CNN_MVP/KV260_Radar_2D_CNN_MVP/   <- this repo
```

All preprocessing entry points accept `--input`; their defaults already
point at `../radar_data/radar_iq.npy` (resolved from the repo root, which
is the working directory the `run_*.sh` wrappers use).

## What stays in the repo

`data/real/` keeps only the small derived window caches (regenerable from
the raw file):

```text
data/real/radar_iq_windows.npz                  (+ .label_map.json)
data/real/radar_iq_windows_4class.npz
data/real/radar_iq_windows_stride32.npz         (+ .label_map.json)
data/real/radar_iq_windows_stride32_4class.npz
```

Regenerate them with:

```bash
python preprocessing/01_make_recordwise_windows.py --input ../radar_data/radar_iq.npy
# (legacy converter: scripts/legacy/10_convert_real_npy_to_windows.py)
```

None of the `.npy`/`.npz` files are tracked by git (see `.gitignore`).

## Labels

Four superclasses:

```text
0 Drone
1 Bird
2 Human
3 CornerReflector
```

If labels are strings, the code maps them to these superclasses; integer
labels are used directly unless a label mapping is provided.
