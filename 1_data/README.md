# Data

The raw dataset is **not** included in this repository.

## Expected input

`radar_iq.npy` is a NumPy object array of shape `130 × 6`, with one row per
recording. Column 0 is an identifier whose letter prefix gives the class, and
column 1 is the complex IQ matrix (5 range bins × N slow-time samples after
reshaping). The other columns are metadata and are not used.

| Prefix | Rows | Class |
|---|---:|---|
| D / d | 44 | Drone |
| B / b | 11 | Bird |
| H / h | 16 | Human |
| C | 19 | Corner Reflector |
| S, P, R | 40 | excluded |

59 recordings are long enough for at least one 256-sample window.

All preprocessing entry points take `--input`. The default is
`../radar_data/radar_iq.npy`, relative to the repository root.

## Generated files (ignored by Git)

```bash
python 2_preprocessing/make_recordwise_windows.py --input ../radar_data/radar_iq.npy
```

Window archives (`*.npz`) and spectrogram caches are regenerated locally.
`real/*.label_map.json` records the class mapping used.

## Labels

```text
0 Drone   1 Bird   2 Human   3 CornerReflector
```
