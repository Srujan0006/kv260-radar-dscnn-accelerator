# Legacy wrapper scripts

Historical one-shot shell wrappers kept for provenance — they document exactly
how each experiment run was launched during development. Nothing here is
needed anymore: every command they wrap is listed directly in the root
[README](../../README.md) command reference, and the maintained entry points
are the Python scripts in `preprocessing/`, `training/`, `evaluation/`,
`quantization/`, and `scripts/`.

`10_convert_real_npy_to_windows.py` is the pre-recordwise dataset converter,
superseded by `preprocessing/01_make_recordwise_windows.py`.
