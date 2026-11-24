from pathlib import Path
import json
import importlib.util

import numpy as np


def _preprocessing_module():
    path = Path(__file__).resolve().parents[1] / "2_preprocessing" / "make_recordwise_windows.py"
    spec = importlib.util.spec_from_file_location("recordwise_preprocessing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grouped_stratified_split_is_disjoint_and_reproducible():
    module = _preprocessing_module()
    rows = np.arange(80)
    labels = np.repeat(np.arange(4), 20)
    first = module.make_stratified_recording_split(rows, labels, 0.70, 0.15, 0.15, 42)
    second = module.make_stratified_recording_split(rows, labels, 0.70, 0.15, 0.15, 42)

    assert first == second
    assert set(first) == set(rows.tolist())
    split_rows = {
        split: {row for row, assigned in first.items() if assigned == split}
        for split in ("train", "val", "test")
    }
    assert split_rows["train"].isdisjoint(split_rows["val"])
    assert split_rows["train"].isdisjoint(split_rows["test"])
    assert split_rows["val"].isdisjoint(split_rows["test"])
    for selected_rows in split_rows.values():
        assert set(labels[list(selected_rows)]) == {0, 1, 2, 3}


def test_generated_recordwise_caches_have_disjoint_rows():
    output_dir = Path("outputs/recordwise")
    paths = {
        split: output_dir / split / "spectrogram_cache.npz"
        for split in ("train", "val", "test")
    }
    if not all(path.exists() for path in paths.values()):
        return

    manifest_path = output_dir / "split_manifest.json"
    strategy = None
    if manifest_path.exists():
        strategy = json.loads(manifest_path.read_text())["strategy"]

    row_sets = {}
    for split, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            rows = data["row_indices"]
            row_sets[split] = set(rows.tolist())
            assert len(data["images"]) == len(data["y"]) == len(rows)
            assert set(data["y"].tolist()) == {0, 1, 2, 3}
            if strategy == "row_index_modulo_3":
                expected_remainder = {"train": 0, "val": 1, "test": 2}[split]
                assert np.all(rows % 3 == expected_remainder)

    assert row_sets["train"].isdisjoint(row_sets["val"])
    assert row_sets["train"].isdisjoint(row_sets["test"])
    assert row_sets["val"].isdisjoint(row_sets["test"])
