import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.noise_selector import (
    NOISE_CLASSES,
    NoiseSelector,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_manifest.jsonl"
)


def test_noise_class_selection():

    selector = NoiseSelector(
        MANIFEST_PATH,
        rng=np.random.default_rng(42),
    )

    assert selector.available_classes() == (
        "stationary",
        "non-stationary",
        "impulsive",
    )

    for noise_class in NOISE_CLASSES:

        assert selector.count(noise_class) > 0

        record = selector.select(noise_class)

        assert record["source_group"] == "noise"
        assert record["noise_class"] == noise_class
        assert "source_path" in record


def test_selection_is_reproducible():

    selector_a = NoiseSelector(
        MANIFEST_PATH,
        rng=np.random.default_rng(123),
    )

    selector_b = NoiseSelector(
        MANIFEST_PATH,
        rng=np.random.default_rng(123),
    )

    for noise_class in NOISE_CLASSES:

        record_a = selector_a.select(noise_class)
        record_b = selector_b.select(noise_class)

        assert (
            record_a["record_id"]
            == record_b["record_id"]
        )


def test_select_many():

    selector = NoiseSelector(
        MANIFEST_PATH,
        rng=np.random.default_rng(42),
    )

    records = selector.select_many(
        "stationary",
        5,
    )

    assert len(records) == 5

    for record in records:

        assert record["source_group"] == "noise"
        assert record["noise_class"] == "stationary"


def test_invalid_class():

    selector = NoiseSelector(
        MANIFEST_PATH,
        rng=np.random.default_rng(42),
    )

    try:
        selector.select("invalid-class")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid noise class should raise ValueError"
        )


def test_split_manifests():

    for split in ("train", "val", "test"):
        split_path = (
            PROJECT_ROOT
            / "data"
            / "manifests"
            / f"{split}_manifest.jsonl"
        )
        selector = NoiseSelector(
            split_path,
            rng=np.random.default_rng(42),
        )

        for noise_class in NOISE_CLASSES:
            assert selector.count(noise_class) > 0
            record = selector.select(noise_class)
            assert record["source_group"] == "noise"
            assert record["noise_class"] == noise_class


if __name__ == "__main__":

    test_noise_class_selection()
    test_selection_is_reproducible()
    test_select_many()
    test_invalid_class()
    test_split_manifests()

    print("noise selector tests: PASS")