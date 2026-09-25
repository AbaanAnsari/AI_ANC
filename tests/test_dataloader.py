import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.dataset import (
    NOISE_CLASS_TO_LABEL,
    Phase1Dataset,
    create_dataloader,
)

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


def test_single_sample_retrieval():
    dataset = Phase1Dataset(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        seed=42,
    )
    sample = dataset[0]

    assert "noisy_features" in sample
    assert "target_stft" in sample
    assert "clean_target" in sample
    assert "noise_class_label" in sample

    if HAS_TORCH:
        assert isinstance(sample["noisy_features"], torch.Tensor)
        assert sample["noisy_features"].shape == (2, 257, 126)
        assert sample["noisy_features"].dtype == torch.float32

        assert isinstance(sample["target_stft"], torch.Tensor)
        assert sample["target_stft"].shape == (257, 126)
        assert sample["target_stft"].dtype == torch.complex64

        assert isinstance(sample["clean_target"], torch.Tensor)
        assert sample["clean_target"].shape == (16000,)
        assert sample["clean_target"].dtype == torch.float32

        assert isinstance(sample["noise_class_label"], torch.Tensor)
        assert sample["noise_class_label"].dtype == torch.int64


def test_batch_generation_and_shapes():
    dataloader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=4,
        shuffle=False,
        seed=100,
    )

    batch = next(iter(dataloader))

    # Check batch shapes for B=4
    noisy_features = batch["noisy_features"]
    target_stft = batch["target_stft"]
    clean_target = batch["clean_target"]
    noise_class_label = batch["noise_class_label"]

    assert noisy_features.shape == (4, 2, 257, 126)
    assert target_stft.shape == (4, 257, 126)
    assert clean_target.shape == (4, 16000)
    assert noise_class_label.shape == (4,)

    if HAS_TORCH and isinstance(noisy_features, torch.Tensor):
        assert noisy_features.dtype == torch.float32
        assert target_stft.dtype == torch.complex64
        assert clean_target.dtype == torch.float32
        assert noise_class_label.dtype == torch.int64
        assert torch.all(torch.isfinite(noisy_features))
        assert torch.all(torch.isfinite(clean_target))
    else:
        assert noisy_features.dtype == np.float32
        assert target_stft.dtype == np.complex64
        assert clean_target.dtype == np.float32
        assert noise_class_label.dtype == np.int64
        assert np.all(np.isfinite(noisy_features))
        assert np.all(np.isfinite(clean_target))


def test_split_isolation_and_no_leakage():
    train_ds = Phase1Dataset(manifest_path=TRAIN_MANIFEST, dataset_root=DATASET_ROOT)
    val_ds = Phase1Dataset(manifest_path=VAL_MANIFEST, dataset_root=DATASET_ROOT)
    test_ds = Phase1Dataset(manifest_path=TEST_MANIFEST, dataset_root=DATASET_ROOT)

    train_clean_ids = {r["record_id"] for r in train_ds.clean_records}
    val_clean_ids = {r["record_id"] for r in val_ds.clean_records}
    test_clean_ids = {r["record_id"] for r in test_ds.clean_records}

    # Strict leakage check
    assert len(train_clean_ids.intersection(val_clean_ids)) == 0
    assert len(train_clean_ids.intersection(test_clean_ids)) == 0
    assert len(val_clean_ids.intersection(test_clean_ids)) == 0


def test_deterministic_batches_and_seed_variability():
    loader_a1 = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=4,
        shuffle=True,
        seed=123,
    )
    loader_a2 = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=4,
        shuffle=True,
        seed=123,
    )
    loader_b = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=4,
        shuffle=True,
        seed=999,
    )

    batch_a1 = next(iter(loader_a1))
    batch_a2 = next(iter(loader_a2))
    batch_b = next(iter(loader_b))

    if HAS_TORCH and isinstance(batch_a1["clean_target"], torch.Tensor):
        assert torch.equal(batch_a1["clean_target"], batch_a2["clean_target"])
        assert not torch.equal(batch_a1["clean_target"], batch_b["clean_target"])
    else:
        assert np.array_equal(batch_a1["clean_target"], batch_a2["clean_target"])
        assert not np.array_equal(batch_a1["clean_target"], batch_b["clean_target"])


def test_noise_class_coverage_in_batches():
    loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=12,
        shuffle=True,
        seed=42,
    )
    batch = next(iter(loader))
    labels = batch["noise_class_label"]

    if HAS_TORCH and isinstance(labels, torch.Tensor):
        unique_labels = torch.unique(labels).tolist()
    else:
        unique_labels = np.unique(labels).tolist()

    for expected in (0, 1, 2):
        assert expected in unique_labels, f"Expected label {expected} in batch"


def test_snr_coverage():
    loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=6,
        shuffle=False,
        seed=42,
    )
    batch = next(iter(loader))
    snrs = batch["target_snr_db"]

    if HAS_TORCH and isinstance(snrs, torch.Tensor):
        snr_list = [float(x) for x in snrs]
    else:
        snr_list = [float(x) for x in snrs]

    assert len(snr_list) == 6
    for snr_val in snr_list:
        assert snr_val in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


if __name__ == "__main__":
    test_single_sample_retrieval()
    test_batch_generation_and_shapes()
    test_split_isolation_and_no_leakage()
    test_deterministic_batches_and_seed_variability()
    test_noise_class_coverage_in_batches()
    test_snr_coverage()
    print("dataloader tests: PASS")
