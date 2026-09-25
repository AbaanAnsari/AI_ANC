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
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_manifest.jsonl"
)
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


def test_supervised_training_sample_contract():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        rng=np.random.default_rng(42),
    )

    sample = dataset.generate_sample(rng=np.random.default_rng(42))

    # 1. Exactly 16,000 samples for M1, M2, clean target
    m1 = sample["m1_waveform"]
    m2 = sample["m2_waveform"]
    clean = sample["clean_target"]

    assert m1.shape == (16000,)
    assert m2.shape == (16000,)
    assert clean.shape == (16000,)

    # 2. All waveform arrays: 1-D, float32, finite
    for arr in (m1, m2, clean):
        assert arr.ndim == 1
        if HAS_TORCH and isinstance(arr, torch.Tensor):
            assert arr.dtype == torch.float32
            assert torch.all(torch.isfinite(arr))
        else:
            assert arr.dtype == np.float32
            assert np.all(np.isfinite(arr))

    # 3. M1 contains speech + noise, clean target != M1
    assert not np.array_equal(clean, m1)

    # 4. M2 is correlated with noise and not identical to M1
    assert not np.array_equal(m1, m2)

    # 5. Measured SNR close to target SNR
    target_snr = sample["target_snr_db"]
    measured_snr = sample["measured_snr_db"]
    assert np.isclose(target_snr, measured_snr, atol=1e-2)

    # 6. Noise class label is valid (0, 1, 2)
    label = sample["noise_class_label"]
    assert label in (0, 1, 2)
    assert NOISE_CLASS_TO_LABEL[sample["noise_class"]] == label

    # 7 & 8 & 9 & 10. STFT shape & complex dtype
    noisy_stft = sample["noisy_stft"]
    target_stft = sample["target_stft"]

    assert noisy_stft.ndim == 2
    assert noisy_stft.shape[0] == 257  # frequency_bins

    if HAS_TORCH and isinstance(noisy_stft, torch.Tensor):
        assert noisy_stft.dtype == torch.complex64
        assert target_stft.dtype == torch.complex64
    else:
        assert noisy_stft.dtype == np.complex64
        assert target_stft.dtype == np.complex64

    assert target_stft.shape == noisy_stft.shape

    # 11 & 12. Delays finite
    assert np.isfinite(sample["gcc_delay_samples"])
    assert np.isfinite(sample["kalman_delay_samples"])


def test_seed_reproducibility():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
    )

    sample_a = dataset.generate_sample(rng=np.random.default_rng(12345))
    sample_b = dataset.generate_sample(rng=np.random.default_rng(12345))

    assert np.array_equal(sample_a["m1_waveform"], sample_b["m1_waveform"])
    assert np.array_equal(sample_a["m2_waveform"], sample_b["m2_waveform"])
    assert np.array_equal(sample_a["clean_target"], sample_b["clean_target"])
    assert sample_a["noise_class"] == sample_b["noise_class"]
    assert sample_a["target_snr_db"] == sample_b["target_snr_db"]


def test_seed_variability():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
    )

    sample_a = dataset.generate_sample(rng=np.random.default_rng(111))
    sample_b = dataset.generate_sample(rng=np.random.default_rng(999))

    # Different seeds generate different combinations or random crops
    assert not (
        np.array_equal(sample_a["m1_waveform"], sample_b["m1_waveform"])
        and np.array_equal(sample_a["clean_target"], sample_b["clean_target"])
    )


def test_all_three_noise_classes():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
    )

    for noise_cls in ("stationary", "non-stationary", "impulsive"):
        sample = dataset.generate_sample(
            noise_class=noise_cls,
            rng=np.random.default_rng(42),
        )

        assert sample["noise_class"] == noise_cls
        assert sample["noise_class_label"] == NOISE_CLASS_TO_LABEL[noise_cls]
        assert sample["m1_waveform"].shape == (16000,)
        assert sample["noisy_stft"].shape[0] == 257


def test_multiple_snr_values():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
    )

    for snr in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0):
        sample = dataset.generate_sample(
            target_snr_db=snr,
            rng=np.random.default_rng(42),
        )

        assert np.isclose(sample["target_snr_db"], snr)
        assert np.isclose(sample["measured_snr_db"], snr, atol=1e-2)


def test_pytorch_style_indexing():
    dataset = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
    )

    assert len(dataset) > 0

    sample = dataset[0]
    assert "m1_waveform" in sample
    assert "noisy_stft" in sample
    assert sample["m1_waveform"].shape == (16000,)


if __name__ == "__main__":
    test_supervised_training_sample_contract()
    test_seed_reproducibility()
    test_seed_variability()
    test_all_three_noise_classes()
    test_multiple_snr_values()
    test_pytorch_style_indexing()
    print("dataset tests: PASS")
