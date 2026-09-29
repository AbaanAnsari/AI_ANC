"""
tests/test_phase3_dataset_diversity.py
======================================
Regression tests for Phase 3 epoch-aware deterministic dataset generation.
Verifies:
1. same epoch + same index -> identical sample
2. different epoch + same index -> different stochastic mixture
3. same seed + same epoch + same index -> reproducible
4. different workers -> independent seeds (no collision)
5. validation samples remain deterministic across calls
6. phase3_v2 train / val / test speaker and file independence (zero leakage)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.data.dataset import Phase1Dataset

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "train_manifest.jsonl"
VAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "val_manifest.jsonl"
PHASE3_V2_DIR = PROJECT_ROOT / "data" / "manifests" / "phase3_v2"


def test_same_epoch_same_index_identical():
    """Test 1: same epoch + same index -> 100% identical sample."""
    ds = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=123, epoch=5)
    s1 = ds[10]
    s2 = ds[10]

    assert torch.equal(
        s1["m1_waveform"], s2["m1_waveform"]
    ), "Same epoch+index must produce bit-identical m1"
    assert torch.equal(
        s1["m2_waveform"], s2["m2_waveform"]
    ), "Same epoch+index must produce bit-identical m2"
    assert torch.equal(
        s1["clean_target"], s2["clean_target"]
    ), "Same epoch+index must produce bit-identical clean target"
    assert s1["target_snr_db"] == s2["target_snr_db"]
    assert s1["configured_delay_samples"] == s2["configured_delay_samples"]


def test_different_epoch_same_index_different():
    """Test 2: different epoch + same index -> different stochastic sample."""
    ds = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=123, epoch=0)
    s_ep0 = ds[10]

    ds.set_epoch(1)
    s_ep1 = ds[10]

    ds.set_epoch(2)
    s_ep2 = ds[10]

    # Mixture waveforms must differ across epochs
    m1_diff_01 = float(
        torch.max(torch.abs(s_ep0["m1_waveform"] - s_ep1["m1_waveform"]))
    )
    m1_diff_12 = float(
        torch.max(torch.abs(s_ep1["m1_waveform"] - s_ep2["m1_waveform"]))
    )

    assert (
        m1_diff_01 > 1e-4
    ), f"Epoch 0 and 1 must produce different mixtures, got diff={m1_diff_01}"
    assert (
        m1_diff_12 > 1e-4
    ), f"Epoch 1 and 2 must produce different mixtures, got diff={m1_diff_12}"
    assert s_ep0["sample_seed"] != s_ep1["sample_seed"]
    assert s_ep1["sample_seed"] != s_ep2["sample_seed"]


def test_same_seed_epoch_index_reproducible():
    """Test 3: same seed + same epoch + same index -> reproducible across dataset instances."""
    ds_a = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=999, epoch=42)
    ds_b = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=999, epoch=42)

    sample_a = ds_a[25]
    sample_b = ds_b[25]

    assert torch.equal(sample_a["m1_waveform"], sample_b["m1_waveform"])
    assert torch.equal(sample_a["noisy_features"], sample_b["noisy_features"])
    assert sample_a["sample_seed"] == sample_b["sample_seed"]


def test_worker_ids_do_not_change_sample_seed():
    """A sample's identity is independent of which DataLoader worker requests it."""
    seed_w0 = Phase1Dataset.compute_sample_seed(
        global_seed=42, epoch=1, worker_id=0, index=100
    )
    seed_w1 = Phase1Dataset.compute_sample_seed(
        global_seed=42, epoch=1, worker_id=1, index=100
    )
    seed_w2 = Phase1Dataset.compute_sample_seed(
        global_seed=42, epoch=1, worker_id=2, index=100
    )

    assert seed_w0 == seed_w1 == seed_w2


def test_worker_assignments_return_identical_sample(monkeypatch):
    dataset = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=123, epoch=5)
    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=0)
    )
    worker_zero_sample = dataset[10]
    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=1)
    )
    worker_one_sample = dataset[10]

    assert worker_zero_sample["sample_seed"] == worker_one_sample["sample_seed"]
    assert torch.equal(
        worker_zero_sample["m1_waveform"], worker_one_sample["m1_waveform"]
    )
    assert torch.equal(
        worker_zero_sample["m2_waveform"], worker_one_sample["m2_waveform"]
    )


def test_validation_samples_remain_deterministic():
    """Test 5: validation dataset remains deterministic regardless of set_epoch."""
    val_ds = Phase1Dataset(manifest_path=VAL_MANIFEST, seed=123, is_validation=True)

    val_ds.set_epoch(0)
    v0 = val_ds[5]

    # Even if set_epoch is called with 10, is_validation=True locks epoch to 0
    val_ds.set_epoch(10)
    v10 = val_ds[5]

    assert torch.equal(
        v0["m1_waveform"], v10["m1_waveform"]
    ), "Validation samples must not change with set_epoch"
    assert v0["epoch"] == 0 and v10["epoch"] == 0


def test_phase3_v2_zero_leakage():
    """Test 6: phase3_v2 manifests guarantee 0 speaker and 0 file overlap."""
    if not (PHASE3_V2_DIR / "train_manifest.jsonl").exists():
        pytest.skip("phase3_v2 manifests not yet generated")

    train_records = [
        json.loads(l)
        for l in (PHASE3_V2_DIR / "train_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    val_records = [
        json.loads(l)
        for l in (PHASE3_V2_DIR / "val_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    test_records = [
        json.loads(l)
        for l in (PHASE3_V2_DIR / "test_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]

    # 1. File-level overlap
    train_files = {r["source_path"] for r in train_records}
    val_files = {r["source_path"] for r in val_records}
    test_files = {r["source_path"] for r in test_records}

    assert (
        len(train_files.intersection(val_files)) == 0
    ), "Train and Val files must not overlap"
    assert (
        len(train_files.intersection(test_files)) == 0
    ), "Train and Test files must not overlap"
    assert (
        len(val_files.intersection(test_files)) == 0
    ), "Val and Test files must not overlap"

    # 2. Clean speaker overlap
    train_speakers = {
        r["speaker_id"] for r in train_records if r["source_group"] == "clean"
    }
    val_speakers = {
        r["speaker_id"] for r in val_records if r["source_group"] == "clean"
    }
    test_speakers = {
        r["speaker_id"] for r in test_records if r["source_group"] == "clean"
    }

    assert (
        len(train_speakers.intersection(val_speakers)) == 0
    ), f"Speaker overlap Train/Val: {train_speakers.intersection(val_speakers)}"
    assert (
        len(train_speakers.intersection(test_speakers)) == 0
    ), f"Speaker overlap Train/Test: {train_speakers.intersection(test_speakers)}"
    assert (
        len(val_speakers.intersection(test_speakers)) == 0
    ), f"Speaker overlap Val/Test: {val_speakers.intersection(test_speakers)}"


def test_strict_checkpoint_loading():
    """Test 7: Checkpoints load into LightweightCNNGRUMaskModel with strict=True (0 missing, 0 unexpected)."""
    from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel

    for ckpt_p in [
        PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt",
        PROJECT_ROOT / "experiments" / "phase3_v2" / "best_checkpoint.pt",
    ]:
        if not ckpt_p.exists():
            continue
        ckpt = torch.load(ckpt_p, map_location="cpu")
        model = LightweightCNNGRUMaskModel()
        incompatible_keys = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        assert (
            len(incompatible_keys.missing_keys) == 0
        ), f"Missing keys in {ckpt_p}: {incompatible_keys.missing_keys}"
        assert (
            len(incompatible_keys.unexpected_keys) == 0
        ), f"Unexpected keys in {ckpt_p}: {incompatible_keys.unexpected_keys}"
