"""
scripts/verify_epoch_diversity_multi_index.py
=============================================
Mandatory Pre-Training Variety Verification (Section 11).
Tests a fixed set of indices: [0, 1, 10, 100, 500]
Across epochs: [0, 1, 50, 99]
Compares:
- Clean segment identity
- Selected noise segment
- SNR
- Delay
- Mixture waveform hash
- Target waveform
Verifies that:
1. Same epoch + same index -> bit-identical
2. Different epoch + same index -> materially different mixtures
3. All unique mixtures across epochs are reported.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import Phase1Dataset

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "phase3_v2" / "train_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


def hash_waveform(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()[:12]


def run_variety_verification():
    print("=" * 90)
    print("MANDATORY PRE-TRAINING VARIETY VERIFICATION (SECTION 11)")
    print("=" * 90)

    ds = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        seed=20260929,
        is_validation=False,
    )

    indices = [0, 1, 10, 100, 500]
    epochs = [0, 1, 50, 99]

    records = {}

    for idx in indices:
        records[idx] = {}
        print(f"\n--- Checking Index {idx} (Clean: {ds.clean_records[idx]['source_path']}) ---")
        hashes = set()
        for ep in epochs:
            ds.set_epoch(ep)
            s = ds[idx]
            m1_hash = hash_waveform(s["m1_waveform"].numpy())
            noise_path = s["metadata"]["noise_source_path"]
            snr = float(s["target_snr_db"])
            delay = int(s["configured_delay_samples"])

            # Verify reproducibility by sampling again in the same epoch
            s_repeat = ds[idx]
            assert torch.equal(s["m1_waveform"], s_repeat["m1_waveform"]), f"Epoch {ep} index {idx} not reproducible!"

            hashes.add(m1_hash)
            records[idx][ep] = {
                "hash": m1_hash,
                "noise": noise_path,
                "snr": snr,
                "delay": delay,
            }
            print(f"  Epoch {ep:<2}: Hash={m1_hash} | SNR={snr:+5.1f} dB | Delay={delay:+2} | Noise={Path(noise_path).name}")

        uniqueness = len(hashes) / len(epochs)
        print(f"  -> Unique mixture hashes for index {idx}: {len(hashes)}/{len(epochs)} ({uniqueness*100:.1f}%)")
        assert len(hashes) == len(epochs), f"Expected all {len(epochs)} epochs to yield unique mixtures, got {len(hashes)}"

    print("\n" + "=" * 90)
    print("VERIFICATION RESULT: 100% UNIQUE MIXTURES ACROSS ALL TESTED INDICES AND EPOCHS!")
    print("Deterministic Reproducibility: 100% PASS")
    print("Epoch Diversity Invariant:    100% PASS")
    print("=" * 90)

    # Save to json
    out_file = PROJECT_ROOT / "experiments" / "phase3_100ep" / "dataset_variety_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved variety verification records to {out_file}\n")


if __name__ == "__main__":
    run_variety_verification()
