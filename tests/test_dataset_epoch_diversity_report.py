"""
tests/test_dataset_epoch_diversity_report.py
============================================
Phase 3 Dataset Diversity Audit & Fingerprinting.
Samples the same sample indices across 10 distinct epochs and measures:
- Unique mixture waveforms
- Unique noise sources
- Unique SNR values
- Unique delay values
- SHA-256 mixture fingerprints
"""
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.data.dataset import Phase1Dataset

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "phase3_v2" / "train_manifest.jsonl"


def fingerprint_waveform(wave: torch.Tensor) -> str:
    arr = wave.cpu().numpy().astype(np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def run_diversity_audit():
    print("=" * 70)
    print("PHASE 3 EPOCH-AWARE DATASET DIVERSITY & FINGERPRINT AUDIT")
    print("=" * 70)

    ds = Phase1Dataset(manifest_path=TRAIN_MANIFEST, seed=123)
    test_indices = [0, 10, 42, 100]
    n_epochs = 10

    for idx in test_indices:
        print(f"\n--- Auditing Sample Index {idx:3d} across Epochs 0 to {n_epochs - 1} ---")
        seen_hashes = set()
        seen_noise_sources = set()
        seen_snrs = set()
        seen_delays = set()

        for ep in range(n_epochs):
            ds.set_epoch(ep)
            sample = ds[idx]

            fp = fingerprint_waveform(sample["m1_waveform"])
            seen_hashes.add(fp)

            noise_src = sample["metadata"]["noise_source_path"]
            seen_noise_sources.add(noise_src)

            snr = float(sample["target_snr_db"])
            seen_snrs.add(snr)

            dly = int(sample["configured_delay_samples"])
            seen_delays.add(dly)

            if ep < 5 or ep == n_epochs - 1:
                print(f"  Epoch {ep:2d} | Hash={fp} | Noise={Path(noise_src).name[:25]:25s} | SNR={snr:+5.1f} dB | Delay={dly:+3d}")

        print(f"Summary for Index {idx}:")
        print(f"  Unique Mixture Hashes: {len(seen_hashes)} / {n_epochs} ({len(seen_hashes)/n_epochs*100:.0f}%)")
        print(f"  Unique Noise Sources:  {len(seen_noise_sources)} / {n_epochs}")
        print(f"  Unique Target SNRs:    {len(seen_snrs)} / {n_epochs}")
        print(f"  Unique Delays:         {len(seen_delays)} / {n_epochs}")

        assert len(seen_hashes) == n_epochs, f"Expected {n_epochs} unique hashes, got {len(seen_hashes)}"

    print("\n" + "=" * 70)
    print("ALL DIVERSITY CHECKS PASSED: 100% Genuine Mixture Variety Established Across Epochs!")
    print("=" * 70)


if __name__ == "__main__":
    run_diversity_audit()
