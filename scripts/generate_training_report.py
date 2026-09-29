"""
scripts/generate_training_report.py
====================================
Generates comprehensive analysis of Phase 3 v2 training run:
- Epoch loss curves (total, enhancement, classification, SI-SDR)
- Train/Validation gap and overfitting analysis
- Fingerprint log verification across epochs
- Learning rate trajectory
- Best epoch selection summary
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments" / "phase3_v2"


def generate_report():
    hist_file = EXP_DIR / "training_history.json"
    fp_file = EXP_DIR / "fingerprint_log.json"

    if not hist_file.exists():
        print(f"File not found: {hist_file}")
        return

    with open(hist_file) as f:
        history = json.load(f)

    fingerprints = {}
    if fp_file.exists():
        with open(fp_file) as f:
            fingerprints = json.load(f)

    print("\n" + "=" * 80)
    print("PHASE 3 V2 TRAINING CURVES & DIVERSITY REPORT")
    print("=" * 80)
    print(f"{'Epoch':<6} {'TrLoss':<10} {'TrEnh':<10} {'ValLoss':<10} {'ValEnh':<10} {'ValAcc':<8} {'LR':<10} {'Hash0':<14} {'Best'}")
    print("-" * 80)

    best_loss = float("inf")
    best_epoch = 0

    for rec in history:
        ep = rec["epoch"]
        tr_loss = rec["train_total_loss"]
        tr_enh = rec["train_enhancement_loss"]
        val_loss = rec["val_total_loss"]
        val_enh = rec["val_enhancement_loss"]
        val_acc = rec["val_accuracy"] * 100.0
        lr = rec.get("learning_rate", rec.get("lr", 0.0))
        h0 = fingerprints.get(str(ep), {}).get("sample_0_hash", "N/A")
        is_b = "*" if val_loss < best_loss else ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = ep

        print(f"{ep:<6} {tr_loss:<10.4f} {tr_enh:<10.4f} {val_loss:<10.4f} {val_enh:<10.4f} {val_acc:<7.1f}% {lr:<10.2e} {h0:<14} {is_b}")

    print("-" * 80)
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_loss:.6f}")

    # Unique hashes count
    unique_hashes = set(v["sample_0_hash"] for v in fingerprints.values())
    print(f"Total Epochs: {len(history)}")
    print(f"Total Unique Sample 0 Hashes: {len(unique_hashes)} / {len(fingerprints)} ({100.0 * len(unique_hashes) / max(1, len(fingerprints)):.1f}% distinct)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generate_report()
