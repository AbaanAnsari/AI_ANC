"""
Phase 0 — Complete dataset pipeline verification.

Checks:
    1. Master manifest exists and is unique.
    2. All split manifests exist.
    3. Every split recording exists in the master manifest.
    4. Every master recording belongs to exactly one split.
    5. No recording-level leakage exists.
    6. Source files exist.
    7. Source WAV files remain readable.
    8. Canonical 16 kHz / mono / float32 contract is present.
    9. Expected dataset categories are represented.
   10. Optional frozen-count check against the current working dataset.

Usage:
    python scripts/verify_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

# The current frozen dataset established by the project.
EXPECTED_TOTAL_WAV = 3820


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(raw_root: Path = DEFAULT_ROOT, manifest_dir: Path = DEFAULT_MANIFEST_DIR):
    raw_root = Path(raw_root).resolve()
    manifest_dir = Path(manifest_dir).resolve()

    errors = []
    warnings = []

    print("=" * 64)
    print("PHASE 0 — COMPLETE DATASET VERIFICATION")
    print("=" * 64)
    print(f"Raw dataset: {raw_root}")

    if not raw_root.exists():
        print("FAIL: raw dataset directory does not exist.")
        return 1

    try:
        master = load_jsonl(manifest_dir / "dataset_manifest.jsonl")
        train = load_jsonl(manifest_dir / "train_manifest.jsonl")
        validation = load_jsonl(manifest_dir / "val_manifest.jsonl")
        test = load_jsonl(manifest_dir / "test_manifest.jsonl")
    except Exception as exc:
        print(f"FAIL: could not load manifests: {exc}")
        return 1

    print(f"\nMaster      : {len(master)}")
    print(f"Train       : {len(train)}")
    print(f"Validation  : {len(validation)}")
    print(f"Test        : {len(test)}")

    # ------------------------------------------------------------------
    # 1. Master uniqueness
    # ------------------------------------------------------------------
    master_paths = [r["source_path"] for r in master]
    master_ids = [r["record_id"] for r in master]

    if len(master_paths) != len(set(master_paths)):
        errors.append("Duplicate source_path in master manifest.")

    if len(master_ids) != len(set(master_ids)):
        errors.append("Duplicate record_id in master manifest.")

    # ------------------------------------------------------------------
    # 2. Split uniqueness
    # ------------------------------------------------------------------
    split_data = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    split_sets = {}

    for split_name, rows in split_data.items():
        paths = [r["source_path"] for r in rows]

        if len(paths) != len(set(paths)):
            errors.append(f"Duplicate source_path inside {split_name} manifest.")

        split_sets[split_name] = set(paths)

    # ------------------------------------------------------------------
    # 3. Leakage
    # ------------------------------------------------------------------
    overlaps = [
        (
            "train",
            "validation",
            split_sets["train"] & split_sets["validation"],
        ),
        (
            "train",
            "test",
            split_sets["train"] & split_sets["test"],
        ),
        (
            "validation",
            "test",
            split_sets["validation"] & split_sets["test"],
        ),
    ]

    for a, b, overlap in overlaps:
        if overlap:
            errors.append(
                f"DATA LEAKAGE: {len(overlap)} recording(s) "
                f"appear in both {a} and {b}."
            )

    # ------------------------------------------------------------------
    # 4. Complete master coverage
    # ------------------------------------------------------------------
    master_set = set(master_paths)

    union = split_sets["train"] | split_sets["validation"] | split_sets["test"]

    missing_from_splits = master_set - union
    unknown_in_splits = union - master_set

    if missing_from_splits:
        errors.append(
            f"{len(missing_from_splits)} master recordings "
            "are absent from all splits."
        )

    if unknown_in_splits:
        errors.append(
            f"{len(unknown_in_splits)} split recordings "
            "do not exist in the master manifest."
        )

    # ------------------------------------------------------------------
    # 5. Source file existence
    # ------------------------------------------------------------------
    missing_files = []

    for record in master:
        source = raw_root / Path(record["source_path"])

        if not source.is_file():
            missing_files.append(record["source_path"])

    if missing_files:
        errors.append(f"{len(missing_files)} manifest source file(s) are missing.")

    # ------------------------------------------------------------------
    # 6. Canonical internal processing contract
    # ------------------------------------------------------------------
    for record in master:
        if record.get("internal_sample_rate_hz") != 16000:
            errors.append(
                f"Incorrect internal sample rate: " f"{record['source_path']}"
            )

        if record.get("internal_channels") != 1:
            errors.append(
                f"Incorrect internal channel count: " f"{record['source_path']}"
            )

        if record.get("internal_dtype") != "float32":
            errors.append(f"Incorrect internal dtype: " f"{record['source_path']}")

    # ------------------------------------------------------------------
    # 7. Current frozen count
    # ------------------------------------------------------------------
    if len(master) != EXPECTED_TOTAL_WAV:
        warnings.append(
            f"Master manifest contains {len(master)} recordings; "
            f"the current frozen dataset target is {EXPECTED_TOTAL_WAV}."
        )

    # ------------------------------------------------------------------
    # 8. Readability
    # ------------------------------------------------------------------
    unreadable = []

    for record in master:
        source = raw_root / Path(record["source_path"])

        if not source.is_file():
            continue

        try:
            with wave.open(str(source), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()

                if frames <= 0 or rate <= 0:
                    unreadable.append(record["source_path"])
        except Exception:
            unreadable.append(record["source_path"])

    if unreadable:
        errors.append(
            f"{len(unreadable)} WAV file(s) failed basic WAV integrity checking."
        )

    # ------------------------------------------------------------------
    # 9. Category coverage
    # ------------------------------------------------------------------
    categories = Counter()

    for record in master:
        if record["source_group"] == "clean":
            categories["clean"] += 1
        else:
            categories[record["noise_class"]] += 1

    required = {
        "clean",
        "stationary",
        "non-stationary",
        "impulsive",
    }

    absent = required - set(categories)

    if absent:
        errors.append(
            "Required dataset categories absent: " + ", ".join(sorted(absent))
        )

    print("\nCategory counts:")
    for category in (
        "clean",
        "stationary",
        "non-stationary",
        "impulsive",
    ):
        print(f"  {category:16s}: {categories.get(category, 0):5d}")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print("\n" + "-" * 64)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  FAIL: {error}")

        print("\nPHASE 0 VERIFICATION: FAIL")
        return 1

    print("PASS: master manifest is unique.")
    print("PASS: all recordings are covered by exactly one split.")
    print("PASS: no recording-level leakage detected.")
    print("PASS: all manifest source files exist.")
    print("PASS: source WAV integrity check passed.")
    print("PASS: canonical 16 kHz / mono / float32 contract is present.")
    print("PASS: required dataset categories are represented.")

    print("\nPHASE 0 VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    args = parser.parse_args()
    sys.exit(main(args.dataset_root, args.manifest_dir))
