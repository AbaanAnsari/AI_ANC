from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.audio_loader import load_audio


PROJECT_ROOT = Path(__file__).resolve().parent

MANIFESTS = {
    "train": PROJECT_ROOT / "data/manifests/train_manifest.jsonl",
    "validation": PROJECT_ROOT / "data/manifests/val_manifest.jsonl",
    "test": PROJECT_ROOT / "data/manifests/test_manifest.jsonl",
}

EXPECTED_GROUPS = {
    "clean",
    "stationary",
    "non-stationary",
    "impulsive",
}

SAMPLES_PER_GROUP_PER_SPLIT = 2


def load_manifest(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def get_group(record):
    """
    Convert the manifest schema into the four dataset groups:

        clean
        stationary
        non-stationary
        impulsive
    """

    source_group = record.get("source_group")

    if source_group == "clean":
        return "clean"

    if source_group == "noise":
        return record.get("noise_class")

    return None


def main():

    print("=" * 70)
    print("DATASET-LEVEL AUDIO LOADER VERIFICATION")
    print("=" * 70)

    total_checked = 0
    failures = []

    for split_name, manifest_path in MANIFESTS.items():

        print(f"\n[{split_name.upper()}]")
        print(f"Manifest: {manifest_path}")

        if not manifest_path.exists():

            failures.append(
                f"{split_name}: manifest not found: {manifest_path}"
            )

            print("FAIL - manifest not found")
            continue

        records = load_manifest(manifest_path)

        print(f"Records: {len(records)}")

        for group in sorted(EXPECTED_GROUPS):

            group_records = [
                record
                for record in records
                if get_group(record) == group
            ]

            if not group_records:

                failures.append(
                    f"{split_name}/{group}: no records found"
                )

                print(
                    f"  {group:15s}: "
                    "FAIL - no records"
                )

                continue

            selected = group_records[
                :SAMPLES_PER_GROUP_PER_SPLIT
            ]

            group_ok = True

            for record in selected:

                relative_path = record["source_path"]

                source_path = (
                    PROJECT_ROOT
                    / "data"
                    / "raw"
                    / "dataset"
                    / relative_path
                )

                if not source_path.exists():

                    failures.append(
                        f"{split_name}/{group}: "
                        f"missing file: {source_path}"
                    )

                    group_ok = False
                    continue

                try:

                    audio = load_audio(source_path)

                    if not isinstance(audio, np.ndarray):

                        raise TypeError(
                            f"output is {type(audio)}, "
                            "expected numpy.ndarray"
                        )

                    if audio.dtype != np.float32:

                        raise TypeError(
                            f"dtype is {audio.dtype}, "
                            "expected float32"
                        )

                    if audio.ndim != 1:

                        raise ValueError(
                            f"shape is {audio.shape}, "
                            "expected 1-D mono"
                        )

                    if audio.size == 0:

                        raise ValueError(
                            "audio contains zero samples"
                        )

                    if not np.all(np.isfinite(audio)):

                        raise ValueError(
                            "audio contains NaN or Inf"
                        )

                except Exception as exc:

                    failures.append(
                        f"{split_name}/{group}: "
                        f"{source_path}: {exc}"
                    )

                    group_ok = False

                total_checked += 1

            status = "PASS" if group_ok else "FAIL"

            print(
                f"  {group:15s}: "
                f"{status} "
                f"({len(selected)} file(s) checked)"
            )

    print("\n" + "=" * 70)

    if failures:

        print("FINAL RESULT: FAIL")
        print("\nFailures:")

        for failure in failures:
            print(f" - {failure}")

    else:

        print("FINAL RESULT: PASS")
        print(f"Files checked: {total_checked}")
        print(
            "All checked files satisfy "
            "the canonical audio contract."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()