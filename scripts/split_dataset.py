"""
Phase 0 — Create recording-level train/validation/test manifests.

Locked split:

    Train       70%
    Validation  15%
    Test        15%

For the current frozen dataset of 3,820 recordings:

    Train       2,674
    Validation   573
    Test         573

The split is deterministic and happens BEFORE segmentation or augmentation.

Leakage rule:

    A source recording can appear in exactly one split.

The split is stratified by the four source categories:

    clean
    stationary
    non-stationary
    impulsive

Usage:

    python scripts/split_dataset.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


MANIFEST = Path("data/manifests/dataset_manifest.jsonl")
OUTPUT_DIR = Path("data/manifests")

SEED = 20260925

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Master manifest not found: {path}\n"
            "Run create_manifest.py first."
        )

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        raise SystemExit("Master manifest is empty.")

    return records


def deterministic_key(record_id: str) -> int:
    """
    Generate a deterministic ordering key from the locked seed
    and recording ID.

    This gives us reproducible shuffling without relying on
    Python's randomized hash().
    """
    value = f"{SEED}:{record_id}".encode("utf-8")
    digest = hashlib.sha256(value).hexdigest()
    return int(digest, 16)


def grouping_key(record: dict) -> str:
    """
    Return the source category used for stratification.
    """
    if record["source_group"] == "clean":
        return "clean"

    return record["noise_class"]


def calculate_targets(total: int) -> dict[str, int]:
    """
    Calculate exact integer split targets.

    For the current dataset:

        3820 -> 2674 / 573 / 573
    """

    train = round(total * TRAIN_FRACTION)
    validation = round(total * VALIDATION_FRACTION)

    # Force exact total coverage.
    test = total - train - validation

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def allocate_group(
    records: list[dict],
    train_target: int,
    validation_target: int,
    test_target: int,
) -> dict[str, list[dict]]:
    """
    Deterministically allocate one category across the three splits.

    The allocation is based on the requested global proportions while
    preserving exact global split totals through the remaining-capacity
    mechanism.
    """

    ordered = sorted(
        records,
        key=lambda record: deterministic_key(record["record_id"]),
    )

    total = len(ordered)

    if total == 0:
        return {
            "train": [],
            "validation": [],
            "test": [],
        }

    desired = {
        "train": total * TRAIN_FRACTION,
        "validation": total * VALIDATION_FRACTION,
        "test": total * TEST_FRACTION,
    }

    base = {
        "train": int(desired["train"]),
        "validation": int(desired["validation"]),
        "test": int(desired["test"]),
    }

    remainder = total - sum(base.values())

    fractions = sorted(
        (
            (desired[name] - base[name], name)
            for name in base
        ),
        reverse=True,
    )

    for _, name in fractions[:remainder]:
        base[name] += 1

    # Respect remaining global capacities.
    capacities = {
        "train": train_target,
        "validation": validation_target,
        "test": test_target,
    }

    allocation = {
        "train": [],
        "validation": [],
        "test": [],
    }

    remaining = list(ordered)

    for split in ("train", "validation", "test"):
        count = min(base[split], capacities[split], len(remaining))

        allocation[split].extend(remaining[:count])
        remaining = remaining[count:]
        capacities[split] -= count

    # Any records left are assigned to splits with remaining capacity.
    for record in remaining:
        available = [
            split
            for split in ("train", "validation", "test")
            if capacities[split] > 0
        ]

        if not available:
            raise RuntimeError(
                "Unable to allocate all recordings while respecting "
                "the requested split capacities."
            )

        # Choose the split with the largest remaining capacity.
        split = max(
            available,
            key=lambda name: capacities[name],
        )

        allocation[split].append(record)
        capacities[split] -= 1

    return allocation


def main() -> None:

    records = load_manifest(MANIFEST)

    total = len(records)

    targets = calculate_targets(total)

    if total != 3820:
        print(
            f"WARNING: Expected frozen dataset size is 3820 recordings, "
            f"but master manifest contains {total}."
        )

    fraction_sum = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + TEST_FRACTION
    )

    if abs(fraction_sum - 1.0) > 1e-9:
        raise SystemExit(
            "Split fractions do not sum to 1.0."
        )

    # ---------------------------------------------------------
    # Group recordings by source category.
    # ---------------------------------------------------------

    groups: dict[str, list[dict]] = {}

    for record in records:
        group = grouping_key(record)
        groups.setdefault(group, []).append(record)

    # ---------------------------------------------------------
    # Allocate groups while respecting global capacities.
    # ---------------------------------------------------------

    split_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    remaining_targets = dict(targets)

    # Process largest groups first for stable capacity allocation.
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    for group_name, group_records in ordered_groups:

        group_total = len(group_records)

        desired = {
            "train": group_total * TRAIN_FRACTION,
            "validation": group_total * VALIDATION_FRACTION,
            "test": group_total * TEST_FRACTION,
        }

        # Initial integer allocation.
        counts = {
            name: int(desired[name])
            for name in desired
        }

        remainder = group_total - sum(counts.values())

        # Largest remainder method.
        remainders = sorted(
            (
                desired[name] - counts[name],
                name,
            )
            for name in desired
        )

        for _, name in reversed(remainders[-remainder:]):
            counts[name] += 1

        # Never exceed remaining global capacity.
        for name in ("train", "validation", "test"):
            counts[name] = min(
                counts[name],
                remaining_targets[name],
            )

        allocated = sum(counts.values())

        # If rounding/capacity left records unallocated, fill remaining
        # global capacities deterministically.
        leftovers = group_total - allocated

        if leftovers > 0:
            split_order = sorted(
                (
                    remaining_targets[name] - counts[name],
                    name,
                )
                for name in ("train", "validation", "test")
            )

            for _, name in reversed(split_order):
                available = (
                    remaining_targets[name]
                    - counts[name]
                )

                add = min(available, leftovers)

                counts[name] += add
                leftovers -= add

                if leftovers == 0:
                    break

        if sum(counts.values()) != group_total:
            raise RuntimeError(
                f"Could not allocate group '{group_name}' "
                f"without violating global split targets."
            )

        allocation = allocate_group(
            group_records,
            counts["train"],
            counts["validation"],
            counts["test"],
        )

        for split in split_records:
            split_records[split].extend(
                allocation[split]
            )
            remaining_targets[split] -= len(
                allocation[split]
            )

    # ---------------------------------------------------------
    # Final exact-target check.
    # ---------------------------------------------------------

    actual_counts = {
        split: len(split_records[split])
        for split in split_records
    }

    if actual_counts != targets:
        raise RuntimeError(
            "Final split counts do not match the requested targets.\n"
            f"Expected: {targets}\n"
            f"Actual:   {actual_counts}"
        )

    # ---------------------------------------------------------
    # Add split field.
    # ---------------------------------------------------------

    for split in split_records:
        split_records[split] = [
            {
                **record,
                "split": split,
            }
            for record in split_records[split]
        ]

    # ---------------------------------------------------------
    # Create output directory.
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "train": OUTPUT_DIR / "train_manifest.jsonl",
        "validation": OUTPUT_DIR / "val_manifest.jsonl",
        "test": OUTPUT_DIR / "test_manifest.jsonl",
    }

    # ---------------------------------------------------------
    # Write deterministic manifests.
    # ---------------------------------------------------------

    for split, path in outputs.items():

        rows = sorted(
            split_records[split],
            key=lambda x: x["source_path"].lower(),
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:

            for row in rows:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # ---------------------------------------------------------
    # Print verification statistics.
    # ---------------------------------------------------------

    print("=" * 64)
    print("PHASE 0 — RECORDING-LEVEL STRATIFIED SPLIT")
    print("=" * 64)

    print(f"Total recordings: {total}")
    print()

    for split in ("train", "validation", "test"):
        count = len(split_records[split])
        percentage = 100.0 * count / total

        print(
            f"{split.capitalize():12s}: "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Expected:")
    print(
        f"Train:       {targets['train']} "
        f"({TRAIN_FRACTION * 100:.2f}%)"
    )
    print(
        f"Validation:  {targets['validation']} "
        f"({VALIDATION_FRACTION * 100:.2f}%)"
    )
    print(
        f"Test:        {targets['test']} "
        f"({TEST_FRACTION * 100:.2f}%)"
    )

    print("\nClass distribution:")

    for split in ("train", "validation", "test"):

        counter = Counter(
            (
                "clean"
                if record["source_group"] == "clean"
                else record["noise_class"]
            )
            for record in split_records[split]
        )

        print(
            f"  {split:10s}: {dict(counter)}"
        )

    print("\nCreated:")

    for path in outputs.values():
        print(f"  {path}")

    print(
        "\nIMPORTANT: these are recording-level splits. "
        "Do not segment or augment before this step."
    )


if __name__ == "__main__":
    main()