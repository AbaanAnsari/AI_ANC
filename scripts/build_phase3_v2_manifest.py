"""
scripts/build_phase3_v2_manifest.py
====================================
Build Phase 3 v2 recording-level and speaker-partitioned manifests.
Guarantees:
1. ZERO speaker leakage between Train, Validation, and Test.
2. ZERO noise recording overlap between Train, Validation, and Test.
3. Every source recording belongs to exactly ONE split.
4. Stratified noise representation across all classes (stationary, non-stationary, impulsive).
5. Deterministic generation using fixed SEED=20260929.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260929
OUTPUT_DIR = Path("data/manifests/phase3_v2")
MASTER_MANIFEST = Path("data/manifests/dataset_manifest.jsonl")


def deterministic_key(s: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{s}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def get_speaker_id(record: dict) -> str:
    src = record["source_path"]
    base = os.path.basename(src)
    if "-" in base:
        return base.split("-")[0]
    elif "_" in base:
        return base.split("_")[0]
    return "synthetic"


def build_phase3_v2_manifests():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MASTER_MANIFEST, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {MASTER_MANIFEST}")

    clean_records = [r for r in records if r["source_group"] == "clean"]
    noise_records = [r for r in records if r["source_group"] == "noise"]

    print(f"Clean records: {len(clean_records)}, Noise records: {len(noise_records)}")

    # -------------------------------------------------------------
    # 1. Partition Clean Records by Speaker
    # -------------------------------------------------------------
    speaker_groups = defaultdict(list)
    for r in clean_records:
        spk = get_speaker_id(r)
        speaker_groups[spk].append(r)

    # Sort speaker IDs deterministically
    sorted_speakers = sorted(speaker_groups.keys(), key=deterministic_key)

    # Ensure speaker 1098 is in TRAIN so the synthetic 12-case benchmark (using 1098-133695-0000.wav)
    # has identical matched-condition comparability with Checkpoint D
    # while unseen speakers test generalization
    train_speakers = []
    val_speakers = []
    test_speakers = []

    # Allocate speakers: ~70% train, ~15% val, ~15% test
    # We prioritize larger speaker sets to ensure adequate volume in each split
    train_count = 0
    val_count = 0
    test_count = 0
    total_clean = len(clean_records)

    # Put 1098 in train explicitly
    train_speakers.append("1098")
    train_count += len(speaker_groups["1098"])

    remaining_speakers = [s for s in sorted_speakers if s != "1098"]

    for spk in remaining_speakers:
        count = len(speaker_groups[spk])
        # If val needs more (target ~15% = 202)
        if val_count + count <= 0.18 * total_clean and len(val_speakers) < len(remaining_speakers) // 4:
            val_speakers.append(spk)
            val_count += count
        elif test_count + count <= 0.18 * total_clean and len(test_speakers) < len(remaining_speakers) // 4:
            test_speakers.append(spk)
            test_count += count
        else:
            train_speakers.append(spk)
            train_count += count

    print(f"\nClean Split by Speaker:")
    print(f"  TRAIN: {train_count} files across {len(train_speakers)} speakers: {train_speakers}")
    print(f"  VAL:   {val_count} files across {len(val_speakers)} speakers: {val_speakers}")
    print(f"  TEST:  {test_count} files across {len(test_speakers)} speakers: {test_speakers}")

    # Build clean split dictionaries
    split_clean = {"train": [], "validation": [], "test": []}
    for spk in train_speakers:
        for r in speaker_groups[spk]:
            r_copy = dict(r)
            r_copy["split"] = "train"
            r_copy["speaker_id"] = spk
            split_clean["train"].append(r_copy)

    for spk in val_speakers:
        for r in speaker_groups[spk]:
            r_copy = dict(r)
            r_copy["split"] = "val"
            r_copy["speaker_id"] = spk
            split_clean["validation"].append(r_copy)

    for spk in test_speakers:
        for r in speaker_groups[spk]:
            r_copy = dict(r)
            r_copy["split"] = "test"
            r_copy["speaker_id"] = spk
            split_clean["test"].append(r_copy)

    # -------------------------------------------------------------
    # 2. Partition Noise Records by Category & Source File
    # -------------------------------------------------------------
    # Group noise by (noise_class, noise_subclass)
    noise_subgroups = defaultdict(list)
    for r in noise_records:
        key = (r["noise_class"], r.get("noise_subclass", "none"))
        noise_subgroups[key].append(r)

    split_noise = {"train": [], "validation": [], "test": []}

    for (n_cls, n_sub), n_list in noise_subgroups.items():
        # Sort files deterministically
        sorted_n = sorted(n_list, key=lambda r: deterministic_key(r["record_id"]))
        n_total = len(sorted_n)
        n_tr = int(round(0.70 * n_total))
        n_val = int(round(0.15 * n_total))
        n_te = n_total - n_tr - n_val

        tr_slice = sorted_n[:n_tr]
        val_slice = sorted_n[n_tr : n_tr + n_val]
        te_slice = sorted_n[n_tr + n_val :]

        for r in tr_slice:
            rc = dict(r)
            rc["split"] = "train"
            split_noise["train"].append(rc)
        for r in val_slice:
            rc = dict(r)
            rc["split"] = "val"
            split_noise["validation"].append(rc)
        for r in te_slice:
            rc = dict(r)
            rc["split"] = "test"
            split_noise["test"].append(rc)

    print("\nNoise Split by Category:")
    for split_name in ["train", "validation", "test"]:
        cnt = Counter((r["noise_class"], r["noise_subclass"]) for r in split_noise[split_name])
        print(f"  {split_name.upper()} ({len(split_noise[split_name])} files):")
        for k, v in sorted(cnt.items()):
            print(f"    {k[0]} / {k[1]}: {v}")

    # -------------------------------------------------------------
    # 3. Write Manifest Files
    # -------------------------------------------------------------
    split_map = {
        "train": "train_manifest.jsonl",
        "validation": "val_manifest.jsonl",
        "test": "test_manifest.jsonl",
    }

    manifest_metadata = {
        "dataset_version": "phase3_v2",
        "seed": SEED,
        "clean_speaker_splits": {
            "train": train_speakers,
            "validation": val_speakers,
            "test": test_speakers,
        },
        "counts": {},
    }

    for split_name, filename in split_map.items():
        combined = split_clean[split_name] + split_noise[split_name]
        out_path = OUTPUT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for r in combined:
                f.write(json.dumps(r) + "\n")

        manifest_metadata["counts"][split_name] = {
            "total_records": len(combined),
            "clean_records": len(split_clean[split_name]),
            "noise_records": len(split_noise[split_name]),
            "speakers": train_speakers if split_name == "train" else (val_speakers if split_name == "validation" else test_speakers),
        }
        print(f"Wrote {len(combined)} records to {out_path}")

    meta_path = OUTPUT_DIR / "manifest_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest_metadata, f, indent=2)
    print(f"Wrote manifest metadata to {meta_path}")


if __name__ == "__main__":
    build_phase3_v2_manifests()
