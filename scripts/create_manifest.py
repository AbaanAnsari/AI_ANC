"""
Phase 0 — Create the master recording-level dataset manifest.

Usage:
    python scripts/create_manifest.py

The raw dataset is never modified.

Output:
    data/manifests/dataset_manifest.jsonl

One manifest record represents one complete source recording.
Segmentation and synthetic noisy-speech generation happen later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.jsonl"


def metadata(path: Path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        return {
            "source_sample_rate_hz": rate,
            "source_channels": w.getnchannels(),
            "source_sample_width_bytes": w.getsampwidth(),
            "source_frames": frames,
            "duration_seconds": round(frames / rate if rate else 0.0, 6),
        }


def classify(root: Path, path: Path):
    parts = path.relative_to(root).parts

    if parts[0].lower() == "clean":
        return "clean", None

    if len(parts) >= 2 and parts[0].lower() == "noise":
        noise_class = parts[1].lower()

        if noise_class == "impulsive":
            return "impulsive", None

        if noise_class in {"stationary", "non-stationary"}:
            subclass = parts[2] if len(parts) >= 3 else None
            return noise_class, subclass

    return None, None


def make_record(root: Path, path: Path):
    rel = path.relative_to(root).as_posix()
    noise_class, subclass = classify(root, path)

    if noise_class is None:
        return None

    meta = metadata(path)
    record_id = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]

    source_group = "clean" if noise_class == "clean" else "noise"

    return {
        "record_id": record_id,
        "source_path": rel,
        "source_group": source_group,
        "noise_class": None if noise_class == "clean" else noise_class,
        "noise_subclass": subclass,
        **meta,
        # Canonical internal processing contract.
        "internal_sample_rate_hz": 16000,
        "internal_channels": 1,
        "internal_dtype": "float32",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    records = []
    skipped = []

    for path in sorted(root.rglob("*.wav")):
        try:
            record = make_record(root, path)
            if record is None:
                skipped.append(path.relative_to(root).as_posix())
            else:
                records.append(record)
        except Exception as exc:
            raise SystemExit(f"Could not read WAV metadata:\n{path}\n{exc}")

    # Recording identity must be unique.
    ids = [r["record_id"] for r in records]
    paths = [r["source_path"] for r in records]

    if len(ids) != len(set(ids)):
        raise SystemExit("Manifest creation failed: duplicate record_id detected.")

    if len(paths) != len(set(paths)):
        raise SystemExit("Manifest creation failed: duplicate source_path detected.")

    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 64)
    print("PHASE 0 — MASTER MANIFEST CREATED")
    print("=" * 64)
    print(f"Dataset root : {root}")
    print(f"Records      : {len(records)}")
    print(f"Skipped      : {len(skipped)}")
    print(f"Output       : {output}")
    print()
    print("The manifest is recording-level.")
    print("No segmentation or synthetic mixing was performed.")


if __name__ == "__main__":
    main()
