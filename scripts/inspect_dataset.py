"""
Phase 0 — Inspect and freeze the working dataset.

This script does not modify the raw dataset.

Usage:
    python scripts/inspect_dataset.py

Optional:
    python scripts/inspect_dataset.py --root D:/SIH/SIH_2026/data
"""

from __future__ import annotations

import argparse
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "dataset_inspection.json",
    )
    return p.parse_args()


def inspect_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "sample_rate_hz": w.getframerate(),
            "channels": w.getnchannels(),
            "sample_width_bytes": w.getsampwidth(),
            "frames": w.getnframes(),
            "duration_seconds": (
                w.getnframes() / w.getframerate() if w.getframerate() else 0.0
            ),
        }


def classify(root: Path, path: Path):
    rel = path.relative_to(root)
    parts = rel.parts

    if not parts:
        return None, None

    if parts[0].lower() == "clean":
        return "clean", None

    if len(parts) >= 2 and parts[0].lower() == "noise":
        noise_class = parts[1].lower()
        if noise_class == "impulsive":
            return "impulsive", None
        if noise_class in {"stationary", "non-stationary"}:
            subclass = parts[2] if len(parts) >= 3 else None
            return noise_class, subclass

    return "unexpected", None


def main():
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    wavs = sorted(root.rglob("*.wav"))
    counts = Counter()
    durations = Counter()
    sample_rates = Counter()
    channels = Counter()
    invalid = []
    unexpected = []

    subclass_counts = defaultdict(int)

    for path in wavs:
        category, subclass = classify(root, path)

        if category == "unexpected":
            unexpected.append(path.relative_to(root).as_posix())
            continue

        if category is None:
            continue

        try:
            meta = inspect_wav(path)
        except Exception as exc:
            invalid.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "error": str(exc),
                }
            )
            continue

        counts[category] += 1
        durations[category] += meta["duration_seconds"]
        sample_rates[meta["sample_rate_hz"]] += 1
        channels[meta["channels"]] += 1

        if subclass:
            subclass_counts[f"{category}/{subclass}"] += 1

    report = {
        "dataset_root": str(root),
        "total_wav_files": len(wavs),
        "successfully_parsed_wav_files": sum(counts.values()),
        "total_duration_seconds": round(sum(durations.values()), 6),
        "categories": dict(counts),
        "category_durations_seconds": {k: round(v, 6) for k, v in durations.items()},
        "subcategories": dict(sorted(subclass_counts.items())),
        "sample_rates": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "invalid_audio_files": invalid,
        "unexpected_audio_files": unexpected,
        "phase_0_contract": {
            "raw_dataset_is_read_only": True,
            "internal_sample_rate_hz": 16000,
            "internal_channels": 1,
            "internal_dtype": "float32",
            "split_level": "recording",
        },
    }

    report_path = args.report
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 64)
    print("PHASE 0 — DATASET INSPECTION")
    print("=" * 64)
    print(f"Root:        {root}")
    print(f"WAV files:   {len(wavs)}")
    print(f"Parsed:      {sum(counts.values())}")
    print(f"Invalid:     {len(invalid)}")
    print()
    for category in ("clean", "impulsive", "non-stationary", "stationary"):
        print(f"{category:16s}: {counts.get(category, 0):5d}")
    print()
    print("Sample rates:", dict(sorted(sample_rates.items())))
    print("Channels:    ", dict(sorted(channels.items())))
    print()
    print(f"Report: {report_path}")

    if invalid:
        print("\nWARNING: invalid audio files detected.")
    if unexpected:
        print("\nWARNING: unexpected WAV files detected.")

    print("\nInspection complete. Raw dataset was not modified.")


if __name__ == "__main__":
    main()
