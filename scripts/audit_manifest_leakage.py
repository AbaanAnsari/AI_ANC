"""Fail-loud audit of source overlap across train, validation, and test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRISPEECH_SPEAKER = re.compile(r"^(\d+)-\d+-\d+\.wav$", re.IGNORECASE)
SPLIT_FILES = {
    "train": "train_manifest.jsonl",
    "validation": "val_manifest.jsonl",
    "test": "test_manifest.jsonl",
}


def _load_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Manifest is empty: {path}")
    records = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(records, list) or any(
        not isinstance(row, dict) for row in records
    ):
        raise ValueError(f"Manifest must contain JSON objects: {path}")
    return records


def audit_manifests(manifest_paths: dict[str, Path], dataset_root: Path) -> list[str]:
    """Return sorted errors; an empty list means no detected leakage."""
    records_by_split = {
        split: _load_manifest(path) for split, path in manifest_paths.items()
    }
    identities: dict[str, dict[str, set[str]]] = {
        key: defaultdict(set)
        for key in ("record_id", "path", "filename", "sha256", "speaker_id")
    }
    errors: list[str] = []

    for split in sorted(records_by_split):
        seen_ids: set[str] = set()
        for row_index, record in enumerate(records_by_split[split], start=1):
            location = f"{split} record {row_index}"
            record_id = str(record.get("record_id", "")).strip()
            if not record_id:
                errors.append(f"{location}: missing record_id")
            elif record_id in seen_ids:
                errors.append(
                    f"{location}: duplicate record_id {record_id!r} within split"
                )
            else:
                seen_ids.add(record_id)
                identities["record_id"][record_id].add(split)

            source = str(record.get("source_path", "")).strip()
            if not source:
                errors.append(f"{location}: missing source_path")
                continue
            source_path = Path(source)
            path = (
                source_path if source_path.is_absolute() else dataset_root / source_path
            ).resolve()
            if not path.is_file():
                errors.append(f"{location}: source file missing: {path}")
                continue

            identities["path"][str(path).casefold()].add(split)
            with path.open("rb") as source_file:
                digest = hashlib.file_digest(source_file, "sha256").hexdigest()
            identities["sha256"][digest].add(split)

            if record.get("source_group") == "clean":
                identities["filename"][path.name.casefold()].add(split)
                match = LIBRISPEECH_SPEAKER.match(path.name)
                declared = str(record.get("speaker_id", "")).strip()
                if match is None:
                    errors.append(
                        f"{location}: clean source has no verifiable speaker ID: {source}"
                    )
                else:
                    speaker_id = match.group(1)
                    if declared and declared != speaker_id:
                        errors.append(
                            f"{location}: speaker_id {declared!r} conflicts with filename ID {speaker_id!r}"
                        )
                    identities["speaker_id"][speaker_id].add(split)

    for identity_type in ("record_id", "path", "sha256", "speaker_id", "filename"):
        for identity, splits in sorted(identities[identity_type].items()):
            if len(splits) > 1:
                errors.append(
                    f"{identity_type} overlap across splits {', '.join(sorted(splits))}: {identity}"
                )
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/manifests/phase3_v2")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/dataset"))
    args = parser.parse_args(argv)
    manifest_dir = (
        args.manifest_dir
        if args.manifest_dir.is_absolute()
        else PROJECT_ROOT / args.manifest_dir
    )
    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else PROJECT_ROOT / args.dataset_root
    )
    paths = {split: manifest_dir / name for split, name in SPLIT_FILES.items()}

    try:
        errors = audit_manifests(paths, dataset_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LEAKAGE AUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    if errors:
        print(f"LEAKAGE AUDIT FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "LEAKAGE AUDIT PASSED: paths, hashes, record IDs, filenames, and speakers are disjoint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
