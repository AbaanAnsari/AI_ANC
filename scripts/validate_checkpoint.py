"""Validate a production checkpoint and optionally write its reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.streaming_pipeline import (
    HOP_SIZE,
    LOOKAHEAD_HOPS,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_SIZE,
)
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.project_paths import PHASE3_V2_MANIFEST_DIR

EXPECTED_ARCHITECTURE = "LightweightCNNGRUMaskModel"
EXPECTED_PARAMETERS = 70_789


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], LightweightCNNGRUMaskModel]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint must contain model_state_dict: {path}")

    config = checkpoint.get("config") or {}
    architecture = checkpoint.get("model_architecture") or config.get(
        "model_architecture"
    )
    if architecture is not None and architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Architecture mismatch: checkpoint says {architecture!r}, expected {EXPECTED_ARCHITECTURE!r}"
        )

    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Parameter count mismatch: loaded {parameter_count}, expected {EXPECTED_PARAMETERS}"
        )
    return checkpoint, model


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
    }


def build_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    checkpoint, model = validate_checkpoint(path)
    config = checkpoint.get("config") or {}
    experiment_dir = path.parent
    config_path = experiment_dir / "config.json"
    recipe_path = PROJECT_ROOT / "configs" / "phase3_v2_training.yaml"
    recipe_hash = sha256_file(recipe_path) if recipe_path.is_file() else None
    checkpoint_recipe_hash = config.get("phase3_config_sha256")
    manifest_sources = config.get("manifests", {})
    manifest_hashes: dict[str, str | None] = {}
    manifest_paths: dict[str, str | None] = {}
    for key, fallback_name in (
        ("train", "train_manifest.jsonl"),
        ("val", "val_manifest.jsonl"),
        ("test", "test_manifest.jsonl"),
    ):
        configured = Path(str(manifest_sources.get(key, "")))
        candidate = (
            configured
            if configured.is_file()
            else PHASE3_V2_MANIFEST_DIR / fallback_name
        )
        if candidate.is_file():
            try:
                manifest_paths[key] = (
                    candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
                )
            except ValueError:
                manifest_paths[key] = None
            manifest_hashes[key] = sha256_file(candidate)
        else:
            manifest_paths[key] = None
            manifest_hashes[key] = None

    try:
        checkpoint_relative = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        checkpoint_relative = None

    return {
        "experiment_id": checkpoint.get("experiment_id"),
        "metadata_generation_git": _git_provenance(),
        "training_git_commit": config.get("git_commit"),
        "training_working_tree_dirty": config.get("git_working_tree_dirty"),
        "model_architecture": EXPECTED_ARCHITECTURE,
        "model_version": checkpoint.get("experiment_id"),
        "parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "config_path": (
            config_path.relative_to(PROJECT_ROOT).as_posix()
            if config_path.is_file()
            else None
        ),
        "config_hash_sha256": (
            sha256_file(config_path) if config_path.is_file() else None
        ),
        "current_recipe_path": (
            recipe_path.relative_to(PROJECT_ROOT).as_posix()
            if recipe_path.is_file()
            else None
        ),
        "current_recipe_sha256": recipe_hash,
        "current_recipe_matches_checkpoint": bool(
            recipe_hash and checkpoint_recipe_hash == recipe_hash
        ),
        "manifests": {
            "paths": manifest_paths,
            "sha256": manifest_hashes,
        },
        "checkpoint_path": checkpoint_relative,
        "checkpoint_sha256": sha256_file(path),
        "random_seed": config.get("random_seed"),
        "training_epochs": config.get("epochs"),
        "best_epoch": checkpoint.get("epoch"),
        "validation_metric": "best_val_loss" if "best_val_loss" in checkpoint else None,
        "validation_metric_value": checkpoint.get("best_val_loss"),
        "training_config": config,
        "training_software_versions": config.get("software_versions"),
        "metadata_generation_environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "training_hardware": config.get("training_hardware"),
        "signal_config": {
            "sample_rate_hz": SAMPLE_RATE,
            "fft_size": N_FFT,
            "hop_size": HOP_SIZE,
            "window_size": WINDOW_SIZE,
            "window": "hann",
            "lookahead_frames": LOOKAHEAD_HOPS,
        },
        "metadata_completeness": "partial: training-time Git state, software versions, and hardware were not recorded in this checkpoint",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args(argv)
    metadata = build_metadata(args.checkpoint)
    if args.metadata_output:
        output = args.metadata_output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Metadata: {output}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint": metadata["checkpoint_path"],
                "sha256": metadata["checkpoint_sha256"],
                "architecture": metadata["model_architecture"],
                "parameters": metadata["parameter_count"],
                "epoch": metadata["best_epoch"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
