"""
scripts/train_phase3_100ep.py
=============================
Phase 3 Full 100-Epoch Retraining Pipeline.
Trains LightweightCNNGRUMaskModel from scratch for 100 epochs on phase3_v2 manifests:
- Epoch-aware deterministic stochastic mixture generation
- Zero speaker leakage (disjoint Train/Val/Test splits)
- Mathematically consistent composite loss (scaled torch.istft and non-negative SI-SDR)
- Logs sample 0 fingerprint per epoch
- Outputs best_checkpoint.pt, final_checkpoint.pt, config.json, training_history.json, training_history.csv
- Directory: experiments/phase3_100ep/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import Phase1Dataset, custom_collate_fn
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.phase3_losses import Phase3MultiTaskLoss
from src.training.training_config import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_phase3_100ep")

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase3_v2_training.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    PHASE3_CONFIG = yaml.safe_load(config_file)

EXPERIMENT_CONFIG = PHASE3_CONFIG["experiment"]
TRAINING_CONFIG = PHASE3_CONFIG["training"]
LOSS_WEIGHTS = PHASE3_CONFIG["loss_weights"]
DATASET_CONFIG = PHASE3_CONFIG["dataset"]

EXP_DIR = PROJECT_ROOT / "experiments" / "phase3_100ep"
MANIFEST_DIR = PROJECT_ROOT / DATASET_CONFIG["manifest_dir"]
DATASET_ROOT = PROJECT_ROOT / DATASET_CONFIG["root"]

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"

BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
LR = float(TRAINING_CONFIG["learning_rate"])
WEIGHT_DECAY = float(TRAINING_CONFIG["weight_decay"])
GRAD_CLIP = float(TRAINING_CONFIG["gradient_clip_norm"])
DEFAULT_EPOCHS = int(TRAINING_CONFIG["epochs"])
SEED = int(TRAINING_CONFIG["seed"])
DATASET_SEED = int(TRAINING_CONFIG["dataset_seed"])


def fingerprint_sample(sample: dict) -> str:
    arr = sample["m1_waveform"].cpu().numpy().astype(np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        commit.stdout.strip() if commit.returncode == 0 else None,
        bool(status.stdout.strip()) if status.returncode == 0 else None,
    )


def assert_finite_losses(
    losses: dict[str, torch.Tensor], epoch: int, phase: str
) -> None:
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite {phase} loss {name} at epoch {epoch}: {value}"
            )


def run_training_100ep(epochs: int = DEFAULT_EPOCHS, device_str: str = "cpu"):
    set_seed(SEED)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info("=" * 75)
    logger.info("STARTING PHASE 3 100-EPOCH RETRAINING FROM SCRATCH")
    logger.info("Experiment Directory: %s", EXP_DIR)
    logger.info(
        "Device: %s | Epochs: %d | Batch Size: %d | Seed: %d",
        device,
        epochs,
        BATCH_SIZE,
        SEED,
    )
    logger.info("=" * 75)

    # 1. Datasets and Loaders
    train_ds = Phase1Dataset(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        seed=DATASET_SEED,
        is_validation=False,
    )
    val_ds = Phase1Dataset(
        manifest_path=VAL_MANIFEST,
        dataset_root=DATASET_ROOT,
        seed=DATASET_SEED,
        is_validation=True,
    )

    logger.info("Train dataset: %d clean segments (%s)", len(train_ds), TRAIN_MANIFEST)
    logger.info("Val dataset:   %d clean segments (%s)", len(val_ds), VAL_MANIFEST)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=int(TRAINING_CONFIG["num_workers"]),
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=int(TRAINING_CONFIG["num_workers"]),
    )

    # 2. Model: LightweightCNNGRUMaskModel (70,789 trainable params)
    model = LightweightCNNGRUMaskModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: %d total, %d trainable", total_params, trainable_params
    )
    assert (
        trainable_params == 70789
    ), f"Expected 70,789 trainable params, got {trainable_params}"

    # 3. Loss & Optimizer (Balanced composite)
    multi_loss = Phase3MultiTaskLoss(**LOSS_WEIGHTS).to(device)

    if TRAINING_CONFIG["optimizer"] != "AdamW":
        raise ValueError(
            f"Unsupported Phase 3 optimizer: {TRAINING_CONFIG['optimizer']}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_config = TRAINING_CONFIG["scheduler"]
    if scheduler_config["type"] != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported Phase 3 scheduler: {scheduler_config['type']}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_lr"]),
    )

    manifest_hashes = {
        "train": sha256_file(TRAIN_MANIFEST),
        "val": sha256_file(VAL_MANIFEST),
        "test": sha256_file(TEST_MANIFEST),
    }
    git_commit, git_dirty = git_provenance()
    config_path_relative = CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    config_dict = {
        "experiment_id": EXPERIMENT_CONFIG["id"],
        "experiment_name": EXPERIMENT_CONFIG["name"],
        "model_architecture": "LightweightCNNGRUMaskModel",
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "initial_learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "random_seed": SEED,
        "dataset_seed": DATASET_SEED,
        "num_workers": int(TRAINING_CONFIG["num_workers"]),
        "mixed_precision": bool(TRAINING_CONFIG["mixed_precision"]),
        "optimizer": TRAINING_CONFIG["optimizer"],
        "scheduler": scheduler_config,
        "epoch_seed_scheme": "sha256(global_seed:epoch:index)",
        "phase3_config_path": config_path_relative,
        "phase3_config_sha256": sha256_file(CONFIG_PATH),
        "git_commit": git_commit,
        "git_working_tree_dirty": git_dirty,
        "software_versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "training_hardware": {
            "device": str(device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
        },
        "manifests": {
            "train": TRAIN_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "val": VAL_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "test": TEST_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        },
        "manifest_sha256": manifest_hashes,
        "loss_weights": LOSS_WEIGHTS,
    }
    with open(EXP_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy dataset statistics report as dataset_audit.json
    stats_src = (
        PROJECT_ROOT / "experiments" / "phase3_v2" / "dataset_statistics_report.json"
    )
    if stats_src.exists():
        with open(stats_src) as f:
            d_audit = json.load(f)
        with open(EXP_DIR / "dataset_audit.json", "w") as f:
            json.dump(d_audit, f, indent=2)

    # 4. 100-Epoch Training Loop with Epoch Variety Fingerprinting
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    fingerprint_log = {}

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Update dataset epoch (CRITICAL FOR DIVERSITY)
        train_ds.set_epoch(epoch)

        # Log sample 0 fingerprint to demonstrate epoch diversity during training
        sample0 = train_ds[0]
        fp0 = fingerprint_sample(sample0)
        fingerprint_log[epoch] = {
            "sample_0_hash": fp0,
            "sample_0_noise": sample0["metadata"]["noise_source_path"],
            "sample_0_snr": float(sample0["target_snr_db"]),
            "sample_0_delay": int(sample0["configured_delay_samples"]),
        }

        # Train epoch
        model.train()
        train_loss_sum = 0.0
        train_enh_sum = 0.0
        train_cls_sum = 0.0
        train_sisdr_sum = 0.0
        n_train_batches = 0

        for batch in train_loader:
            noisy_feat = batch["noisy_features"].to(device)
            target_stft = batch["target_stft"].to(device)
            noise_lbl = batch["noise_class_label"].to(device)
            clean_wave = batch["clean_target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            enh_out, cls_logits, mask = model(noisy_feat)

            losses = multi_loss(
                enhanced_output=enh_out,
                predicted_mask=mask,
                classification_logits=cls_logits,
                noisy_features=noisy_feat,
                target_stft=target_stft,
                noise_class_label=noise_lbl,
                clean_waveform=clean_wave,
            )
            assert_finite_losses(losses, epoch, "training")

            total_l = losses["total_loss"]
            total_l.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP, error_if_nonfinite=True
            )
            optimizer.step()

            train_loss_sum += total_l.item()
            train_enh_sum += losses["enhancement_loss"].item()
            train_cls_sum += losses["classification_loss"].item()
            train_sisdr_sum += losses["sisdr_loss"].item()
            n_train_batches += 1

        # Validate epoch (deterministic validation)
        model.eval()
        val_loss_sum = 0.0
        val_enh_sum = 0.0
        val_cls_sum = 0.0
        val_sisdr_sum = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                noisy_feat = batch["noisy_features"].to(device)
                target_stft = batch["target_stft"].to(device)
                noise_lbl = batch["noise_class_label"].to(device)
                clean_wave = batch["clean_target"].to(device)

                enh_out, cls_logits, mask = model(noisy_feat)

                losses = multi_loss(
                    enhanced_output=enh_out,
                    predicted_mask=mask,
                    classification_logits=cls_logits,
                    noisy_features=noisy_feat,
                    target_stft=target_stft,
                    noise_class_label=noise_lbl,
                    clean_waveform=clean_wave,
                )
                assert_finite_losses(losses, epoch, "validation")

                val_loss_sum += losses["total_loss"].item()
                val_enh_sum += losses["enhancement_loss"].item()
                val_cls_sum += losses["classification_loss"].item()
                val_sisdr_sum += losses["sisdr_loss"].item()
                n_val_batches += 1

                preds = torch.argmax(cls_logits, dim=-1)
                val_correct += (preds == noise_lbl).sum().item()
                val_total += len(noise_lbl)

        epoch_time = time.perf_counter() - t0
        mean_tr_loss = train_loss_sum / max(1, n_train_batches)
        mean_tr_enh = train_enh_sum / max(1, n_train_batches)
        mean_tr_cls = train_cls_sum / max(1, n_train_batches)
        mean_tr_sisdr = train_sisdr_sum / max(1, n_train_batches)

        mean_val_loss = val_loss_sum / max(1, n_val_batches)
        mean_val_enh = val_enh_sum / max(1, n_val_batches)
        mean_val_cls = val_cls_sum / max(1, n_val_batches)
        val_acc = val_correct / max(1, val_total)

        scheduler.step(mean_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = mean_val_loss < best_val_loss
        if is_best:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "experiment_id": "phase3_100ep",
                    "experiment_name": "Phase 3 100-Epoch Retrained Model",
                    "config": config_dict,
                },
                EXP_DIR / "best_checkpoint.pt",
            )

        # Save final checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "final_val_loss": mean_val_loss,
                "experiment_id": "phase3_100ep",
                "experiment_name": "Phase 3 100-Epoch Retrained Model",
                "config": config_dict,
            },
            EXP_DIR / "final_checkpoint.pt",
        )

        record = {
            "epoch": epoch,
            "train_total_loss": round(mean_tr_loss, 6),
            "train_enhancement_loss": round(mean_tr_enh, 6),
            "train_classification_loss": round(mean_tr_cls, 6),
            "train_sisdr_loss": round(mean_tr_sisdr, 6),
            "val_total_loss": round(mean_val_loss, 6),
            "val_enhancement_loss": round(mean_val_enh, 6),
            "val_classification_loss": round(mean_val_cls, 6),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": current_lr,
            "epoch_time_s": round(epoch_time, 2),
            "is_best": is_best,
            "sample_0_hash": fp0,
        }
        history.append(record)

        # Save history json & csv incrementally
        with open(EXP_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        keys = list(record.keys())
        with open(EXP_DIR / "training_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)

        with open(EXP_DIR / "fingerprint_log.json", "w") as f:
            json.dump(fingerprint_log, f, indent=2)

        star = "*" if is_best else " "
        logger.info(
            "Epoch %3d/%3d [%4.1fs] | TrLoss: %+7.4f (enh: %+7.4f) | ValLoss: %+7.4f (enh: %+7.4f, acc: %4.1f%%) | Best: %+7.4f (Ep %d) | Hash0: %s %s",
            epoch,
            epochs,
            epoch_time,
            mean_tr_loss,
            mean_tr_enh,
            mean_val_loss,
            mean_val_enh,
            val_acc * 100.0,
            best_val_loss,
            best_epoch,
            fp0,
            star,
        )

    logger.info("=" * 75)
    logger.info(
        "100-EPOCH TRAINING COMPLETE! Best val loss: %f at epoch %d",
        best_val_loss,
        best_epoch,
    )
    logger.info("Saved best checkpoint to %s", EXP_DIR / "best_checkpoint.pt")
    logger.info("Saved final checkpoint to %s", EXP_DIR / "final_checkpoint.pt")
    logger.info("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 100-Epoch Retraining")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_training_100ep(epochs=args.epochs, device_str=args.device)
