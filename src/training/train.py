"""
Phase 1 Step 14/15 — Training Entry Point
==========================================
Executable script for controlled training runs.

Usage (controlled 2-epoch run):
    python src/training/train.py --epochs 2

Usage (full 30-epoch run):
    python src/training/train.py --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---- ensure project root is on path ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import create_dataloader
from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.trainer import create_trainer
from src.training.training_config import TrainingConfig, set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(epochs: int = 2, checkpoint_dir: str = "models/checkpoints") -> None:
    """
    Execute a controlled training run.

    Parameters
    ----------
    epochs : int
        Number of epochs to train (default 2 for controlled validation run).
    checkpoint_dir : str
        Directory to save checkpoints.
    """
    logger.info("=" * 64)
    logger.info("PHASE 1 — TRAINING RUN")
    logger.info("epochs=%d | checkpoint_dir=%s", epochs, checkpoint_dir)
    logger.info("=" * 64)

    # --- Locked configuration ---
    config = TrainingConfig.create(
        device="auto",
        batch_size=16,
        epochs=epochs,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=5.0,
        seed=123,
        num_workers=0,
        validation_frequency=1,
        log_frequency=10,
        mixed_precision=False,
    )

    logger.info("Device: %s", config.device)
    logger.info("Seed:   %d", config.seed)

    # --- Reproducibility ---
    set_seed(config.seed)

    # --- DataLoaders ---
    logger.info("Loading train manifest: %s", TRAIN_MANIFEST)
    train_loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    logger.info("Loading val manifest:   %s", VAL_MANIFEST)
    val_loader = create_dataloader(
        manifest_path=VAL_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    logger.info("Train batches: %d | Val batches: %d", n_train_batches, n_val_batches)

    # --- Model ---
    model = LightweightCNNTGRUModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: LightweightCNNTGRUModel | Trainable params: %d", trainable)
    assert trainable == 70_789, f"Unexpected param count: {trainable}"

    # --- Trainer ---
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer._checkpoint_dir = Path(checkpoint_dir)

    # --- Run ---
    run_start = time.perf_counter()
    history = trainer.run(epochs=epochs, checkpoint_dir=checkpoint_dir)
    total_duration = time.perf_counter() - run_start

    # --- Report ---
    logger.info("=" * 64)
    logger.info("TRAINING COMPLETE")
    logger.info("Total time: %.1fs", total_duration)
    logger.info("=" * 64)

    nan_detected = False
    for record in history:
        epoch = int(record["epoch"])
        tl = record.get("train_total_loss", float("nan"))
        te = record.get("train_enhancement_loss", float("nan"))
        tc = record.get("train_classification_loss", float("nan"))
        vl = record.get("val_total_loss", float("nan"))
        ve = record.get("val_enhancement_loss", float("nan"))
        vc = record.get("val_classification_loss", float("nan"))
        lr = record.get("learning_rate", float("nan"))
        dur = record.get("epoch_duration_seconds", float("nan"))

        logger.info(
            "Epoch %2d | train_loss=%.6f  enh=%.6f  cls=%.6f "
            "| val_loss=%.6f  enh=%.6f  cls=%.6f "
            "| lr=%.2e | %.1fs",
            epoch,
            tl,
            te,
            tc,
            vl,
            ve,
            vc,
            lr,
            dur,
        )

        # NaN/Inf check
        for val in (tl, te, tc, vl, ve, vc):
            if val != val or (val == float("inf")):  # NaN or Inf check
                nan_detected = True

    if nan_detected:
        logger.warning("WARNING: NaN or Inf detected in training metrics!")
    else:
        logger.info("NaN/Inf check: PASS — all losses are finite")

    logger.info("Best val_total_loss: %.6f", trainer._best_val_loss)
    logger.info("Checkpoint dir: %s", Path(checkpoint_dir).resolve())

    # --- Verify checkpoints exist ---
    best_ckpt = Path(checkpoint_dir) / "best_checkpoint.pt"
    last_ckpt = Path(checkpoint_dir) / "last_checkpoint.pt"
    logger.info("best_checkpoint exists: %s", best_ckpt.exists())
    logger.info("last_checkpoint exists: %s", last_ckpt.exists())

    # --- Verify checkpoint can be loaded and model has 70,789 params ---
    if best_ckpt.exists():
        ckpt = Trainer_static_load(best_ckpt)
        verify_model = LightweightCNNTGRUModel()
        verify_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        loaded_params = sum(
            p.numel() for p in verify_model.parameters() if p.requires_grad
        )
        logger.info("Checkpoint model param count: %d", loaded_params)
        assert loaded_params == 70_789
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        stored_epoch = ckpt["epoch"]
        stored_best_loss = ckpt["best_val_loss"]
        logger.info(
            "Checkpoint epoch: %d | stored best_val_loss: %.6f",
            stored_epoch,
            stored_best_loss,
        )
        assert stored_epoch >= 1
        assert stored_best_loss < float("inf")
        logger.info("Checkpoint verification: PASS")


def Trainer_static_load(filepath):
    """Load checkpoint dict without instantiating Trainer."""
    import torch

    return torch.load(filepath, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Training Run")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2 for controlled run)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    main(epochs=args.epochs, checkpoint_dir=args.checkpoint_dir)
