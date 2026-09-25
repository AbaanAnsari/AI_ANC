"""
evaluation/evaluate.py — Evaluation Runner
==========================================
Runs formal evaluation (MODE A: val, MODE B: test) using the saved checkpoint.

Usage:
    python evaluation/evaluate.py --mode val
    python evaluation/evaluate.py --mode test
    python evaluation/evaluate.py --mode both
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import create_dataloader
from src.training.losses import MultiTaskLoss
from src.training.training_config import TrainingConfig, set_seed
from src.training.validation import (
    evaluate,
    load_model_from_checkpoint,
    print_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"
BEST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"


def run_evaluation(mode: str = "both") -> dict:
    """
    Run evaluation on val and/or test sets.

    Parameters
    ----------
    mode : str
        'val', 'test', or 'both'.

    Returns
    -------
    dict with 'val' and/or 'test' keys.
    """
    config = TrainingConfig.create(device="auto")
    set_seed(config.seed)

    logger.info("Loading model from: %s", BEST_CHECKPOINT)
    model, checkpoint = load_model_from_checkpoint(BEST_CHECKPOINT, device=config.device)

    loss_fn = MultiTaskLoss(
        enhancement_weight=config.loss.enhancement_weight,
        classification_weight=config.loss.classification_weight,
        enhancement_l1_weight=config.loss.enhancement_l1_weight,
        enhancement_l2_weight=config.loss.enhancement_l2_weight,
    )

    results = {}

    if mode in ("val", "both"):
        logger.info("MODE A: Validation set evaluation")
        val_loader = create_dataloader(
            manifest_path=VAL_MANIFEST,
            dataset_root=DATASET_ROOT,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
        )
        val_result = evaluate(model, val_loader, loss_fn, device=config.device)
        print_evaluation_report(val_result, title="PHASE 1 STEP 16 — VALIDATION SET RESULTS")
        results["val"] = val_result

    if mode in ("test", "both"):
        logger.info("MODE B: Test set evaluation")
        test_loader = create_dataloader(
            manifest_path=TEST_MANIFEST,
            dataset_root=DATASET_ROOT,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
        )
        test_result = evaluate(model, test_loader, loss_fn, device=config.device)
        print_evaluation_report(test_result, title="PHASE 1 STEP 16 — TEST SET RESULTS")
        results["test"] = test_result

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()
    run_evaluation(args.mode)
