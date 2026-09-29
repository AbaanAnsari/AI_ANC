"""Repository-relative paths for the currently authoritative Phase 3 v2 artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE3_V2_CHECKPOINT = (
    PROJECT_ROOT / "experiments" / "phase3_100ep" / "best_checkpoint.pt"
)
PHASE3_V2_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "phase3_v2"
