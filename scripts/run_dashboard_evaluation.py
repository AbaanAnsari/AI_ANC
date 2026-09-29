"""Run held-out evaluation metrics or the downstream DSP ablation suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("metrics", "ablation"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    from evaluation.evaluate_model import BEST_CHECKPOINT, TEST_MANIFEST
    checkpoint = args.checkpoint or BEST_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.task == "metrics":
        from evaluation.evaluate_model import print_final_report, run_evaluation

        manifest = TEST_MANIFEST
        if not manifest.is_file():
            raise FileNotFoundError(f"Test manifest not found: {manifest}")
        results_dir = PROJECT_ROOT / "experiments" / "final_project_validation"
        metrics, _ = run_evaluation(
            checkpoint_path=checkpoint,
            test_manifest=manifest,
            output_dir=results_dir,
        )
        print_final_report(metrics)
        print(f"Results: {results_dir / 'aggregated_metrics.json'}")
        return

    from evaluation.benchmark_downstream_ablation import (
        CLEAN_SPEECH_FILE,
        NOISE_FILES,
        run_full_ablation_suite,
    )

    required_audio = [CLEAN_SPEECH_FILE, *NOISE_FILES.values()]
    missing_audio = [path for path in required_audio if not path.is_file()]
    if missing_audio:
        missing = "\n".join(str(path) for path in missing_audio)
        raise FileNotFoundError(f"Ablation audio files not found:\n{missing}")

    results = run_full_ablation_suite(checkpoint_path=checkpoint)
    output_path = PROJECT_ROOT / "experiments" / "dashboard_evaluation" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()