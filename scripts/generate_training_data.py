"""
Phase 1 placeholder — synthetic supervised training-data generation.

This file intentionally remains in the original locked project structure.

Do NOT run this during Phase 0.

Phase 1 will implement:
    clean speech + selected noise + target SNR
        -> synthetic noisy speech
        -> supervised training sample

The implementation will consume the recording-level manifests produced by
Phase 0 and will generate mixtures without changing the frozen raw dataset.
"""

from __future__ import annotations


def main():
    raise SystemExit(
        "generate_training_data.py belongs to Phase 1. "
        "Complete Phase 0 first: create manifests, split recordings, "
        "and run verify_dataset.py."
    )


if __name__ == "__main__":
    main()
