from __future__ import annotations

import json
from pathlib import Path

import numpy as np


NOISE_CLASSES = (
    "stationary",
    "non-stationary",
    "impulsive",
)


class NoiseSelector:
    """
    Select noise records from a dataset manifest by noise class.

    The selector only chooses dataset records.
    It does not load, modify, mix, or generate audio.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        rng: np.random.Generator | None = None,
    ) -> None:

        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        content = self.manifest_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(
                f"Manifest file is empty: {self.manifest_path}"
            )

        if content.startswith("["):
            manifest = json.loads(content)
        else:
            manifest = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]

        if not isinstance(manifest, list):
            raise ValueError(
                "Manifest must contain a JSON list or JSONL records."
            )

        self.records = manifest
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

        self._records_by_class = {
            noise_class: []
            for noise_class in NOISE_CLASSES
        }

        for record in self.records:

            if record.get("source_group") != "noise":
                continue

            noise_class = record.get("noise_class")

            if noise_class in self._records_by_class:
                self._records_by_class[noise_class].append(
                    record
                )

        for noise_class in NOISE_CLASSES:

            if not self._records_by_class[noise_class]:
                raise ValueError(
                    f"No records found for noise class: "
                    f"{noise_class}"
                )

    def available_classes(self) -> tuple[str, ...]:
        """Return the supported noise classes."""

        return NOISE_CLASSES

    def count(self, noise_class: str) -> int:
        """Return the number of records available for a class."""

        self._validate_class(noise_class)

        return len(
            self._records_by_class[noise_class]
        )

    def select(
        self,
        noise_class: str,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Randomly select one noise record from a class.
        """

        self._validate_class(noise_class)

        records = self._records_by_class[noise_class]
        generator = rng if rng is not None else self.rng

        index = int(
            generator.integers(
                0,
                len(records),
            )
        )

        return records[index].copy()

    def select_many(
        self,
        noise_class: str,
        count: int,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """
        Randomly select multiple noise records.

        Selection is with replacement, which allows the same
        source file to be used in different generated mixtures.
        """

        self._validate_class(noise_class)

        if count <= 0:
            raise ValueError(
                f"count must be > 0, got {count}"
            )

        records = self._records_by_class[noise_class]
        generator = rng if rng is not None else self.rng

        indices = generator.integers(
            0,
            len(records),
            size=count,
        )

        return [
            records[int(index)].copy()
            for index in indices
        ]

    @staticmethod
    def _validate_class(noise_class: str) -> None:

        if noise_class not in NOISE_CLASSES:
            raise ValueError(
                f"Unsupported noise class: {noise_class!r}. "
                f"Expected one of: {NOISE_CLASSES}"
            )