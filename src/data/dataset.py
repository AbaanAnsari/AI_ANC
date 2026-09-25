from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object
    TorchDataLoader = None

from src.data.audio_loader import load_audio
from src.data.dual_mic import simulate_dual_mic
from src.data.mixer import DEFAULT_SNR_SET, calculate_snr_db, mix_signals
from src.data.noise_selector import NOISE_CLASSES, NoiseSelector
from src.data.segmenter import segment_audio
from src.dsp.gcc_phat import gcc_phat
from src.dsp.kalman import DelayKalmanFilter
from src.features.stft import compute_stft

# Locked Noise Class Mapping
NOISE_CLASS_TO_LABEL = {
    "stationary": 0,
    "non-stationary": 1,
    "impulsive": 2,
}

LABEL_TO_NOISE_CLASS = {
    0: "stationary",
    1: "non-stationary",
    2: "impulsive",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


class Phase1Dataset(TorchDataset):
    """
    On-the-fly Supervised Training Sample Generator for Phase 1.

    Generates training samples containing:
      - m1_waveform (speech + N1)
      - m2_waveform (correlated reference N2)
      - clean_target (pure speech target, float32)
      - noisy_stft (complex STFT of M1, complex64)
      - noisy_features (2, 257, T) float32 [channel 0 = real, channel 1 = imag]
      - target_stft (complex STFT of clean target, complex64)
      - noise_class & noise_class_label (0: stationary, 1: non-stationary, 2: impulsive)
      - metadata (SNR, GCC-PHAT delay, Kalman filtered delay, file paths)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path | None = None,
        snr_set: Sequence[float] = DEFAULT_SNR_SET,
        segment_samples: int = 16000,
        return_tensors: bool = True,
        seed: int = 20260925,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.dataset_root = (
            Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
        )
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.snr_set = tuple(float(x) for x in snr_set)
        self.segment_samples = segment_samples
        self.return_tensors = return_tensors and HAS_TORCH
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Load manifest records
        content = self.manifest_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Manifest file is empty: {self.manifest_path}")

        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]

        self.clean_records = [
            r for r in records if r.get("source_group") == "clean"
        ]
        if not self.clean_records:
            raise ValueError(
                f"No clean speech records found in manifest: {self.manifest_path}"
            )

        self.noise_selector = NoiseSelector(self.manifest_path, rng=self.rng)

    def __len__(self) -> int:
        return len(self.clean_records)

    def __getitem__(self, index: int) -> dict:
        """
        PyTorch-compatible dataset indexing.
        Generates deterministic sample for given index using index seed.
        """
        sample_rng = np.random.default_rng(self.seed + index)
        clean_record = self.clean_records[index % len(self.clean_records)]
        return self.generate_sample(clean_record=clean_record, rng=sample_rng)

    def generate_sample(
        self,
        clean_record: dict | None = None,
        noise_class: str | None = None,
        target_snr_db: float | None = None,
        relative_delay_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Generate a single supervised training sample on-the-fly.
        """
        if rng is None:
            rng = self.rng

        if clean_record is None:
            c_idx = int(rng.integers(0, len(self.clean_records)))
            clean_record = self.clean_records[c_idx]

        if noise_class is None:
            n_cls_idx = int(rng.integers(0, len(NOISE_CLASSES)))
            noise_class = NOISE_CLASSES[n_cls_idx]

        if noise_class not in NOISE_CLASS_TO_LABEL:
            raise ValueError(
                f"Unsupported noise_class: {noise_class!r}. Must be one of {tuple(NOISE_CLASS_TO_LABEL.keys())}"
            )

        noise_record = self.noise_selector.select(noise_class, rng=rng)

        if target_snr_db is None:
            snr_idx = int(rng.integers(0, len(self.snr_set)))
            target_snr_db = self.snr_set[snr_idx]

        if relative_delay_samples is None:
            relative_delay_samples = int(rng.integers(-20, 21))

        # 1. Load Audio
        clean_path = self.dataset_root / clean_record["source_path"]
        noise_path = self.dataset_root / noise_record["source_path"]

        clean_raw = load_audio(clean_path)
        noise_raw = load_audio(noise_path)

        # 2. Segment Audio
        clean_seg = segment_audio(
            clean_raw, segment_samples=self.segment_samples, rng=rng
        )
        noise_seg = segment_audio(
            noise_raw, segment_samples=self.segment_samples, rng=rng
        )

        # 3. Mix Speech & Noise at Target SNR
        noisy_speech, scaled_noise = mix_signals(
            clean_seg, noise_seg, target_snr_db=target_snr_db, rng=rng
        )
        measured_snr = calculate_snr_db(clean_seg, scaled_noise)

        # 4. Dual Microphone Simulation
        m1, m2 = simulate_dual_mic(
            clean_speech=clean_seg,
            scaled_noise=scaled_noise,
            relative_delay_samples=relative_delay_samples,
            gain_mismatch=1.0,
            filter_coefficients=(0.9, 0.1),
            rng=rng,
        )

        # 5. GCC-PHAT Delay Estimation
        gcc_res = gcc_phat(m1, m2, sample_rate=16000, max_tau=30)
        gcc_delay = int(gcc_res["estimated_delay_samples"])

        # 6. Kalman Filtered Delay
        kf = DelayKalmanFilter(initial_delay=0.0)
        kalman_delay = float(kf.update(gcc_delay))

        # 7. STFT Feature Extraction
        noisy_stft = compute_stft(m1)  # (257, T) complex64
        target_stft = compute_stft(clean_seg)  # (257, T) complex64

        # Model-ready features: (2, 257, T) float32 [Ch0=Real, Ch1=Imag]
        noisy_features = np.stack(
            [noisy_stft.real, noisy_stft.imag], axis=0
        ).astype(np.float32)

        noise_label = NOISE_CLASS_TO_LABEL[noise_class]

        metadata = {
            "clean_source_path": clean_record["source_path"],
            "clean_record_id": clean_record["record_id"],
            "noise_source_path": noise_record["source_path"],
            "noise_record_id": noise_record["record_id"],
            "noise_class": noise_class,
            "noise_subclass": noise_record.get("noise_subclass"),
            "target_snr_db": float(target_snr_db),
            "measured_snr_db": float(measured_snr),
            "configured_delay_samples": int(relative_delay_samples),
            "gcc_delay_samples": gcc_delay,
            "kalman_delay_samples": kalman_delay,
        }

        if self.return_tensors and HAS_TORCH:
            return {
                "m1_waveform": torch.from_numpy(m1),
                "m2_waveform": torch.from_numpy(m2),
                "clean_target": torch.from_numpy(clean_seg),
                "noisy_stft": torch.from_numpy(noisy_stft),
                "noisy_features": torch.from_numpy(noisy_features),
                "target_stft": torch.from_numpy(target_stft),
                "noise_class": noise_class,
                "noise_class_label": torch.tensor(noise_label, dtype=torch.int64),
                "noise_subclass": noise_record.get("noise_subclass"),
                "target_snr_db": torch.tensor(target_snr_db, dtype=torch.float32),
                "measured_snr_db": torch.tensor(measured_snr, dtype=torch.float32),
                "configured_delay_samples": torch.tensor(
                    relative_delay_samples, dtype=torch.int64
                ),
                "gcc_delay_samples": torch.tensor(gcc_delay, dtype=torch.int64),
                "kalman_delay_samples": torch.tensor(
                    kalman_delay, dtype=torch.float32
                ),
                "metadata": metadata,
            }

        return {
            "m1_waveform": m1,
            "m2_waveform": m2,
            "clean_target": clean_seg,
            "noisy_stft": noisy_stft,
            "noisy_features": noisy_features,
            "target_stft": target_stft,
            "noise_class": noise_class,
            "noise_class_label": np.int64(noise_label),
            "noise_subclass": noise_record.get("noise_subclass"),
            "target_snr_db": float(target_snr_db),
            "measured_snr_db": float(measured_snr),
            "configured_delay_samples": int(relative_delay_samples),
            "gcc_delay_samples": gcc_delay,
            "kalman_delay_samples": kalman_delay,
            "metadata": metadata,
        }


def custom_collate_fn(batch_list: list[dict]) -> dict:
    """
    Collate a list of sample dictionaries into a batch dictionary.
    Supports PyTorch tensors or NumPy arrays.
    """
    first = batch_list[0]
    is_tensor = HAS_TORCH and isinstance(first["clean_target"], torch.Tensor)

    if is_tensor:
        noisy_features = torch.stack([s["noisy_features"] for s in batch_list], dim=0)
        target_stft = torch.stack([s["target_stft"] for s in batch_list], dim=0)
        clean_target = torch.stack([s["clean_target"] for s in batch_list], dim=0)
        noise_class_label = torch.stack([s["noise_class_label"] for s in batch_list], dim=0)
        target_snr_db = torch.stack([s["target_snr_db"] for s in batch_list], dim=0)
        measured_snr_db = torch.stack([s["measured_snr_db"] for s in batch_list], dim=0)
        gcc_delay_samples = torch.stack([s["gcc_delay_samples"] for s in batch_list], dim=0)
        kalman_delay_samples = torch.stack([s["kalman_delay_samples"] for s in batch_list], dim=0)
    else:
        noisy_features = np.stack([s["noisy_features"] for s in batch_list], axis=0)
        target_stft = np.stack([s["target_stft"] for s in batch_list], axis=0)
        clean_target = np.stack([s["clean_target"] for s in batch_list], axis=0)
        noise_class_label = np.array([s["noise_class_label"] for s in batch_list], dtype=np.int64)
        target_snr_db = np.array([s["target_snr_db"] for s in batch_list], dtype=np.float32)
        measured_snr_db = np.array([s["measured_snr_db"] for s in batch_list], dtype=np.float32)
        gcc_delay_samples = np.array([s["gcc_delay_samples"] for s in batch_list], dtype=np.int64)
        kalman_delay_samples = np.array([s["kalman_delay_samples"] for s in batch_list], dtype=np.float32)

    return {
        "noisy_features": noisy_features,
        "target_stft": target_stft,
        "clean_target": clean_target,
        "noise_class_label": noise_class_label,
        "target_snr_db": target_snr_db,
        "measured_snr_db": measured_snr_db,
        "gcc_delay_samples": gcc_delay_samples,
        "kalman_delay_samples": kalman_delay_samples,
        "metadata": [s["metadata"] for s in batch_list],
    }


def create_dataloader(
    manifest_path: str | Path,
    dataset_root: str | Path | None = None,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    return_tensors: bool = True,
) -> TorchDataLoader | list[dict]:
    """
    Create a batched DataLoader for training, validation, or testing.
    """
    dataset = Phase1Dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        return_tensors=return_tensors,
        seed=seed,
    )

    if HAS_TORCH and return_tensors:
        generator = torch.Generator().manual_seed(seed)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=custom_collate_fn,
            generator=generator,
        )

    # Fallback generator for non-PyTorch environments
    indices = list(range(len(dataset)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_samples = [dataset[idx] for idx in batch_indices]
        batches.append(custom_collate_fn(batch_samples))
    return batches
