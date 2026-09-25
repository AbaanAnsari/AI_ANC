from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classification_head import ClassificationHead
from src.models.enhancement_head import EnhancementHead
from src.models.model_utils import count_parameters, get_model_memory_bytes


class LightweightCNNTGRUModel(nn.Module):
    """
    Lightweight CNN + GRU Multi-Task Speech Enhancement & Classification Model.

    Inputs:
        noisy_features: Tensor of shape (B, 2, 257, T)
        [Channel 0 = Real, Channel 1 = Imaginary STFT]

    Outputs:
        enhanced_stft: Tensor of shape (B, 2, 257, T)
        classification_logits: Tensor of shape (B, 3)

    Hard Constraint:
        Total Trainable Parameters MUST be strictly < 100,000.
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_freq_bins: int = 257,
        cnn_channels: tuple[int, ...] = (16, 32, 16),
        proj_dim: int = 32,
        gru_hidden_dim: int = 48,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_freq_bins = n_freq_bins

        # 1. Lightweight 2D CNN Feature Extractor
        # Frequency dimension reduction: 257 -> 129 -> 65 -> 33
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                cnn_channels[0],
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
            ),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                cnn_channels[0],
                cnn_channels[1],
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
            ),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                cnn_channels[1],
                cnn_channels[2],
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
            ),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
        )

        # Spatial dimension after 3 strides of 2: 33 frequency bins
        self.reduced_freq_bins = 33
        flattened_cnn_dim = cnn_channels[2] * self.reduced_freq_bins

        # 2. Linear Feature Projection before GRU
        self.proj = nn.Sequential(
            nn.Linear(flattened_cnn_dim, proj_dim),
            nn.ReLU(),
        )

        # 3. Lightweight GRU for Temporal Sequential Modeling
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 4. Multi-Task Heads
        self.enhancement_head = EnhancementHead(
            hidden_dim=gru_hidden_dim,
            n_freq_bins=n_freq_bins,
            out_channels=2,
        )

        self.classification_head = ClassificationHead(
            hidden_dim=gru_hidden_dim,
            num_classes=num_classes,
            bottleneck_dim=16,
        )

        # Enforce hard parameter constraint at initialization
        total_trainable_params = count_parameters(self, trainable_only=True)
        if total_trainable_params >= 100000:
            raise ValueError(
                f"HARD CONSTRAINT VIOLATED: Total trainable parameters {total_trainable_params} "
                "exceeds the maximum budget of 100,000 parameters."
            )

    def forward(
        self, noisy_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        noisy_features : torch.Tensor
            Input complex STFT features of shape (B, 2, 257, T).

        Returns
        -------
        enhanced_stft : torch.Tensor
            Predicted complex STFT features of shape (B, 2, 257, T).
        classification_logits : torch.Tensor
            Noise class prediction logits of shape (B, 3).
        """
        batch_size, channels, freq_bins, time_frames = noisy_features.shape

        # 1. CNN Feature Extraction
        x = self.conv1(noisy_features)  # (B, 16, 129, T)
        x = self.conv2(x)               # (B, 32, 65, T)
        x = self.conv3(x)               # (B, 16, 33, T)

        # 2. Reshape for sequential processing: (B, C, F_red, T) -> (B, T, C * F_red)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_frames, -1)  # (B, T, 528)

        # 3. Projection to GRU input size
        x_proj = self.proj(x)  # (B, T, 32)

        # 4. GRU Temporal Tracking
        gru_out, _ = self.gru(x_proj)  # (B, T, 48)

        # 5. Multi-Task Task Heads
        enhanced_stft = self.enhancement_head(gru_out)      # (B, 2, 257, T)
        classification_logits = self.classification_head(gru_out)  # (B, 3)

        return enhanced_stft, classification_logits

    def parameter_summary(self) -> dict:
        """Return parameter count summary and memory footprint."""
        total_params = count_parameters(self, trainable_only=False)
        trainable_params = count_parameters(self, trainable_only=True)
        mem_bytes = get_model_memory_bytes(self)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "fp32_memory_bytes": mem_bytes,
            "fp32_memory_kb": round(mem_bytes / 1024.0, 2),
        }
