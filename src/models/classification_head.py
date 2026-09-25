from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Supervised 3-Class Noise Classification Head.

    Input:
        GRU output features of shape (B, T, hidden_dim)

    Output:
        Classification logits of shape (B, num_classes)
        Classes: 0 = stationary, 1 = non-stationary, 2 = impulsive.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        num_classes: int = 3,
        bottleneck_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            GRU features of shape (B, T, hidden_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 3).
        """
        # Temporal average pooling over time frames: (B, T, hidden_dim) -> (B, hidden_dim)
        pooled = torch.mean(x, dim=1)

        # Logits: (B, hidden_dim) -> (B, num_classes)
        logits = self.classifier(pooled)
        return logits
