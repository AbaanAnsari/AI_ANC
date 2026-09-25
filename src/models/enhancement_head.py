from __future__ import annotations

import torch
import torch.nn as nn


class EnhancementHead(nn.Module):
    """
    Supervised Speech Enhancement Head.

    Input:
        GRU output features of shape (B, T, hidden_dim)

    Output:
        Predicted enhanced STFT features of shape (B, 2, n_freq_bins, T)
        where channel 0 = Real, channel 1 = Imaginary.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        n_freq_bins: int = 257,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freq_bins = n_freq_bins
        self.out_channels = out_channels

        # Single linear layer per frame to project hidden_dim -> (out_channels * n_freq_bins)
        self.proj = nn.Linear(hidden_dim, out_channels * n_freq_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            GRU features of shape (B, T, hidden_dim).

        Returns
        -------
        torch.Tensor
            Enhanced STFT features of shape (B, 2, n_freq_bins, T).
        """
        batch_size, time_frames, _ = x.shape

        # (B, T, hidden_dim) -> (B, T, out_channels * n_freq_bins)
        out = self.proj(x)

        # Reshape to (B, T, out_channels, n_freq_bins)
        out = out.view(batch_size, time_frames, self.out_channels, self.n_freq_bins)

        # Permute to (B, out_channels, n_freq_bins, T)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out
