"""Per-user localization model for geometry-consistent wireless channels."""
import torch
import torch.nn as nn


class PerUserLocalizationModel(nn.Module):
    """Predict one (x, y, z) triplet per user from that user's channel signature.

    Inputs follow task-4 layout: 16 received-symbol features followed by the
    real and imaginary channel coefficients.  Keeping users on a separate
    axis prevents the global pooled-head collapse to the scene mean.
    """
    def __init__(self, num_users: int = 8, num_antennas: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.channel_encoder = nn.Sequential(
            nn.Linear(num_antennas * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.position_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_subcarriers, _ = inputs.shape
        start = 16
        split = start + self.num_users * self.num_antennas
        h_real = inputs[:, :, start:split].reshape(
            batch_size, num_subcarriers, self.num_users, self.num_antennas
        )
        h_imag = inputs[:, :, split:split + self.num_users * self.num_antennas].reshape(
            batch_size, num_subcarriers, self.num_users, self.num_antennas
        )
        # Average frequency fading while retaining each user's UPA signature.
        channel_features = torch.cat([h_real.mean(dim=1), h_imag.mean(dim=1)], dim=-1)
        return torch.sigmoid(self.position_head(self.channel_encoder(channel_features)))
