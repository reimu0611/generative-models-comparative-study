import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAE(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid(),  # keluaran di [0, 1], sama dengan rentang data
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def compute_loss(self, x: torch.Tensor):
        """Return (loss, dict log). Antarmuka ini dipakai train.py."""
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        return loss, {"loss": loss.item(), "recon": loss.item()}