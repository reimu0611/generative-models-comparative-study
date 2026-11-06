# Variational Autoencoder (MLP).

# Arsitektur: 784 -> 256 -> (mu, logvar) dengan dimensi latent_dim
# latent_dim -> 256 -> 784 (sigmoid)

# Loss (negatif ELBO, dijumlah per citra lalu dirata-ratakan per batch):
# L = BCE(x, x_hat) + beta * KL( q(z|x) || N(0, I) )
# KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

# Reparameterization trick: z = mu + sigma * eps,  eps ~ N(0, I).

# API disamakan dengan VanillaAE supaya skrip evaluasi bisa memakai keduanya:
# model.encode(x)  -> mu (deterministik)
# model.decode(z)  -> citra [B, 1, 28, 28]
# model(x)         -> rekonstruksi deterministik (memakai mu)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid(),
        )

    def encode_dist(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Representasi deterministik = mean posterior."""
        return self.encode_dist(x)[0]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """Sampling dari prior N(0, I) lalu decode."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def compute_loss(self, x: torch.Tensor):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        batch = x.size(0)
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
        loss = recon + self.beta * kl

        mse = F.mse_loss(x_hat, x)  # hanya untuk dibandingkan dengan AE
        return loss, {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "mse": mse.item(),
        }