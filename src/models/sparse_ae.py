# Sparse Autoencoder (MLP, penalti L1 pada hidden layer 256).
# Arsitektur identik dengan VanillaAE (784 -> 256 -> 64 -> 256 -> 784), jadi satu-satunya perbedaan adalah penalti sparsity. Perbandingan keduanya adil.
# Loss: L = MSE(x, x_hat) + lambda * mean(|h|),  h = aktivasi hidden 256 (setelah ReLU)

import torch
import torch.nn.functional as F
from models.vanilla_ae import VanillaAE

class SparseAE(VanillaAE):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, l1_weight: float = 1e-3):
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.l1_weight = l1_weight

    def encode_with_hidden(self, x: torch.Tensor):
        """Return (z, h). h = aktivasi layer 256 yang dikenai penalti."""
        h = self.encoder[:3](x)  # Flatten -> Linear(784,256) -> ReLU
        z = self.encoder[3](h)   # Linear(256, latent)
        return z, h

    def compute_loss(self, x: torch.Tensor):
        z, h = self.encode_with_hidden(x)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()
        loss = recon + self.l1_weight * l1
        active = (h > 0).float().mean()  # fraksi unit aktif (untuk analisis sparsity)
        return loss, {
            "loss": loss.item(),
            "recon": recon.item(),
            "l1": l1.item(),
            "active": active.item(),
        }