# Vanilla GAN (MLP)

# Generator    : z(100) -> 256 -> 512 (BatchNorm) -> 784, Tanh  => citra di [-1, 1]
# Discriminator: 784 -> 512 -> 256 -> 1, LeakyReLU(0.2), Dropout(0.3), Sigmoid

# Catatan rentang piksel: GAN bekerja di [-1, 1] (Tanh). Gunakan data.to_minus1_1() untuk data nyata dan data.to_zero1() untuk menampilkan atau menghitung metrik.

# Loss (dihitung di train_gan.py):
# L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
# L_G = -E[log D(G(z))]          (versi non-saturating)


import torch
import torch.nn as nn


class VanillaGenerator(nn.Module):
    def __init__(self, z_dim: int = 100, img_dim: int = 784):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """n citra dari z ~ N(0, I), di rentang [-1, 1]. Panggil saat model.eval()."""
        return self(torch.randn(n, self.z_dim, device=device))


class VanillaDiscriminator(nn.Module):
    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))