"""DCGAN untuk citra 28x28 (Radford et al., 2016).

Prinsip DCGAN yang diterapkan:
  1. All-convolutional: tanpa pooling dan tanpa fully-connected di tengah.
     Down-sampling dengan strided conv (D), up-sampling dengan transposed conv (G).
  2. BatchNorm di hidden layer, TIDAK di output G dan TIDAK di input D.
  3. Aktivasi: ReLU di G, Tanh di output G; LeakyReLU(0.2) di D.
  4. Inisialisasi bobot N(0, 0.02).

Generator    : z(100,1,1) -> ConvT 7x7 (256) -> ConvT 14x14 (128) -> ConvT 28x28 (1), Tanh
Discriminator: 28x28 -> Conv 14x14 (64) -> Conv 7x7 (128) -> Conv 1x1 (1), Sigmoid

Antarmuka sama dengan VanillaGenerator/VanillaDiscriminator, sehingga
train_gan.py bisa dipakai tanpa perubahan loop.
"""

import torch
import torch.nn as nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),  # 7x7
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(-1, self.z_dim, 1, 1))

    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """n citra dari z ~ N(0, I), di rentang [-1, 1]. Panggil saat model.eval()."""
        return self(torch.randn(n, self.z_dim, device=device))


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 14x14, tanpa BN di input
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0),  # 1x1
            nn.Sigmoid(),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)