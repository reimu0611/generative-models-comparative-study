"""DDPM (Ho et al., 2020) untuk citra 28x28.

Data bekerja di rentang [-1, 1] (pakai data.to_minus1_1 / to_zero1).
Indeks t di kode ini 0-based: t = 0 .. T-1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Embedding posisi sinusoidal untuk timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, dengan embedding waktu ditambahkan di tengah."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net kecil: 28x28 -> 14x14 -> 7x7 -> kembali ke 28x28, dengan skip connection."""

    def __init__(self, base: int = 64, t_dim: int = 256):
        super().__init__()
        b = base
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(b), nn.Linear(b, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.conv_in = nn.Conv2d(1, b, 3, padding=1)

        # encoder
        self.down1 = ResBlock(b, b, t_dim)  # 28x28
        self.ds1 = nn.Conv2d(b, b, 3, stride=2, padding=1)  # -> 14x14
        self.down2 = ResBlock(b, 2 * b, t_dim)
        self.ds2 = nn.Conv2d(2 * b, 2 * b, 3, stride=2, padding=1)  # -> 7x7
        self.down3 = ResBlock(2 * b, 2 * b, t_dim)

        # bottleneck
        self.mid1 = ResBlock(2 * b, 2 * b, t_dim)
        self.mid2 = ResBlock(2 * b, 2 * b, t_dim)

        # decoder (input = hasil upsample digabung skip connection)
        self.up3 = ResBlock(4 * b, 2 * b, t_dim)  # 7x7
        self.us2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(2 * b, 2 * b, 3, padding=1))  # -> 14x14
        self.up2 = ResBlock(4 * b, 2 * b, t_dim)
        self.us1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(2 * b, 2 * b, 3, padding=1))  # -> 28x28
        self.up1 = ResBlock(3 * b, b, t_dim)

        self.out = nn.Sequential(nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)
        h = self.conv_in(x)
        s1 = self.down1(h, te)                      # (b, 28, 28)
        s2 = self.down2(self.ds1(s1), te)           # (2b, 14, 14)
        s3 = self.down3(self.ds2(s2), te)           # (2b, 7, 7)

        h = self.mid2(self.mid1(s3, te), te)        # (2b, 7, 7)

        h = self.up3(torch.cat([h, s3], dim=1), te)   # (2b, 7, 7)
        h = self.up2(torch.cat([self.us2(h), s2], dim=1), te)  # (2b, 14, 14)
        h = self.up1(torch.cat([self.us1(h), s1], dim=1), te)  # (b, 28, 28)
        return self.out(h)


class GaussianDiffusion(nn.Module):
    """Jadwal noise, forward process, loss, dan sampling reverse. Tidak punya parameter belajar."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Loncat langsung ke x_t dari x_0 (closed-form)."""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """L_simple: MSE antara noise asli dan noise prediksi."""
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, device, save_steps=None):
        """Ancestral sampling x_T ~ N(0, I) -> x_0, T langkah iteratif.

        save_steps: daftar indeks t yang ingin direkam (untuk Fig 5). Snapshot diambil
        SEBELUM langkah denoising di indeks t, jadi x awal (noise murni) = indeks T-1.
        Kalau save_steps diberikan, return (x_0, [(t, x_t), ...]); kalau tidak, return x_0.
        """
        model.eval()
        keep = set(save_steps) if save_steps is not None else set()
        snapshots = []
        x = torch.randn(n, 1, 28, 28, device=device)
        for t in reversed(range(self.T)):
            if t in keep:
                snapshots.append((t, x.clone()))
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            ab_t = self.alpha_bar[t]
            mean = (x - beta_t / (1.0 - ab_t).sqrt() * eps) / alpha_t.sqrt()
            x = mean + beta_t.sqrt() * torch.randn_like(x) if t > 0 else mean
        x = x.clamp(-1.0, 1.0)
        return (x, snapshots) if save_steps is not None else x