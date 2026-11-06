# Metrik evaluasi untuk model keluarga AE.

# Semua metrik dihitung pada citra di rentang [0, 1] (data_range = 1.0).

# - MSE : rata-rata kuadrat selisih per piksel.
# - PSNR: 10 * log10(1 / MSE) per citra, lalu dirata-ratakan antar citra.
# - SSIM: skimage.structural_similarity per citra (jendela 7x7 default), dirata-ratakan.
# - FID : belum di sini, dihitung terpisah untuk VAE, GAN, DCGAN, DDPM.


import time

import numpy as np
import torch
from skimage.metrics import structural_similarity


def per_image_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return ((x - x_hat) ** 2).flatten(1).mean(dim=1)


def per_image_psnr(mse: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / (mse + eps))


def per_image_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    a = x.squeeze(1).detach().cpu().numpy()
    b = x_hat.squeeze(1).detach().cpu().numpy()
    return np.array(
        [structural_similarity(a[i], b[i], data_range=1.0) for i in range(len(a))]
    )


@torch.no_grad()
def reconstruction_metrics(model, loader, device) -> dict:
    """MSE, PSNR, SSIM rata-rata di seluruh loader (pakai test loader)."""
    model.eval()
    mses, psnrs, ssims = [], [], []
    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        mse = per_image_mse(x, x_hat)
        mses.append(mse.cpu())
        psnrs.append(per_image_psnr(mse).cpu())
        ssims.append(torch.from_numpy(per_image_ssim(x, x_hat)))
    return {
        "mse": torch.cat(mses).mean().item(),
        "psnr": torch.cat(psnrs).mean().item(),
        "ssim": torch.cat(ssims).mean().item(),
    }


@torch.no_grad()
def active_unit_fraction(model, loader, device) -> float:
    """Fraksi unit aktif (h > 0) pada hidden layer 256 encoder, rata-rata di loader."""
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        h = model.encoder[:3](x)  # Flatten -> Linear(784, 256) -> ReLU
        total += (h > 0).float().mean().item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def measure_latency(fn, device, n_warmup: int = 20, n_runs: int = 200) -> float:
    """Rata-rata waktu (ms) satu pemanggilan fn(). fn harus sudah memuat input batch 64."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0