"""Latency inferensi per batch 64 sampel untuk keenam model, di SATU perangkat yang sama.

Latency tidak bergantung pada nilai bobot, jadi model dibuat dengan bobot acak (tanpa
checkpoint). Operasi yang diukur:
    Vanilla AE, Sparse AE : rekonstruksi (forward encoder + decoder)
    VAE                   : rekonstruksi dan sampling dari prior (decode z)
    Vanilla GAN, DCGAN    : generate 64 citra dari z ~ N(0, I)  (satu langkah)
    DDPM                  : generate 64 citra lewat T = 1000 langkah denoising

Contoh:
    python src/latency.py
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_device, set_seed  # noqa: E402
from metrics import measure_latency  # noqa: E402
from models.dcgan import DCGANGenerator  # noqa: E402
from models.ddpm import GaussianDiffusion, UNet  # noqa: E402
from models.sparse_ae import SparseAE  # noqa: E402
from models.vae import VAE  # noqa: E402
from models.vanilla_ae import VanillaAE  # noqa: E402
from models.vanilla_gan import VanillaGenerator  # noqa: E402

BATCH = 64


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_runs", type=int, default=1, help="jumlah pengulangan sampling DDPM penuh")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {name}")

    x = torch.rand(BATCH, 1, 28, 28, device=device)
    z64 = torch.randn(BATCH, 64, device=device)
    z100 = torch.randn(BATCH, 100, device=device)

    ae = VanillaAE(64).to(device).eval()
    sparse = SparseAE(64).to(device).eval()
    vae = VAE(64).to(device).eval()
    gan_g = VanillaGenerator(100).to(device).eval()
    dcgan_g = DCGANGenerator(100).to(device).eval()
    unet = UNet(64).to(device).eval()
    diffusion = GaussianDiffusion(1000).to(device)

    # pemanasan U-Net (cuDNN autotune, alokasi memori) sebelum sampling penuh diukur
    t0 = torch.zeros(BATCH, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(3):
            unet(x, t0)

    tests = [
        ("Vanilla AE", "rekonstruksi", lambda: ae(x), 20, 200),
        ("Sparse AE", "rekonstruksi", lambda: sparse(x), 20, 200),
        ("VAE", "rekonstruksi", lambda: vae(x), 20, 200),
        ("VAE", "sampling prior", lambda: vae.decode(z64), 20, 200),
        ("Vanilla GAN", "generate", lambda: gan_g(z100), 20, 200),
        ("DCGAN", "generate", lambda: dcgan_g(z100), 20, 200),
        ("DDPM", "generate (T=1000)", lambda: diffusion.sample(unet, BATCH, device), 0, args.ddpm_runs),
    ]

    results = {}
    print("\n| Model | Operasi | ms / batch 64 |")
    print("|---|---|---|")
    for model_name, op, fn, n_warmup, n_runs in tests:
        ms = measure_latency(fn, device, n_warmup=n_warmup, n_runs=n_runs)
        results[f"{model_name} ({op})"] = {"latency_ms_b64": ms, "n_runs": n_runs}
        print(f"| {model_name} | {op} | {ms:.3f} |")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "latency_results.json")
    with open(path, "w") as f:
        json.dump({"device": name, "batch": BATCH, "results": results}, f, indent=2)
    print(f"\ntersimpan: {path}")


if __name__ == "__main__":
    main()