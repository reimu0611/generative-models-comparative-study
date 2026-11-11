"""FID (Frechet Inception Distance) untuk model generatif: VAE, Vanilla GAN, DCGAN, DDPM.

Referensi real = seluruh test set (7.000 citra). Tiap model menghasilkan n_samples citra
dengan jumlah yang sama, supaya perbandingan antarmodel sepadan. Citra grayscale 28x28
diubah ke 3 kanal dan di-resize ke 299x299 oleh Inception (dilakukan torchmetrics).

Catatan: FID dengan sampel sedikit bias ke atas. Nilainya hanya sebanding antarmodel
yang memakai n_samples dan referensi real yang sama.

Contoh:
    python src/fid.py                       # keempat model
    python src/fid.py --models ddpm         # satu model; hasil digabung ke fid_results.json
"""

import argparse
import json
import os
import sys

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed, to_zero1  # noqa: E402
from models.dcgan import DCGANGenerator  # noqa: E402
from models.ddpm import GaussianDiffusion, UNet  # noqa: E402
from models.vae import VAE  # noqa: E402
from models.vanilla_gan import VanillaGenerator  # noqa: E402

ALL_MODELS = ["vae", "gan", "dcgan", "ddpm"]


def load_sampler(name, ckpt_dir, device):
    """Return fungsi sample(n) -> citra [n, 1, 28, 28] di rentang [0, 1]."""

    def load(model, fname):
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, fname), map_location=device))
        return model.to(device).eval()

    if name == "vae":
        m = load(VAE(64), "vae_best.pt")
        return lambda n: m.sample(n, device)  # decoder VAE sudah di [0, 1]
    if name == "gan":
        g = load(VanillaGenerator(100), "gan_G_last.pt")
        return lambda n: to_zero1(g.sample(n, device))  # generator di [-1, 1]
    if name == "dcgan":
        g = load(DCGANGenerator(100), "dcgan_G_last.pt")
        return lambda n: to_zero1(g.sample(n, device))
    if name == "ddpm":
        unet = load(UNet(64), "ddpm_ema_last.pt")
        diffusion = GaussianDiffusion(1000).to(device)
        return lambda n: to_zero1(diffusion.sample(unet, n, device))
    raise ValueError(name)


def generate(sampler, n, batch):
    chunks, done = [], 0
    while done < n:
        b = min(batch, n - done)
        chunks.append(sampler(b).clamp(0.0, 1.0).cpu())
        done += b
    return torch.cat(chunks)


def to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--gen_batch", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    _, _, test_loader = get_dataloaders(128, args.data_dir)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(device)

    n_real = 0
    for x, _ in test_loader:
        fid.update(to_rgb(x).to(device), real=True)
        n_real += x.size(0)
    print(f"fitur real dihitung dari {n_real} citra test set")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "fid_results.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results["n_real"] = n_real

    for name in args.models:
        sampler = load_sampler(name, args.ckpt_dir, device)
        torch.manual_seed(args.seed)  # seed sama untuk tiap model, independen dari urutan
        imgs = generate(sampler, args.n_samples, args.gen_batch)
        for i in range(0, len(imgs), 250):
            fid.update(to_rgb(imgs[i:i + 250]).to(device), real=False)
        score = fid.compute().item()
        fid.reset()  # hanya fitur fake yang dihapus, fitur real dipertahankan
        results[name] = {"fid": score, "n_samples": args.n_samples}
        print(f"FID {name}: {score:.2f} (n_samples={args.n_samples})")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"tersimpan: {path}")


if __name__ == "__main__":
    main()