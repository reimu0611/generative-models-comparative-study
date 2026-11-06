# Evaluasi Vanilla AE, Sparse AE, dan VAE pada test set.

# Menghasilkan tabel MSE / PSNR / SSIM / unit aktif / waktu per epoch / latency, disimpan ke outputs/results_ae_family.json.

# Contoh:
# python src/evaluate.py
# python src/evaluate.py --sparse_tag sparse_l1e-2

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from metrics import (  # noqa: E402
    active_unit_fraction,
    measure_latency,
    reconstruction_metrics,
)
from models.sparse_ae import SparseAE  # noqa: E402
from models.vae import VAE  # noqa: E402
from models.vanilla_ae import VanillaAE  # noqa: E402

LATENT = 64


def load_models(ckpt_dir, sparse_tag, device, ae_tag="ae", vae_tag="vae"):
    """Return dict {nama: (model, tag)} dengan bobot terbaik dimuat dari checkpoint."""
    specs = {
        "Vanilla AE": (VanillaAE(LATENT), ae_tag),
        "Sparse AE": (SparseAE(LATENT), sparse_tag),
        "VAE": (VAE(LATENT), vae_tag),
    }
    models = {}
    for name, (model, tag) in specs.items():
        path = os.path.join(ckpt_dir, f"{tag}_best.pt")
        model.load_state_dict(torch.load(path, map_location=device))
        models[name] = (model.to(device).eval(), tag)
    return models


def mean_epoch_time(out_dir, tag):
    path = os.path.join(out_dir, f"{tag}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        times = json.load(f)["epoch_time"]
    return sum(times) / len(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sparse_tag", default="sparse_l1e-2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    print(f"device: {device}")
    _, _, test_loader = get_dataloaders(args.batch_size, args.data_dir)
    models = load_models(args.ckpt_dir, args.sparse_tag, device)

    probe = torch.rand(64, 1, 28, 28, device=device)  # batch 64 untuk latency
    results = {}
    for name, (model, tag) in models.items():
        res = reconstruction_metrics(model, test_loader, device)
        res["active_units"] = active_unit_fraction(model, test_loader, device)
        res["epoch_time_s"] = mean_epoch_time(args.out_dir, tag)
        res["latency_ms_b64"] = measure_latency(lambda m=model: m(probe), device)
        results[name] = res

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "results_ae_family.json")
    with open(out_path, "w") as f:
        json.dump({"device": str(device), "results": results}, f, indent=2)

    print("\n| Model | MSE | PSNR (dB) | SSIM | Unit aktif | s/epoch | ms/batch-64 |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        t = f"{r['epoch_time_s']:.1f}" if r["epoch_time_s"] is not None else "-"
        print(
            f"| {name} | {r['mse']:.5f} | {r['psnr']:.2f} | {r['ssim']:.4f} | "
            f"{r['active_units'] * 100:.1f}% | {t} | {r['latency_ms_b64']:.3f} |"
        )
    print(f"\ntersimpan: {out_path}")


if __name__ == "__main__":
    main()