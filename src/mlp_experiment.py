"""Eksperimen pembuktian batasan MLP Classifier (Tahap 2, 3, 4, 5) dan Fig 3.

Prasyarat: checkpoints/mlp_cls_best.pt, mlp_dec_best.pt, dan ae_best.pt sudah ada.

Contoh:
    python src/mlp_experiment.py
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from metrics import (  # noqa: E402
    per_image_mse,
    per_image_psnr,
    per_image_ssim,
    reconstruction_metrics,
)
from models.mlp import (  # noqa: E402
    FrozenLatentReconstructor,
    MLPClassifier,
    MLPDecoder,
)
from models.vanilla_ae import VanillaAE  # noqa: E402

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def backproject_from_class(clf, y_onehot):
    """Tahap 2: z_hat = ReLU(W3^T y), h1_hat = ReLU(W2^T z_hat), x_hat = Sigmoid(W1^T h1_hat).

    Bentuk baris: y @ W3 sama dengan (W3^T y)^T untuk W3 berbentuk (10, 64). Bias diabaikan.
    """
    z_hat = F.relu(y_onehot @ clf.fc3.weight)
    h1_hat = F.relu(z_hat @ clf.fc2.weight)
    return torch.sigmoid(h1_hat @ clf.fc1.weight).view(-1, 1, 28, 28)


def backproject_from_latent(clf, z):
    """Tahap 3: x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP)). Bias diabaikan."""
    h1_hat = F.relu(z @ clf.fc2.weight)
    return torch.sigmoid(h1_hat @ clf.fc1.weight).view(-1, 1, 28, 28)


@torch.no_grad()
def eval_projection(fn, loader, device):
    """MSE, PSNR, SSIM rata-rata antara citra asli dan fn(x, y) di seluruh loader."""
    mses, psnrs, ssims = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_hat = fn(x, y)
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
def latent_statistics(clf, loader, device):
    """mu_z dan sigma_z per dimensi dari z_MLP pada test set (Tahap 5)."""
    zs = torch.cat([clf.features(x.to(device)).cpu() for x, _ in loader])
    return zs.mean(dim=0), zs.std(dim=0), zs


@torch.no_grad()
def accuracy(clf, loader, device):
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (clf(x).argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return correct / n


def make_figure3(rows, labels, out_path, n):
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.4))
    for r, (name, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(imgs[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
            if r == 0:
                ax.set_title(CLASS_NAMES[labels[c]], fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"tersimpan: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_show", type=int, default=8)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    _, _, test_loader = get_dataloaders(128, args.data_dir)

    clf = MLPClassifier().to(device)
    clf.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "mlp_cls_best.pt"), map_location=device))
    dec = MLPDecoder().to(device)
    dec.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "mlp_dec_best.pt"), map_location=device))
    ae = VanillaAE(64).to(device)
    ae.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "ae_best.pt"), map_location=device))
    for m in (clf, dec, ae):
        m.eval()

    results = {"test_accuracy": accuracy(clf, test_loader, device)}
    print(f"akurasi test classifier: {results['test_accuracy']:.4f}")

    # Tahap 2: proyeksi balik W^T dari vektor one-hot kelas sebenarnya
    results["stage2_WT_from_class"] = eval_projection(
        lambda x, y: backproject_from_class(clf, F.one_hot(y, 10).float()), test_loader, device
    )
    # Tahap 3: proyeksi balik W^T dari z_MLP citra uji
    results["stage3_WT_from_latent"] = eval_projection(
        lambda x, y: backproject_from_latent(clf, clf.features(x)), test_loader, device
    )
    # Tahap 4: frozen latent decoder vs Vanilla AE (latent sama-sama 64)
    results["stage4_frozen_decoder"] = reconstruction_metrics(
        FrozenLatentReconstructor(clf, dec), test_loader, device
    )
    results["stage4_vanilla_ae"] = reconstruction_metrics(ae, test_loader, device)

    # Tahap 5: sampling z_rand ~ N(mu_z, diag(sigma_z^2)) -> decoder
    mu, sigma, zs = latent_statistics(clf, test_loader, device)
    mu, sigma = mu.to(device), sigma.to(device)
    n = args.n_show
    z_rand = mu + sigma * torch.randn(n, 64, device=device)
    with torch.no_grad():
        x_rand = dec(z_rand)
    results["stage5"] = {
        "negative_fraction_of_sampled_z": (z_rand < 0).float().mean().item(),
        "dead_dimensions": int((sigma < 1e-6).sum().item()),
        "min_of_real_z": zs.min().item(),
    }

    # Fig 3
    x, y = next(iter(test_loader))
    x, y = x[:n].to(device), y[:n].to(device)
    with torch.no_grad():
        rows = [
            ("(1) Asli", x),
            ("(2a) W$^T$ dari $y_c$", backproject_from_class(clf, F.one_hot(y, 10).float())),
            ("(2b) W$^T$ dari $z_{MLP}$", backproject_from_latent(clf, clf.features(x))),
            ("(3a) Frozen decoder", dec(clf.features(x))),
            ("(3b) Vanilla AE", ae(x)),
            ("(4) Sampling $z_{rand}$", x_rand),
        ]
    make_figure3(rows, y.cpu().tolist(), os.path.join(args.out_dir, "fig3_mlp_limits.png"), n)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "results_mlp_experiment.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n| Tahap | MSE | PSNR (dB) | SSIM |")
    print("|---|---|---|---|")
    for key, name in [
        ("stage2_WT_from_class", "2: W^T dari y_c"),
        ("stage3_WT_from_latent", "3: W^T dari z_MLP"),
        ("stage4_frozen_decoder", "4: Frozen decoder (z_MLP)"),
        ("stage4_vanilla_ae", "4: Vanilla AE (z_AE)"),
    ]:
        r = results[key]
        print(f"| {name} | {r['mse']:.5f} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
    print(f"\nTahap 5: {results['stage5']}")
    print(f"tersimpan: {out_json}")


if __name__ == "__main__":
    main()