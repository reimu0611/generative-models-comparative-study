# Pembuat figur untuk laporan.

# Saat ini: Fig 1 (grid rekonstruksi: asli vs Vanilla AE vs Sparse AE vs VAE).

# Contoh:
# python src/figures.py --fig 1


import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from evaluate import load_models  # noqa: E402


@torch.no_grad()
def fig1_reconstruction(models, test_loader, device, out_path, n=8):
    """Baris: asli, lalu tiap model. Sampel = n citra pertama test set (tetap)."""
    x, _ = next(iter(test_loader))  # test loader tidak di-shuffle -> sampel selalu sama
    x = x[:n].to(device)

    rows = [("Asli", x)]
    for name, (model, _) in models.items():
        rows.append((name, model(x)))

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.35))
    for r, (name, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(imgs[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"tersimpan: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, choices=[1], default=1)
    p.add_argument("--sparse_tag", default="sparse_l1e-2")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    _, _, test_loader = get_dataloaders(128, args.data_dir)
    models = load_models(args.ckpt_dir, args.sparse_tag, device)

    if args.fig == 1:
        fig1_reconstruction(
            models, test_loader, device, os.path.join(args.out_dir, "fig1_reconstruction.png")
        )


if __name__ == "__main__":
    main()