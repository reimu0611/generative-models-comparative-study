"""Figur 2, 4, dan 5 untuk laporan (Fig 1 ada di figures.py, Fig 3 di mlp_experiment.py).

  Fig 2: grid sampling acak 4x4 dari VAE, Vanilla GAN, DCGAN, DDPM
  Fig 4: interpolasi linier latent (alpha = 0, 0.2, ..., 1) pada 5 model
  Fig 5: kurva loss 6 model + trajektori reverse diffusion DDPM

Contoh:
    python src/figures_gen.py                 # ketiga figur
    python src/figures_gen.py --fig 4
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed, to_zero1  # noqa: E402
from models.dcgan import DCGANGenerator  # noqa: E402
from models.ddpm import GaussianDiffusion, UNet  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402
from models.sparse_ae import SparseAE  # noqa: E402
from models.vae import VAE  # noqa: E402
from models.vanilla_ae import VanillaAE  # noqa: E402
from models.vanilla_gan import VanillaGenerator  # noqa: E402

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRAJ_STEPS = [999, 800, 600, 400, 200, 100, 50, 20, 0]  # indeks t (0-based), 999 = noise murni


def load(model, args, fname, device):
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, fname), map_location=device))
    return model.to(device).eval()


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"tersimpan: {path}")


# --------------------------------------------------------------------------- Fig 2
def fig2(args, device):
    vae = load(VAE(64), args, "vae_best.pt", device)
    gan_g = load(VanillaGenerator(100), args, "gan_G_last.pt", device)
    dcgan_g = load(DCGANGenerator(100), args, "dcgan_G_last.pt", device)
    unet = load(UNet(64), args, "ddpm_ema_last.pt", device)
    diffusion = GaussianDiffusion(1000).to(device)

    panels = [
        ("(A) VAE", lambda: vae.sample(16, device)),
        ("(B) Vanilla GAN", lambda: to_zero1(gan_g.sample(16, device))),
        ("(C) DCGAN", lambda: to_zero1(dcgan_g.sample(16, device))),
        ("(D) DDPM", lambda: to_zero1(diffusion.sample(unet, 16, device))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (title, fn) in zip(axes, panels):
        torch.manual_seed(args.seed)  # seed sama tiap panel
        imgs = fn().clamp(0, 1).cpu()
        grid = make_grid(imgs, nrow=4, padding=2, pad_value=1.0)
        ax.imshow(grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    save(fig, os.path.join(args.out_dir, "fig2_sampling.png"))


# --------------------------------------------------------------------------- Fig 4
def first_of_class(loader, cls, device):
    """Citra test set pertama dengan label cls (test loader tidak di-shuffle -> tetap)."""
    for x, y in loader:
        idx = (y == cls).nonzero()
        if len(idx):
            return x[idx[0, 0]].unsqueeze(0).to(device)
    raise RuntimeError(f"kelas {cls} tidak ditemukan di test set")


@torch.no_grad()
def noise_for_class(gen, clf, cls, z_dim, device, seed, n=2000):
    """GAN tidak punya encoder: ambil z acak (seed tetap) yang dikenali classifier sebagai kelas cls."""
    torch.manual_seed(seed)
    z = torch.randn(n, z_dim, device=device)
    pred = clf(to_zero1(gen(z))).argmax(dim=1)
    idx = (pred == cls).nonzero()
    if len(idx) == 0:
        raise RuntimeError(f"tidak ada sampel kelas {cls} di {n} noise acak")
    return z[idx[0, 0]].unsqueeze(0)


def interpolate(z_a, z_b, decode):
    a = torch.tensor(ALPHAS, device=z_a.device).view(-1, 1)
    return decode((1.0 - a) * z_a + a * z_b).clamp(0, 1).cpu()


@torch.no_grad()
def fig4(args, device):
    _, _, test_loader = get_dataloaders(128, args.data_dir)
    x_a = first_of_class(test_loader, args.class_a, device)
    x_b = first_of_class(test_loader, args.class_b, device)

    ae = load(VanillaAE(64), args, "ae_best.pt", device)
    sparse = load(SparseAE(64), args, f"{args.sparse_tag}_best.pt", device)
    vae = load(VAE(64), args, "vae_best.pt", device)
    gan_g = load(VanillaGenerator(100), args, "gan_G_last.pt", device)
    dcgan_g = load(DCGANGenerator(100), args, "dcgan_G_last.pt", device)
    clf = load(MLPClassifier(), args, "mlp_cls_best.pt", device)

    rows = []
    for name, model in [("Vanilla AE", ae), ("Sparse AE", sparse), ("VAE", vae)]:
        rows.append((name, interpolate(model.encode(x_a), model.encode(x_b), model.decode)))
    for name, g in [("Vanilla GAN", gan_g), ("DCGAN", dcgan_g)]:
        z_a = noise_for_class(g, clf, args.class_a, 100, device, args.seed)
        z_b = noise_for_class(g, clf, args.class_b, 100, device, args.seed)
        rows.append((name, interpolate(z_a, z_b, lambda z, g=g: to_zero1(g(z)))))

    fig, axes = plt.subplots(len(rows), len(ALPHAS), figsize=(len(ALPHAS) * 1.5, len(rows) * 1.6))
    for r, (name, imgs) in enumerate(rows):
        for c in range(len(ALPHAS)):
            ax = axes[r][c]
            ax.imshow(imgs[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(rf"$\alpha$={ALPHAS[c]:.1f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(
        f"Interpolasi linier: {CLASS_NAMES[args.class_a]} ($\\alpha$=0) $\\rightarrow$ "
        f"{CLASS_NAMES[args.class_b]} ($\\alpha$=1)", fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(args.out_dir, "fig4_interpolation.png"))


# --------------------------------------------------------------------------- Fig 5
def read_history(args, name):
    with open(os.path.join(args.out_dir, name)) as f:
        return json.load(f)


def curve(h, split, key):
    """Dukung format history baru (list of dict) dan format lama (train_loss / val_loss)."""
    if isinstance(h.get(split), list):
        return [d[key] for d in h[split]]
    return h[f"{split}_loss"]


@torch.no_grad()
def fig5(args, device):
    ae_h = read_history(args, "ae_history.json")
    sp_h = read_history(args, f"{args.sparse_tag}_history.json")
    vae_h = read_history(args, "vae_history.json")
    gan_h = read_history(args, "gan_history.json")
    dcgan_h = read_history(args, "dcgan_history.json")
    ddpm_h = read_history(args, "ddpm_history.json")

    unet = load(UNet(64), args, "ddpm_ema_last.pt", device)
    diffusion = GaussianDiffusion(1000).to(device)
    torch.manual_seed(args.seed)
    x0, snaps = diffusion.sample(unet, 2, device, save_steps=TRAJ_STEPS)
    frames = [(f"t={t}", x) for t, x in snaps] + [("$x_0$", x0)]

    fig = plt.figure(figsize=(13, 10.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.15], hspace=0.22)
    top = outer[0].subgridspec(2, 3, hspace=0.45, wspace=0.28)

    def panel(i, title, series, ylabel="Loss"):
        ax = fig.add_subplot(top[i // 3, i % 3])
        for label, values, style in series:
            ax.plot(range(1, len(values) + 1), values, style, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    panel(0, "(a) Vanilla AE: loss rekonstruksi (MSE)", [
        ("train", curve(ae_h, "train", "recon"), "-"), ("val", curve(ae_h, "val", "recon"), "--")])
    panel(1, "(b) Sparse AE: loss rekonstruksi (MSE)", [
        ("train", curve(sp_h, "train", "recon"), "-"), ("val", curve(sp_h, "val", "recon"), "--")])
    panel(2, "(c) VAE: negatif ELBO (BCE + KL)", [
        ("train", curve(vae_h, "train", "loss"), "-"), ("val", curve(vae_h, "val", "loss"), "--")])
    panel(3, "(d) Vanilla GAN: loss D dan G", [
        ("D", gan_h["d_loss"], "-"), ("G", gan_h["g_loss"], "-")])
    panel(4, "(e) DCGAN: loss D dan G", [
        ("D", dcgan_h["d_loss"], "-"), ("G", dcgan_h["g_loss"], "-")])
    panel(5, "(f) DDPM: loss prediksi derau (MSE)", [("train", ddpm_h["loss"], "-")])

    n_cols = len(frames)
    bottom = outer[1].subgridspec(3, n_cols, height_ratios=[0.25, 1, 1], hspace=0.12, wspace=0.05)
    title_ax = fig.add_subplot(bottom[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.3, "(g) Trajektori reverse diffusion DDPM: $x_T$ (derau) $\\rightarrow$ $x_0$ (citra)",
                  ha="center", va="center", fontsize=10)
    for s in range(2):
        for c, (label, x) in enumerate(frames):
            ax = fig.add_subplot(bottom[1 + s, c])
            ax.imshow(to_zero1(x[s, 0]).clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_title(label, fontsize=8)
    save(fig, os.path.join(args.out_dir, "fig5_loss_and_trajectory.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, nargs="+", choices=[2, 4, 5], default=[2, 4, 5])
    p.add_argument("--sparse_tag", default="sparse_l1e-2")
    p.add_argument("--class_a", type=int, default=1, help="kelas titik A (1 = Trouser)")
    p.add_argument("--class_b", type=int, default=9, help="kelas titik B (9 = Ankle boot)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")
    for f in args.fig:
        {2: fig2, 4: fig4, 5: fig5}[f](args, device)


if __name__ == "__main__":
    main()