# Training loop adversarial untuk Vanilla GAN (dan DCGAN nanti).

# GAN tidak punya val loss yang bermakna, jadi checkpoint yang disimpan adalah epoch terakhir. Kurva D(x) dan D(G(z)) dicatat untuk menganalisis kestabilan (vanishing gradient, mode collapse) di Q3.

# Contoh:
# python src/train_gan.py --model gan --epochs 50
# python src/train_gan.py --model gan --label_smooth 0.9 --tag gan_smooth

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed, to_minus1_1, to_zero1  # noqa: E402
from models.vanilla_gan import VanillaDiscriminator, VanillaGenerator  # noqa: E402

MODEL_REGISTRY = {
    "gan": lambda a: (VanillaGenerator(z_dim=a.z_dim), VanillaDiscriminator()),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="gan")
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--label_smooth", type=float, default=1.0,
                   help="target label untuk citra real; 1.0 = tanpa smoothing, 0.9 = one-sided")
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--tag", default=None)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    tag = args.tag or args.model

    set_seed()
    device = get_device()
    print(f"device: {device}")

    train_loader, _, _ = get_dataloaders(args.batch_size, args.data_dir)
    G, D = MODEL_REGISTRY[args.model](args)
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, f"{tag}_samples")
    os.makedirs(sample_dir, exist_ok=True)

    fixed_z = torch.randn(16, args.z_dim, device=device)  # noise tetap untuk memantau progres
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        t0 = time.time()
        sums, n = defaultdict(float), 0

        for x, _ in train_loader:
            real = to_minus1_1(x.to(device))
            b = real.size(0)
            real_target = torch.full((b, 1), args.label_smooth, device=device)
            fake_target = torch.zeros(b, 1, device=device)

            # ---- update Discriminator ----
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_d = F.binary_cross_entropy(d_real, real_target) + F.binary_cross_entropy(
                d_fake, fake_target
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- update Generator (non-saturating: label fake dianggap real) ----
            z = torch.randn(b, args.z_dim, device=device)
            d_gen = D(G(z))
            loss_g = F.binary_cross_entropy(d_gen, torch.ones(b, 1, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            for k, v in {
                "d_loss": loss_d.item(),
                "g_loss": loss_g.item(),
                "d_real": d_real.mean().item(),  # D(x): rata-rata skor untuk citra real
                "d_fake": d_fake.mean().item(),  # D(G(z)): rata-rata skor untuk citra palsu
            }.items():
                sums[k] += v * b
            n += b

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        logs = {k: v / n for k, v in sums.items()}
        for k, v in logs.items():
            history[k].append(v)
        history["epoch_time"].append(epoch_time)
        print(
            f"epoch {epoch:3d}/{args.epochs} | D {logs['d_loss']:.3f} | G {logs['g_loss']:.3f} | "
            f"D(x) {logs['d_real']:.3f} | D(G(z)) {logs['d_fake']:.3f} | {epoch_time:.1f}s"
        )

        if epoch % args.sample_every == 0 or epoch == args.epochs or epoch == 1:
            G.eval()
            with torch.no_grad():
                imgs = to_zero1(G(fixed_z))
            G.train()
            save_image(imgs, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), nrow=4)

    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, f"{tag}_G_last.pt"))
    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, f"{tag}_D_last.pt"))
    with open(os.path.join(args.out_dir, f"{tag}_history.json"), "w") as f:
        json.dump(history, f)
    print("selesai.")


if __name__ == "__main__":
    main()