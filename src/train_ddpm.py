"""Training DDPM (prediksi noise) pada Fashion-MNIST.

Memakai EMA (exponential moving average) bobot untuk sampling dan checkpoint final.
DDPM tidak punya val loss yang dipakai untuk memilih checkpoint, jadi yang disimpan
adalah epoch terakhir.

Contoh:
    python src/train_ddpm.py --epochs 50
    python src/train_ddpm.py --epochs 1 --max_batches 5 --T 50 --tag ddpm_test   # tes cepat
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed, to_minus1_1, to_zero1  # noqa: E402
from models.ddpm import GaussianDiffusion, UNet  # noqa: E402


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000, help="jumlah langkah difusi")
    p.add_argument("--base", type=int, default=64, help="lebar channel dasar U-Net")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=0, help="batasi batch per epoch (tes cepat)")
    p.add_argument("--tag", default="ddpm")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    print(f"device: {device}")

    train_loader, _, _ = get_dataloaders(args.batch_size, args.data_dir)
    model = UNet(base=args.base).to(device)
    ema_model = copy.deepcopy(model).eval()
    for prm in ema_model.parameters():
        prm.requires_grad_(False)
    diffusion = GaussianDiffusion(T=args.T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"parameter U-Net: {sum(q.numel() for q in model.parameters()):,}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, f"{args.tag}_samples")
    os.makedirs(sample_dir, exist_ok=True)

    history = {"loss": [], "epoch_time": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for i, (x, _) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x = to_minus1_1(x.to(device))
            loss = diffusion.loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema)
            total += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        avg = total / n
        history["loss"].append(avg)
        history["epoch_time"].append(epoch_time)
        print(f"epoch {epoch:3d}/{args.epochs} | loss {avg:.5f} | {epoch_time:.1f}s")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            imgs = diffusion.sample(ema_model, 16, device)
            save_image(to_zero1(imgs), os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), nrow=4)

    torch.save(ema_model.state_dict(), os.path.join(args.ckpt_dir, f"{args.tag}_ema_last.pt"))
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.tag}_last.pt"))
    with open(os.path.join(args.out_dir, f"{args.tag}_history.json"), "w") as f:
        json.dump(history, f)
    print("selesai.")


if __name__ == "__main__":
    main()