# Training loop umum untuk AE (Vanilla AE, Sparse AE, VAE).

# Contoh:
# python src/train.py --model ae --latent 64 --epochs 30
# python src/train.py --model sparse --l1 1e-3 --epochs 30

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from models.sparse_ae import SparseAE  # noqa: E402
from models.vanilla_ae import VanillaAE  # noqa: E402
from models.vae import VAE  # noqa: E402

MODEL_REGISTRY = {
    "ae": lambda a: VanillaAE(latent_dim=a.latent),
    "sparse": lambda a: SparseAE(latent_dim=a.latent, l1_weight=a.l1),
    "vae": lambda a: VAE(latent_dim=a.latent, beta=a.beta),
}


def average_logs(sums, n):
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums, n = defaultdict(float), 0
    for x, _ in loader:
        x = x.to(device)
        _, logs = model.compute_loss(x)
        for k, v in logs.items():
            sums[k] += v * x.size(0)
        n += x.size(0)
    return average_logs(sums, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--l1", type=float, default=1e-2, help="bobot penalti L1 (sparse AE)")
    p.add_argument("--beta", type=float, default=1.0, help="bobot KL (VAE)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default=None, help="nama file output, default = nama model")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    tag = args.tag or args.model

    set_seed()
    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.data_dir)
    model = MODEL_REGISTRY[args.model](args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    history = {"train": [], "val": [], "epoch_time": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums, n = defaultdict(float), 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss, logs = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            for k, v in logs.items():
                sums[k] += v * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        train_logs = average_logs(sums, n)
        val_logs = evaluate(model, val_loader, device)
        history["train"].append(train_logs)
        history["val"].append(val_logs)
        history["epoch_time"].append(epoch_time)

        extra = "".join(
            f" | val {k} {v:.4f}" for k, v in val_logs.items() if k not in ("loss",)
        )
        print(
            f"epoch {epoch:3d}/{args.epochs} | train {train_logs['loss']:.5f} | "
            f"val {val_logs['loss']:.5f}{extra} | {epoch_time:.1f}s"
        )

        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{tag}_best.pt"))

    with open(os.path.join(args.out_dir, f"{tag}_history.json"), "w") as f:
        json.dump(history, f)
    print(f"selesai. best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()