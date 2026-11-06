# Training loop umum untuk keluarga AE (Vanilla AE, Sparse AE, VAE).
# Contoh: python src/train.py --model ae --latent 64 --epochs 30

import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from models.vanilla_ae import VanillaAE  # noqa: E402

MODEL_REGISTRY = {
    "ae": lambda a: VanillaAE(latent_dim=a.latent),
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        loss, _ = model.compute_loss(x)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    set_seed()
    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.data_dir)
    model = MODEL_REGISTRY[args.model](args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss, _ = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        train_loss = total / n
        val_loss = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train {train_loss:.5f} | "
            f"val {val_loss:.5f} | {epoch_time:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.model}_best.pt"))

    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f)
    print(f"selesai. best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()