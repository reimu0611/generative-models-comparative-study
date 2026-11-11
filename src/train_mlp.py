"""Training untuk eksperimen pembuktian MLP Classifier.

Tahap 1 (classifier): latih 784-256-64-10 dengan Cross-Entropy, target akurasi test >= 88%.
Tahap 4 (decoder)   : bekukan classifier, latih decoder 64-256-784 dengan MSE pada z_MLP.

Contoh:
    python src/train_mlp.py --stage classifier
    python src/train_mlp.py --stage decoder
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_dataloaders, get_device, set_seed  # noqa: E402
from models.mlp import MLPClassifier, MLPDecoder  # noqa: E402


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return correct / n


def run_classifier(args, device, train_loader, val_loader, test_loader):
    model = MLPClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_acc": [], "epoch_time": []}
    best_acc = 0.0
    path = os.path.join(args.ckpt_dir, "mlp_cls_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        epoch_time = time.time() - t0
        val_acc = accuracy(model, val_loader, device)
        history["train_loss"].append(total / n)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        print(f"epoch {epoch:3d}/{args.epochs} | train CE {total / n:.4f} | "
              f"val acc {val_acc:.4f} | {epoch_time:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path)

    model.load_state_dict(torch.load(path, map_location=device))
    test_acc = accuracy(model, test_loader, device)
    history["test_acc"] = test_acc
    print(f"akurasi test (checkpoint terbaik): {test_acc:.4f}")
    if test_acc < 0.88:
        print("PERINGATAN: akurasi test < 88% (syarat tugas). Tambah epoch atau atur lr.")
    return history


def run_decoder(args, device, train_loader, val_loader):
    clf = MLPClassifier().to(device)
    clf.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "mlp_cls_best.pt"), map_location=device))
    clf.eval()
    for prm in clf.parameters():
        prm.requires_grad_(False)  # classifier dibekukan

    dec = MLPDecoder().to(device)
    optimizer = torch.optim.Adam(dec.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")
    path = os.path.join(args.ckpt_dir, "mlp_dec_best.pt")

    for epoch in range(1, args.epochs + 1):
        dec.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z = clf.features(x)
            loss = F.mse_loss(dec(z), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        epoch_time = time.time() - t0

        dec.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                v_total += F.mse_loss(dec(clf.features(x)), x).item() * x.size(0)
                v_n += x.size(0)
        val_loss = v_total / v_n
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)
        print(f"epoch {epoch:3d}/{args.epochs} | train {total / n:.5f} | val {val_loss:.5f} | {epoch_time:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dec.state_dict(), path)

    print(f"selesai. best val MSE: {best_val:.5f}")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["classifier", "decoder"], required=True)
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
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir)
    if args.stage == "classifier":
        history = run_classifier(args, device, train_loader, val_loader, test_loader)
        name = "mlp_cls_history.json"
    else:
        history = run_decoder(args, device, train_loader, val_loader)
        name = "mlp_dec_history.json"
    with open(os.path.join(args.out_dir, name), "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()