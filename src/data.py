import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
SPLIT = (0.8, 0.1, 0.1)

def set_seed(seed: int = SEED):
    # Kunci semua sumber randomness supaya hasil bisa direproduksi
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    # [0, 1] -> [-1, 1]
    return x * 2.0 - 1.0
 
 
def to_zero1(x: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    return (x + 1.0) / 2.0
 
 
def get_dataloaders(batch_size: int = 128, data_dir: str = "data", seed: int = SEED, num_workers: int = 0):

    if batch_size not in (128, 256):
        raise ValueError("Tugas mensyaratkan batch_size 128 atau 256.")
 
    tfm = transforms.ToTensor()
    train_raw = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_raw = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    full = ConcatDataset([train_raw, test_raw])  # 70.000 images
 
    n_total = len(full)
    n_train = int(SPLIT[0] * n_total)
    n_val = int(SPLIT[1] * n_total)
    n_test = n_total - n_train - n_val
 
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full, [n_train, n_val, n_test], generator=generator
    )
 
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
 
 
if __name__ == "__main__":
    set_seed()
    tr, va, te = get_dataloaders()
    print(f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
    x, y = next(iter(tr))
    print(f"batch: x={tuple(x.shape)} range=[{x.min():.2f}, {x.max():.2f}] y={tuple(y.shape)}")