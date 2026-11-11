"""MLP Classifier pendamping dan komponen eksperimen pembuktian (Bagian III).

Classifier : 784 -> 256 -> 64 -> 10   (ReLU, loss Cross-Entropy pada logits)
Decoder    : 64 -> 256 -> 784         (Sigmoid, loss MSE) -- dilatih di atas fitur beku

z_MLP didefinisikan sebagai aktivasi layer 64 SETELAH ReLU, yaitu masukan layer
klasifikasi akhir (lapisan sebelum klasifikasi).

Konvensi bobot PyTorch (out_features, in_features):
    W1 = fc1.weight  (256, 784)
    W2 = fc2.weight  (64, 256)
    W3 = fc3.weight  (10, 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """z_MLP in R^64 (setelah ReLU)."""
        h1 = F.relu(self.fc1(x.flatten(start_dim=1)))
        return F.relu(self.fc2(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))  # logits


class MLPDecoder(nn.Module):
    """Decoder 64 -> 256 -> 784 (arsitektur sama dengan decoder Vanilla AE)."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class FrozenLatentReconstructor(nn.Module):
    """x -> z_MLP (classifier beku) -> decoder -> x_hat. Antarmuka sama dengan model AE."""

    def __init__(self, classifier: MLPClassifier, decoder: MLPDecoder):
        super().__init__()
        self.classifier = classifier
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.classifier.features(x)
        return self.decoder(z)