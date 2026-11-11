# Studi Komparatif Generative Models: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421), Sesi 07. Enam model generatif dibandingkan pada
**Fashion-MNIST** dengan PyTorch, ditambah eksperimen yang membuktikan bahwa bobot dan fitur
laten MLP Classifier tidak dapat dipakai untuk image generation.

## Struktur repositori

```
src/
├── data.py              # loader Fashion-MNIST, split 80/10/10, seed tetap, konversi rentang piksel
├── metrics.py           # MSE, PSNR, SSIM, fraksi unit aktif, pengukur latency
├── models/
│   ├── vanilla_ae.py    # Vanilla Autoencoder
│   ├── sparse_ae.py     # Sparse Autoencoder (penalti L1 pada hidden 256)
│   ├── vae.py           # Variational Autoencoder
│   ├── vanilla_gan.py   # Vanilla GAN (MLP)
│   ├── dcgan.py         # DCGAN
│   ├── ddpm.py          # U-Net + proses difusi DDPM
│   └── mlp_classifier.py# MLP Classifier dan decoder untuk eksperimen pembuktian
├── train.py             # training keluarga AE (ae, sparse, vae)
├── train_gan.py         # training adversarial (gan, dcgan)
├── train_ddpm.py        # training DDPM
├── train_mlp.py         # training MLP Classifier (tahap 1) dan frozen decoder (tahap 4)
├── evaluate.py          # MSE / PSNR / SSIM / unit aktif model AE-family di test set
├── mlp_experiment.py    # eksperimen pembuktian tahap 2-5 dan Fig 3
├── fid.py               # FID untuk VAE, GAN, DCGAN, DDPM
├── latency.py           # latency batch 64 untuk keenam model
├── figures.py           # Fig 1
└── figures_gen.py       # Fig 2, 4, 5
```

Checkpoint (`checkpoints/`), dataset (`data/`), dan sampel hasil training tidak ikut di-commit.

## Instalasi

```bash
pip install -r requirements.txt
```

`torchmetrics[image]` dipakai hanya oleh `fid.py`. Dataset diunduh otomatis lewat
`torchvision.datasets.FashionMNIST` saat pertama kali dijalankan.

## Data

70.000 citra Fashion-MNIST (train 60.000 + test 10.000 bawaan) digabung lalu dibagi ulang
**80/10/10** (56.000 / 7.000 / 7.000) dengan seed 42. Piksel di rentang [0, 1]. GAN dan DDPM
bekerja di [-1, 1] dan dikonversi di dalam skrip training. Semua metrik dihitung di test set
pada rentang [0, 1].

## Menjalankan (urutan)

Semua perintah dijalankan dari root repositori. Seed tetap 42.

```bash
# 1. Keluarga Autoencoder
python src/train.py --model ae --latent 64
python src/train.py --model sparse --latent 64 --l1 1e-2 --tag sparse_l1e-2
python src/train.py --model vae --latent 64

# 2. GAN, DCGAN, DDPM
python src/train_gan.py --model gan
python src/train_gan.py --model dcgan
python src/train_ddpm.py

# 3. Eksperimen pembuktian MLP Classifier
python src/train_mlp.py --stage classifier
python src/train_mlp.py --stage decoder
python src/mlp_experiment.py          # tahap 2-5, Fig 3

# 4. Evaluasi kuantitatif
python src/evaluate.py                # MSE, PSNR, SSIM, unit aktif (AE, Sparse AE, VAE)
python src/latency.py                 # latency batch 64, keenam model, satu perangkat
python src/fid.py                     # FID VAE, GAN, DCGAN, DDPM (2.000 sampel vs test set)

# 5. Figur
python src/figures.py --fig 1
python src/figures_gen.py             # Fig 2, 4, 5
```

DCGAN, DDPM, dan FID dijalankan di Google Colab (GPU T4). Di Colab: `git clone` repositori,
`pip install -r requirements.txt`, unggah checkpoint yang dibutuhkan ke `checkpoints/`, lalu
jalankan skrip yang sama. Training DDPM berjalan sekitar 1 menit per epoch di T4.
`train_ddpm.py` menerima `--epochs 1 --max_batches 5 --T 50 --tag ddpm_test` untuk tes cepat.

## Konfigurasi

| Model | Arsitektur | Dim laten | Optimizer | LR | Epoch |
|---|---|---|---|---|---|
| Vanilla AE | 784-256-64-256-784, ReLU, Sigmoid | 64 | Adam | 1e-3 | 30 |
| Sparse AE | sama; penalti L1 (λ = 1e-2) pada hidden 256 | 64 / 256 | Adam | 1e-3 | 30 |
| VAE | 784-256-(μ, logσ²) 64-256-784; BCE + KL (β = 1) | 64 | Adam | 1e-3 | 30 |
| Vanilla GAN | G: 100-256-512(BN)-784 Tanh; D: 784-512-256-1 (Dropout 0,3) | 100 | Adam (β1 = 0,5) | 2e-4 | 50 |
| DCGAN | G: ConvT 7x7-14x14-28x28; D: Conv 14x14-7x7-1x1; BN, LeakyReLU | 100 | Adam (β1 = 0,5) | 2e-4 | 50 |
| DDPM | U-Net 3.098.049 parameter, T = 1000, jadwal beta linear 1e-4..0,02, EMA 0,999 | 28x28 | Adam | 2e-4 | 50 |
| MLP Classifier | 784-256-64-10, Cross-Entropy; frozen decoder 64-256-784 (MSE) | 64 | Adam | 1e-3 | 30 |

Batch size 128 untuk semua model. Checkpoint AE-family dipilih berdasarkan val loss
terendah, GAN/DCGAN memakai epoch terakhir, DDPM memakai bobot EMA epoch terakhir.

## Hasil (test set Fashion-MNIST, seed 42)

| Model | s/epoch | Latency ms / batch 64 (CPU) | MSE | PSNR (dB) | SSIM | FID |
|---|---|---|---|---|---|---|
| Vanilla AE | 7,8 (CPU) | 0,50 (rekonstruksi) | 0,00655 | 22,95 | 0,828 | N/A |
| Sparse AE | 8,2 (CPU) | 0,56 (rekonstruksi) | 0,00821 | 21,92 | 0,793 | N/A |
| VAE | 8,6 (CPU) | 0,62 (rekonstruksi) / 0,28 (sampling) | 0,01325 | 19,62 | 0,708 | 92,89 |
| Vanilla GAN | 15-19 (CPU) | 0,85 | N/A | N/A | N/A | 105,66 |
| DCGAN | 23,4 (Colab T4) | 23,09 | N/A | N/A | N/A | 24,01 |
| DDPM | 61,3 (Colab T4) | 389.144 (T = 1000) | N/A | N/A | N/A | 12,98 |

Catatan pengukuran:

- Latency keenam model diukur di satu perangkat (CPU). Latency DDPM diukur satu kali
  (sampling penuh 1000 langkah), model lain dirata-ratakan dari 200 pengulangan.
- Waktu per epoch tidak sebanding langsung antara model yang dilatih di CPU dan di Colab T4.
- FID memakai 2.000 sampel per model terhadap 7.000 citra test set, sehingga bias ke atas dan
  hanya sebanding antarmodel di pengaturan ini.
- Sparse AE: unit aktif pada hidden 256 turun dari 54,5% (Vanilla AE) menjadi 19,3%.

### Eksperimen pembuktian MLP Classifier

Akurasi test classifier 89,3%.

| Tahap | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| 2: proyeksi balik W^T dari vektor kelas | 0,164 | 7,93 | 0,062 |
| 3: proyeksi balik W^T dari z_MLP | 0,298 | 5,30 | 0,078 |
| 4: frozen decoder dari z_MLP | 0,0179 | 18,27 | 0,656 |
| 4: Vanilla AE (z_AE, pembanding) | 0,00655 | 22,95 | 0,828 |

Tahap 5 (sampling dari N(μ_z, diag(σ_z²))): 18,9% elemen sampel z bernilai negatif padahal z_MLP
asli tidak pernah negatif (ReLU), dan 11 dari 64 dimensi mati di test set.

## Reproduktibilitas

Seed 42 dikunci di `data.set_seed` untuk Python, NumPy, dan PyTorch, dan split data memakai
generator ber-seed. Operasi GPU dapat menghasilkan selisih numerik kecil antar-run. Seluruh
hasil di atas berasal dari satu run per model.