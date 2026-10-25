# Code

Everything needed to reproduce the paper: data preparation, classical baselines, the five reconstruction networks, testing, the acceleration sweep and the ablation studies.

[Back to paper overview](../README.md)

---

## Folder Structure

```
codes/
├── README.md                     you are here
├── requirements.txt
│
├── common/                       shared code
│   ├── config.py                 all settings from the paper
│   ├── data.py                   fastMRI HDF5 loader, cropping, augmentation, masking
│   ├── masks.py                  random / equispaced Cartesian masks with 24 ACS lines
│   ├── mri_ops.py                FFTs, forward model A = P F S, adjoint, data consistency
│   ├── models.py                 Cascade-Net, KIKI-Net, U-Net, PD-Net, U-Net+DC
│   ├── losses.py                 0.6 L2 + 0.3 SSIM + 0.1 perceptual
│   ├── metrics.py                SSIM, PSNR, NMSE, SNR
│   ├── classical.py              zero-filled, CS-TV, SENSE, GRAPPA
│   ├── trainer.py / tester.py    shared training and testing loops
│   ├── baseline_runner.py        shared runner for the classical baselines
│   ├── cli.py                    common command-line options
│   └── evaluation.py             metrics, timing and example figures
│
├── data_preparation/
│   ├── make_splits.py            70 / 15 / 15 split by volume
│   ├── precompute_espirit_maps.py
│   └── make_synthetic_volumes.py small fake volumes for a quick test run
│
├── classical_baselines/
│   ├── Zero_Filled/  CS_TV/  SENSE/  GRAPPA/
│   └── run_all.sh
│
├── training/
│   ├── Cascade_Net/  KIKI_Net/  UNet/  PD_Net/  UNet_DC/
│   └── train_all.sh
│
├── Test/
│   ├── Cascade_Net/  KIKI_Net/  UNet/  PD_Net/  UNet_DC/
│   ├── test_all.sh
│   ├── acceleration_sweep.py     Table III
│   ├── compare_methods.py        Fig. 3-style comparison
│   └── summarize_results.py      Table II
│
└── ablation/
    ├── run_ablation.py           Table IV
    └── iterations_sweep.py       Table V
```

---

## 1. Setup

```bash
git clone https://github.com/1umair003/AI-Based-Reconstruction-for-fastMRI.git
cd AI-Based-Reconstruction-for-fastMRI/codes

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for training. The paper used one Tesla V100 (32 GB) with batch size 4.

---

## 2. Get the Data

1. Request access to fastMRI at **https://fastmri.med.nyu.edu** and download the **knee multi-coil** set (`knee_multicoil_train`, `knee_multicoil_val`).
2. Each `.h5` file holds one volume, with raw k-space under the key `kspace` (slices × coils × height × width).
3. Split by volume, 70 / 15 / 15 as in the paper:

```bash
python data_preparation/make_splits.py \
    --data_dirs /data/knee_multicoil_train /data/knee_multicoil_val --output splits.json
```

Optional, for more accurate coil sensitivity maps (the default estimates them from the ACS lines on the fly):

```bash
pip install sigpy
python data_preparation/precompute_espirit_maps.py --splits splits.json --output_dir data/espirit_maps
# then add  --maps_dir data/espirit_maps  to any script
```

To try the pipeline before downloading fastMRI:

```bash
python data_preparation/make_synthetic_volumes.py --output_dir data/synthetic --n_volumes 6
python data_preparation/make_splits.py --data_dirs data/synthetic --output splits.json
```

---

## 3. Classical Baselines

```bash
python classical_baselines/Zero_Filled/run_zero_filled.py --splits splits.json --acceleration 4
python classical_baselines/CS_TV/run_cs_tv.py             --splits splits.json --acceleration 4
python classical_baselines/SENSE/run_sense.py             --splits splits.json --acceleration 4
python classical_baselines/GRAPPA/run_grappa.py           --splits splits.json --acceleration 4

# or all four
bash classical_baselines/run_all.sh splits.json 4
```

---

## 4. Train the Networks

```bash
python training/UNet_DC/train_unet_dc.py --splits splits.json --acceleration 4

# all five networks
bash training/train_all.sh splits.json 4
```

| Setting | Default |
|---|---|
| Epochs / batch size | 50 / 4 |
| Optimiser | Adam, lr 1e-4, weight decay 1e-5 |
| LR schedule | Halved after 10 epochs without validation improvement |
| Loss | 0.6 L2 + 0.3 SSIM + 0.1 perceptual (`--loss l2` for plain L2) |
| Augmentation | Random flips, 90° rotations, Gaussian noise (`--no_augment` to disable) |
| Image size | 320 × 320 centre crop (`--crop_size`) |

Each run saves `best_model.pt` (best validation SSIM), `last_model.pt`, `history.csv` and `training_curves.png` in the model's `outputs/` folder.

**Faster experiments:** `--max_slices_per_volume 10` uses only the central slices, and `--crop_size 160` trains on smaller images.

**U-Net+DC options:** `--iterations K` (default 10), `--no_dc`, `--no_kspace`, `--base_channels`.

---

## 5. Test the Networks

```bash
python Test/UNet_DC/test_unet_dc.py --splits splits.json --acceleration 4

# all five networks
bash Test/test_all.sh splits.json 4
```

Each test loads the matching `training/<Model>/outputs/best_model.pt` and writes to `Test/<Model>/results/<acc>x/`:

| File | Contents |
|---|---|
| `summary.json` | Mean ± std SSIM, PSNR, NMSE, SNR; parameters; ms per slice |
| `per_slice_metrics.csv` | Metrics for every slice, plus zero-filled metrics for reference |
| `examples/` | Ground truth, mask, zero-filled, reconstruction and residual |

---

## 6. Tables and Figures

```bash
python Test/summarize_results.py --acceleration 4            # Table II
python Test/acceleration_sweep.py --splits splits.json       # Table III (2x, 4x, 6x, 8x)
python Test/compare_methods.py --splits splits.json          # Fig. 3-style comparison
```

---

## 7. Ablation Studies

```bash
python ablation/run_ablation.py --splits splits.json         # Table IV
python ablation/iterations_sweep.py --splits splits.json     # Table V, K = 1, 3, 5, 10, 15, 20
```

Extra training options (for example `--epochs 20`) are passed through to every training run.

---

## Quick Reference

| Step | Command |
|---|---|
| Install | `pip install -r requirements.txt` |
| Split data | `python data_preparation/make_splits.py --data_dirs <dirs> --output splits.json` |
| Baselines | `bash classical_baselines/run_all.sh splits.json 4` |
| Train | `bash training/train_all.sh splits.json 4` |
| Test | `bash Test/test_all.sh splits.json 4` |
| Table II | `python Test/summarize_results.py` |
| Table III | `python Test/acceleration_sweep.py --splits splits.json` |
| Table IV | `python ablation/run_ablation.py --splits splits.json` |
| Table V | `python ablation/iterations_sweep.py --splits splits.json` |

Every script supports `--help`.
