# Training

| Model | Folder | Parameters | Type |
|---|---|---|---|
| Cascade-Net | [`Cascade_Net/`](Cascade_Net) | 1.9M | DL |
| KIKI-Net | [`KIKI_Net/`](KIKI_Net) | 2.4M | DL |
| U-Net | [`UNet/`](UNet) | 4.4M | DL |
| PD-Net | [`PD_Net/`](PD_Net) | 1.4M | DL |
| **U-Net + DC (proposed)** | [`UNet_DC/`](UNet_DC) | 4.4M | DL + DC |

## Run

```bash
# from codes/: all five networks at 4x
bash training/train_all.sh splits.json 4
```

## Shared training setup

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 1e-4, weight decay 1e-5 |
| Epochs / batch | 50 / 4 |
| Schedule | Learning rate halved after 10 epochs without validation improvement |
| Loss | 0.6 L2 + 0.3 (1 − SSIM) + 0.1 perceptual (VGG16) |
| Augmentation | Random flips, 90° rotations, additive Gaussian noise |
| Masks | Random Cartesian, 24 ACS lines, new mask every training sample |
| Checkpoint | Best validation SSIM |

Matching tests: [`codes/Test`](../Test)
