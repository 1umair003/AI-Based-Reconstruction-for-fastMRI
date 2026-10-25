# U-Net: Training

Residual U-Net (4 levels, 3×3 conv, GroupNorm, LeakyReLU) applied to the zero-filled image, with no data consistency.

**Parameters:** 4.4M

## Run

```bash
# from codes/
python training/UNet/train_unet.py --splits splits.json --acceleration 4
```

Main options: `--epochs 50` · `--batch_size 4` · `--lr 1e-4` · `--loss combined|l2` · `--crop_size 320` · `--max_slices_per_volume N` · `--maps_dir DIR`

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.pt` | Checkpoint with the best validation SSIM (used by the test script) |
| `last_model.pt` | Checkpoint after the final epoch |
| `history.csv` | Loss and validation SSIM / PSNR / NMSE / SNR per epoch |
| `training_curves.png` | Training loss and validation SSIM |

Test it with [`codes/Test/UNet`](../../Test/UNet).
