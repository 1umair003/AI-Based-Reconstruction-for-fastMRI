# U-Net + DC (proposed): Training

Shared-weight U-Net with a k-space branch and a closed-form data-consistency layer with learned λ, unrolled K = 10 times.

**Parameters:** 4.4M

## Run

```bash
# from codes/
python training/UNet_DC/train_unet_dc.py --splits splits.json --acceleration 4
```

Main options: `--epochs 50` · `--batch_size 4` · `--lr 1e-4` · `--loss combined|l2` · `--crop_size 320` · `--max_slices_per_volume N` · `--maps_dir DIR`

## Options for the ablation

| Flag | Effect |
|---|---|
| `--iterations K` | Number of unrolled iterations (default 10) |
| `--no_dc` | Remove the data-consistency layers |
| `--no_kspace` | Remove the k-space branch |
| `--loss l2` | Plain L2 loss instead of the combined loss |
| `--base_channels` | U-Net width (24 gives 4.4M parameters) |

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.pt` | Checkpoint with the best validation SSIM (used by the test script) |
| `last_model.pt` | Checkpoint after the final epoch |
| `history.csv` | Loss and validation SSIM / PSNR / NMSE / SNR per epoch |
| `training_curves.png` | Training loss and validation SSIM |

Test it with [`codes/Test/UNet_DC`](../../Test/UNet_DC).
