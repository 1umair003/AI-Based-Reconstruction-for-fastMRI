# Cascade-Net: Training

Deep cascade of 5 CNNs, each with 5 convolution layers (118 filters), with hard data consistency after every cascade.

**Parameters:** 1.9M

## Run

```bash
# from codes/
python training/Cascade_Net/train_cascade_net.py --splits splits.json --acceleration 4
```

Main options: `--epochs 50` · `--batch_size 4` · `--lr 1e-4` · `--loss combined|l2` · `--crop_size 320` · `--max_slices_per_volume N` · `--maps_dir DIR`

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.pt` | Checkpoint with the best validation SSIM (used by the test script) |
| `last_model.pt` | Checkpoint after the final epoch |
| `history.csv` | Loss and validation SSIM / PSNR / NMSE / SNR per epoch |
| `training_curves.png` | Training loss and validation SSIM |

Test it with [`codes/Test/Cascade_Net`](../../Test/Cascade_Net).
