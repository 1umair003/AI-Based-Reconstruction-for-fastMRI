# KIKI-Net: Training

Cross-domain network: a k-space CNN, then an image CNN, then data consistency, repeated twice (6 layers, 128 filters).

**Parameters:** 2.4M

## Run

```bash
# from codes/
python training/KIKI_Net/train_kiki_net.py --splits splits.json --acceleration 4
```

Main options: `--epochs 50` · `--batch_size 4` · `--lr 1e-4` · `--loss combined|l2` · `--crop_size 320` · `--max_slices_per_volume N` · `--maps_dir DIR`

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.pt` | Checkpoint with the best validation SSIM (used by the test script) |
| `last_model.pt` | Checkpoint after the final epoch |
| `history.csv` | Loss and validation SSIM / PSNR / NMSE / SNR per epoch |
| `training_curves.png` | Training loss and validation SSIM |

Test it with [`codes/Test/KIKI_Net`](../../Test/KIKI_Net).
