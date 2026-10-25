# PD-Net: Training

Learned primal-dual network: 10 unrolled iterations of a dual update in k-space and a primal update in the image domain, using the full multi-coil operator.

**Parameters:** 1.4M

## Run

```bash
# from codes/
python training/PD_Net/train_pd_net.py --splits splits.json --acceleration 4
```

Main options: `--epochs 50` · `--batch_size 4` · `--lr 1e-4` · `--loss combined|l2` · `--crop_size 320` · `--max_slices_per_volume N` · `--maps_dir DIR`

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.pt` | Checkpoint with the best validation SSIM (used by the test script) |
| `last_model.pt` | Checkpoint after the final epoch |
| `history.csv` | Loss and validation SSIM / PSNR / NMSE / SNR per epoch |
| `training_curves.png` | Training loss and validation SSIM |

Test it with [`codes/Test/PD_Net`](../../Test/PD_Net).
