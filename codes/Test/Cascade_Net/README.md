# Cascade-Net: Testing

Evaluates the model trained in [`codes/training/Cascade_Net`](../../training/Cascade_Net).

## Run

```bash
# from codes/
python Test/Cascade_Net/test_cascade_net.py --splits splits.json --acceleration 4
```

By default it loads `codes/training/Cascade_Net/outputs/best_model.pt`; use `--checkpoint` to point elsewhere. Run it with `--acceleration 2`, `6` or `8` to test other factors.

## Outputs (`results/<acc>x/`)

| File | Contents |
|---|---|
| `summary.json` | Mean ± std SSIM, PSNR, NMSE, SNR; parameter count; inference ms per slice |
| `per_slice_metrics.csv` | Metrics for every test slice |
| `examples/` | Ground truth, mask, zero-filled, reconstruction and residual |
