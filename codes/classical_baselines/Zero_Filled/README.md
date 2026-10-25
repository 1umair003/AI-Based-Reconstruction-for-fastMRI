# Zero-Filled IFFT

```bash
# from codes/
python classical_baselines/Zero_Filled/run_zero_filled.py --splits splits.json --acceleration 4
```

Results go to `results/<acc>x/`: `summary.json` (SSIM, PSNR, NMSE, SNR, ms per slice), `per_slice_metrics.csv` and `examples/`. See `--help` for the method's options.
