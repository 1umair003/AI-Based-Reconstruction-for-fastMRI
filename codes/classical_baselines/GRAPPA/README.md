# GRAPPA

```bash
# from codes/
python classical_baselines/GRAPPA/run_grappa.py --splits splits.json --acceleration 4
```

Results go to `results/<acc>x/`: `summary.json` (SSIM, PSNR, NMSE, SNR, ms per slice), `per_slice_metrics.csv` and `examples/`. See `--help` for the method's options.
