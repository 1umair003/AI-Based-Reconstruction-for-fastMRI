# Test

| Folder / script | Purpose |
|---|---|
| [`Cascade_Net/`](Cascade_Net) · [`KIKI_Net/`](KIKI_Net) · [`UNet/`](UNet) · [`PD_Net/`](PD_Net) · [`UNet_DC/`](UNet_DC) | Test each trained network |
| `test_all.sh` | Test all five networks |
| `summarize_results.py` | Table II: all networks and baselines in one table |
| `acceleration_sweep.py` | Table III: SSIM at 2×, 4×, 6×, 8× |
| `compare_methods.py` | Fig. 3-style side-by-side comparison on one slice |

## Run (from `codes/`)

```bash
bash Test/test_all.sh splits.json 4
python Test/summarize_results.py --acceleration 4
python Test/acceleration_sweep.py --splits splits.json
python Test/compare_methods.py --splits splits.json --classical Zero_Filled GRAPPA
```

Test masks are fixed per slice (seeded by file name and slice index), so every method sees exactly the same under-sampling.
