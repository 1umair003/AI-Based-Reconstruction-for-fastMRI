# Data Preparation

| Script | Purpose |
|---|---|
| `make_splits.py` | Split fastMRI volumes 70 / 15 / 15 into `splits.json` (by volume, seed 42) |
| `precompute_espirit_maps.py` | Optional ESPIRiT coil sensitivity maps (needs `sigpy`) |
| `make_synthetic_volumes.py` | Small phantom volumes in fastMRI format for a quick test run |

```bash
# from codes/
python data_preparation/make_splits.py --data_dirs /data/knee_multicoil_train /data/knee_multicoil_val --output splits.json
python data_preparation/precompute_espirit_maps.py --splits splits.json --output_dir data/espirit_maps
```

### How each slice is prepared

1. Inverse FFT per coil, centre-crop to 320 × 320, FFT back to get a fully sampled k-space matching the cropped field of view.
2. Training only: random flips, 90° rotations and additive Gaussian noise.
3. Coil sensitivity maps from the 24 ACS lines, or pre-computed ESPIRiT maps.
4. SENSE-combined complex target image.
5. Cartesian mask: random per sample for training, fixed per slice for validation and testing.
6. Normalisation by the maximum of the zero-filled image.
