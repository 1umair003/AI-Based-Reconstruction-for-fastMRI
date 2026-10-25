# Classical Baselines

| Method | Folder | How it works | Default mask |
|---|---|---|---|
| Zero-filled IFFT | [`Zero_Filled/`](Zero_Filled) | Inverse FFT with missing lines set to zero, coil-combined | Random |
| CS-TV | [`CS_TV/`](CS_TV) | FISTA for ½‖Ax − y‖² + λ TV(x) (Eqs. 4-5) | Random |
| SENSE | [`SENSE/`](SENSE) | Conjugate-gradient solution of (AᴴA + μI)x = Aᴴy (Eq. 6) | Random |
| GRAPPA | [`GRAPPA/`](GRAPPA) | k-space interpolation with weights fitted on the ACS lines (Eq. 7) | Equispaced |

## Run (from `codes/`)

```bash
bash classical_baselines/run_all.sh splits.json 4

# individual methods with their main options
python classical_baselines/CS_TV/run_cs_tv.py --splits splits.json --lam 0.003 --iterations 50
python classical_baselines/SENSE/run_sense.py --splits splits.json --reg 1e-3 --iterations 30
python classical_baselines/GRAPPA/run_grappa.py --splits splits.json --kernel_kx 3
```

Each writes `results/<acc>x/summary.json`, `per_slice_metrics.csv` and example figures, in the same format as the network tests.

> CS-TV runs on the CPU and is slow on the full test set. Use `--max_slices_per_volume 5` for a quicker estimate.
