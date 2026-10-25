# Ablation Studies

## Table IV: component ablation

```bash
# from codes/
python ablation/run_ablation.py --splits splits.json
```

| Configuration | DC | k-space branch | Combined loss | K |
|---|---|---|---|---|
| U-Net baseline (L2) | No | No | No | 1 |
| + Data Consistency | Yes | No | No | 5 |
| + Combined Loss | Yes | No | Yes | 5 |
| + K-Space Domain | Yes | Yes | Yes | 5 |
| Full Model | Yes | Yes | Yes | 10 |

The paper doesn't state K for the intermediate rows, so 5 is used (`--ablation_iterations` to change it). Run a subset with `--only plus_dc full_model_k10`.

## Table V: number of unrolled iterations

```bash
python ablation/iterations_sweep.py --splits splits.json --k_values 1 3 5 10 15 20
```

Any extra training option (for example `--epochs 20` or `--max_slices_per_volume 10`) is passed to every run. Results are saved as `table_IV_ablation.csv/.md` and `table_V_iterations.csv/.md`.
