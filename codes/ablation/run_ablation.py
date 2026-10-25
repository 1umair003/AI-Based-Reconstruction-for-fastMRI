"""Ablation study of the proposed model (Table IV).

Trains and tests five configurations of U-Net + DC:

| Configuration          | DC | k-space branch | Combined loss | K  |
|------------------------|----|----------------|---------------|----|
| U-Net baseline (L2)    | no | no             | no (L2)       | 1  |
| + Data Consistency     | yes| no             | no (L2)       | Ka |
| + Combined Loss        | yes| no             | yes           | Ka |
| + K-Space Domain       | yes| yes            | yes           | Ka |
| Full Model (K=10)      | yes| yes            | yes           | 10 |

The paper does not state K for the intermediate rows; Ka defaults to 5
(`--ablation_iterations`).

Usage (from codes/):
    python ablation/run_ablation.py --splits splits.json
    python ablation/run_ablation.py --splits splits.json --epochs 20      # quicker
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent

TRAIN = CODES / "training" / "UNet_DC" / "train_unet_dc.py"
TEST = CODES / "Test" / "UNet_DC" / "test_unet_dc.py"


def configs(ka: int):
    return [
        ("unet_baseline_l2", "U-Net baseline (L2)", ["--no_dc", "--no_kspace", "--loss", "l2", "--iterations", "1"]),
        ("plus_dc", "+ Data Consistency (DC)", ["--no_kspace", "--loss", "l2", "--iterations", str(ka)]),
        ("plus_combined_loss", "+ Combined Loss", ["--no_kspace", "--loss", "combined", "--iterations", str(ka)]),
        ("plus_kspace_domain", "+ K-Space Domain", ["--loss", "combined", "--iterations", str(ka)]),
        ("full_model_k10", "Full Model (K=10 iter.)", ["--loss", "combined", "--iterations", "10"]),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", required=True)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--ablation_iterations", type=int, default=5)
    p.add_argument("--output_dir", default=str(HERE / "ablation_runs"))
    p.add_argument("--only", nargs="+", help="Run only these configuration ids")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_slices_per_volume", type=int, default=None)
    p.add_argument("--crop_size", type=int, default=320)
    args, passthrough = p.parse_known_args()  # extra args (e.g. --epochs 20) go to training

    out = Path(args.output_dir)
    rows = []
    for cid, label, flags in configs(args.ablation_iterations):
        if args.only and cid not in args.only:
            continue
        run_dir = out / cid
        common = ["--splits", args.splits, "--acceleration", str(args.acceleration),
                  "--workers", str(args.workers), "--device", args.device, "--crop_size", str(args.crop_size)]
        if args.max_slices_per_volume:
            common += ["--max_slices_per_volume", str(args.max_slices_per_volume)]
        print(f"\n===== {label} =====")
        subprocess.run([sys.executable, str(TRAIN), *common, *flags, "--output_dir", str(run_dir / "train"),
                        *passthrough], check=True)
        subprocess.run([sys.executable, str(TEST), *common, "--checkpoint", str(run_dir / "train" / "best_model.pt"),
                        "--output_dir", str(run_dir / "test"), "--n_examples", "1"], check=True)
        s = json.loads((run_dir / "test" / f"{args.acceleration}x" / "summary.json").read_text())
        rows.append({"Configuration": label, "DC": "--no_dc" not in flags, "K-Dom.": "--no_kspace" not in flags,
                     "Comb. Loss": "combined" in flags, "SSIM": round(s["SSIM"]["mean"], 3),
                     "PSNR": round(s["PSNR"]["mean"], 1), "NMSE": round(s["NMSE"]["mean"], 4),
                     "Time (ms)": round(s["inference_ms_per_slice"], 1)})

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "table_IV_ablation.csv", index=False)
    (out / "table_IV_ablation.md").write_text(df.to_markdown(index=False))
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
