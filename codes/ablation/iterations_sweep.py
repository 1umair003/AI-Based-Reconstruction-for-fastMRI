"""Effect of the number of unrolled iterations K (Table V).

Trains and tests the full U-Net + DC model for K in {1, 3, 5, 10, 15, 20}.

Usage (from codes/):
    python ablation/iterations_sweep.py --splits splits.json
    python ablation/iterations_sweep.py --splits splits.json --k_values 1 5 10 --epochs 20
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", required=True)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5, 10, 15, 20])
    p.add_argument("--output_dir", default=str(HERE / "iteration_runs"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_slices_per_volume", type=int, default=None)
    p.add_argument("--crop_size", type=int, default=320)
    args, passthrough = p.parse_known_args()

    out = Path(args.output_dir)
    rows = []
    for k in args.k_values:
        run = out / f"K{k}"
        common = ["--splits", args.splits, "--acceleration", str(args.acceleration),
                  "--workers", str(args.workers), "--device", args.device, "--crop_size", str(args.crop_size)]
        if args.max_slices_per_volume:
            common += ["--max_slices_per_volume", str(args.max_slices_per_volume)]
        print(f"\n===== K = {k} =====")
        subprocess.run([sys.executable, str(TRAIN), *common, "--iterations", str(k),
                        "--output_dir", str(run / "train"), *passthrough], check=True)
        subprocess.run([sys.executable, str(TEST), *common, "--checkpoint", str(run / "train" / "best_model.pt"),
                        "--output_dir", str(run / "test"), "--n_examples", "1"], check=True)
        s = json.loads((run / "test" / f"{args.acceleration}x" / "summary.json").read_text())
        rows.append({"K (Iterations)": k, "SSIM": round(s["SSIM"]["mean"], 3), "PSNR (dB)": round(s["PSNR"]["mean"], 1),
                     "NMSE": round(s["NMSE"]["mean"], 4), "Params (M)": s["params_M"],
                     "Time (ms)": round(s["inference_ms_per_slice"], 1)})

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "table_V_iterations.csv", index=False)
    (out / "table_V_iterations.md").write_text(df.to_markdown(index=False))
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
