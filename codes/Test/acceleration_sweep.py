"""SSIM across acceleration factors 2x, 4x, 6x, 8x for every method (Table III).

Evaluates each trained network (best checkpoint) and each classical baseline
at every acceleration, then writes a Table III-style summary.

By default one checkpoint per network is used at all accelerations. To use a
model trained separately for each factor, train with
`--acceleration X --output_dir outputs_Xx` and pass
`--checkpoint_template "training/{model}/outputs_{acc}x/best_model.pt"`.

Usage (from codes/):
    python Test/acceleration_sweep.py --splits splits.json
    python Test/acceleration_sweep.py --splits splits.json --skip_classical      # networks only
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
sys.path.insert(0, str(CODES))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import config  # noqa: E402
from common.classical import METHODS  # noqa: E402
from common.cli import get_device, seed_everything  # noqa: E402
from common.data import list_volumes, load_split, make_loader  # noqa: E402
from common.evaluation import evaluate  # noqa: E402
from common.models import count_parameters  # noqa: E402
from common.tester import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits")
    p.add_argument("--test_dir")
    p.add_argument("--accelerations", nargs="+", type=int, default=list(config.ACCELERATIONS))
    p.add_argument("--models", nargs="+", default=config.MODELS)
    p.add_argument("--classical", nargs="+", default=["Zero_Filled", "GRAPPA"],
                   help="Classical baselines to include (CS-TV is slow; add it explicitly)")
    p.add_argument("--skip_classical", action="store_true")
    p.add_argument("--checkpoint_template", default=str(CODES / "training" / "{model}" / "outputs" / "best_model.pt"))
    p.add_argument("--maps_dir", default=None)
    p.add_argument("--max_slices_per_volume", type=int, default=None)
    p.add_argument("--crop_size", type=int, default=config.CROP_SIZE)
    p.add_argument("--output_dir", default=str(HERE / "acceleration_sweep"))
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    seed_everything()
    device = get_device(args.device)
    files = list_volumes(args.test_dir) if args.test_dir else load_split(args.splits, "test")
    out = Path(args.output_dir)
    rows = []

    for acc in args.accelerations:
        for m in args.models:
            ckpt = Path(args.checkpoint_template.format(model=m, acc=acc))
            if not ckpt.exists():
                print(f"skip {m} @ {acc}x: {ckpt} not found")
                continue
            model, _ = load_model(ckpt, device)
            dl = make_loader(files, acc, 1, False, "random", args.workers, args.maps_dir, args.max_slices_per_volume,
                             crop_size=args.crop_size)
            s = evaluate(lambda b: model(b["zero_filled"], b["kspace"], b["mask"], b["maps"]), dl,
                         out / m / f"{acc}x", m.replace("_", "-"), acc, count_parameters(model), 1, device)
            rows.append({"method": m.replace("_", "-"), "acceleration": acc, "SSIM": s["SSIM"]["mean"],
                         "PSNR": s["PSNR"]["mean"], "NMSE": s["NMSE"]["mean"]})
        if not args.skip_classical:
            for c in args.classical:
                mask = "equispaced" if c == "GRAPPA" else "random"
                dl = make_loader(files, acc, 1, False, mask, 0, args.maps_dir, args.max_slices_per_volume,
                             crop_size=args.crop_size)
                with torch.no_grad():
                    s = evaluate(METHODS[c], dl, out / c / f"{acc}x", c.replace("_", "-"), acc, None, 1)
                rows.append({"method": c.replace("_", "-"), "acceleration": acc, "SSIM": s["SSIM"]["mean"],
                             "PSNR": s["PSNR"]["mean"], "NMSE": s["NMSE"]["mean"]})

    if not rows:
        sys.exit("Nothing evaluated.")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep_long.csv", index=False)
    table = df.pivot(index="method", columns="acceleration", values="SSIM")
    table.columns = [f"{c}x SSIM" for c in table.columns]
    table = table.sort_values(table.columns[0])
    table.round(3).to_csv(out / "table_III_ssim_vs_acceleration.csv")
    (out / "table_III_ssim_vs_acceleration.md").write_text(table.round(3).to_markdown())
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table.round(3).to_string())
    _plot(df, out)


def _plot(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for m, g in df.groupby("method"):
        g = g.sort_values("acceleration")
        plt.plot(g.acceleration, g.SSIM, marker="o", lw=2, label=m)
    plt.xlabel("Acceleration factor"), plt.ylabel("SSIM"), plt.grid(alpha=0.3), plt.legend(fontsize=8)
    plt.title("SSIM vs acceleration")
    plt.tight_layout()
    plt.savefig(out / "ssim_vs_acceleration.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
