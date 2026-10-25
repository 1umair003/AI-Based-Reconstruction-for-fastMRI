"""Side-by-side comparison of every method on one slice (like Fig. 3).

Rows: k-space sampling mask, reconstruction, residual |GT - recon|.
Columns: fully sampled ground truth + each method.

Usage (from codes/):
    python Test/compare_methods.py --splits splits.json --acceleration 4
    python Test/compare_methods.py --splits splits.json --volume file1000001.h5 --slice 18
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
sys.path.insert(0, str(CODES))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import config, metrics  # noqa: E402
from common.classical import METHODS  # noqa: E402
from common.cli import get_device  # noqa: E402
from common.data import FastMRISliceDataset, list_volumes, load_split  # noqa: E402
from common.mri_ops import fft2c_np  # noqa: E402
from common.tester import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits")
    p.add_argument("--test_dir")
    p.add_argument("--volume", help="File name of the volume to show (default: first test volume)")
    p.add_argument("--slice", type=int, default=None, help="Slice index (default: middle slice)")
    p.add_argument("--acceleration", type=int, default=config.DEFAULT_ACCELERATION)
    p.add_argument("--models", nargs="+", default=config.MODELS)
    p.add_argument("--classical", nargs="+", default=["Zero_Filled"])
    p.add_argument("--maps_dir", default=None)
    p.add_argument("--crop_size", type=int, default=config.CROP_SIZE)
    p.add_argument("--checkpoint_template", default=str(CODES / "training" / "{model}" / "outputs" / "best_model.pt"))
    p.add_argument("--output", default=str(HERE / "comparison.png"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    files = list_volumes(args.test_dir) if args.test_dir else load_split(args.splits, "test")
    if args.volume:
        files = [f for f in files if Path(f).name == args.volume] or sys.exit(f"{args.volume} not in test files")
    ds = FastMRISliceDataset(files[:1], args.acceleration, "random", crop_size=args.crop_size, maps_dir=args.maps_dir)
    idx = args.slice if args.slice is not None else len(ds) // 2
    b = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in ds[idx].items()}
    gt = b["target"][0].abs().numpy()
    mask_img = np.tile(b["mask"].reshape(-1).numpy()[None], (gt.shape[0], 1))

    recons = []
    for c in args.classical:
        with torch.no_grad():
            recons.append((c.replace("_", "-"), METHODS[c](b)[0].abs().numpy()))
    for m in args.models:
        ckpt = Path(args.checkpoint_template.format(model=m))
        if not ckpt.exists():
            print(f"skip {m}: no checkpoint at {ckpt}")
            continue
        model, _ = load_model(ckpt, device)
        with torch.no_grad():
            r = model(*(b[k].to(device) for k in ("zero_filled", "kspace", "mask", "maps")))
        recons.append((m.replace("_", "-"), r[0].abs().cpu().numpy()))

    n = len(recons) + 1
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8))
    vmax = gt.max()
    k_full = np.abs(fft2c_np(b["target"][0].numpy()))
    ax[0, 0].imshow(np.log1p(k_full / max(k_full.max(), 1e-12) * 1e4), cmap="viridis")
    ax[0, 0].set_title("Fully sampled\nk-space", fontsize=9)
    ax[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=vmax)
    ax[1, 0].set_title("Ground truth", fontsize=9)
    ax[2, 0].axis("off")
    for j, (name, rec) in enumerate(recons, start=1):
        m = metrics.all_metrics(gt, rec)
        ax[0, j].imshow(mask_img, cmap="viridis", vmin=0, vmax=1)
        ax[0, j].set_title(f"{name}\n{args.acceleration}x mask", fontsize=9)
        ax[1, j].imshow(rec, cmap="gray", vmin=0, vmax=vmax)
        ax[1, j].set_title(f"SSIM {m['SSIM']:.3f} | PSNR {m['PSNR']:.1f}", fontsize=8)
        ax[2, j].imshow(np.abs(gt - rec), cmap="inferno", vmin=0, vmax=vmax / 5)
        ax[2, j].set_title("Residual (x5)", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.output, dpi=140)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
