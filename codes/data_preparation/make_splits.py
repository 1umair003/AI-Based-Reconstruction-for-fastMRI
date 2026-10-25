"""Split fastMRI volumes into train / validation / test (70 / 15 / 15 by volume, Sec. IV-F).

Splitting by volume keeps every slice of a patient scan in the same split.

Usage (from codes/):
    python data_preparation/make_splits.py --data_dirs /data/knee_multicoil_train /data/knee_multicoil_val \
        --output splits.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dirs", nargs="+", required=True, help="Folders containing fastMRI .h5 volumes")
    p.add_argument("--output", default="splits.json")
    p.add_argument("--fractions", nargs=3, type=float, default=config.SPLIT_FRACTIONS)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    files = sorted({str(f.resolve()) for d in args.data_dirs for f in Path(d).rglob("*.h5")})
    if len(files) < 3:
        sys.exit(f"Found {len(files)} volumes; need at least 3.")
    random.Random(args.seed).shuffle(files)
    n = len(files)
    n_train = int(round(args.fractions[0] * n))
    n_val = max(1, int(round(args.fractions[1] * n)))
    n_train = min(n_train, n - n_val - 1)
    splits = {"train": files[:n_train], "val": files[n_train:n_train + n_val], "test": files[n_train + n_val:]}
    Path(args.output).write_text(json.dumps(splits, indent=2))
    print(" | ".join(f"{k}: {len(v)} volumes" for k, v in splits.items()) + f"  ->  {args.output}")


if __name__ == "__main__":
    main()
