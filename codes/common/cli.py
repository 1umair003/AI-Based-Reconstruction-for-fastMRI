"""Command-line helpers shared by the training, testing and baseline scripts."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from common import config
from common.data import list_volumes, load_split


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_data_args(p: argparse.ArgumentParser, splits: tuple[str, ...]) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--splits", help="splits.json from data_preparation/make_splits.py")
    for s in splits:
        g.add_argument(f"--{s}_dir", help=f"Folder of fastMRI .h5 volumes for {s} (instead of --splits)")
    g.add_argument("--acceleration", type=int, default=config.DEFAULT_ACCELERATION)
    g.add_argument("--mask_type", choices=["random", "equispaced"], default="random")
    g.add_argument("--maps_dir", default=None, help="ESPIRiT maps from precompute_espirit_maps.py (optional)")
    g.add_argument("--crop_size", type=int, default=config.CROP_SIZE,
                   help="Image size after centre crop (320 in the paper; smaller is faster for experiments)")
    g.add_argument("--max_slices_per_volume", type=int, default=None,
                   help="Use only the central N slices of each volume (faster experiments)")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int, default=config.SEED)


def resolve_files(args, split: str) -> list[str]:
    folder = getattr(args, f"{split}_dir", None)
    if folder:
        return list_volumes(folder)
    if args.splits:
        return load_split(args.splits, split)
    raise SystemExit(f"Provide --splits or --{split}_dir")
