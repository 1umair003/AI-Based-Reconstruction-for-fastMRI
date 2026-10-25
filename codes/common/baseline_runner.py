"""Runner shared by the classical baseline scripts."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch

from common.classical import METHODS
from common.cli import add_data_args, resolve_files, seed_everything
from common.data import make_loader
from common.evaluation import evaluate


def baseline_parser(name: str, default_output: Path, default_mask: str = "random") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{name.replace('_', '-')} reconstruction: SSIM, PSNR, NMSE, SNR.")
    add_data_args(p, ("test",))
    p.set_defaults(mask_type=default_mask, workers=0)
    p.add_argument("--output_dir", default=str(default_output))
    p.add_argument("--n_examples", type=int, default=3)
    return p


def run_baseline(name: str, args, **method_kwargs) -> dict:
    seed_everything(args.seed)
    files = resolve_files(args, "test")
    loader = make_loader(files, args.acceleration, 1, False, args.mask_type, args.workers,
                         args.maps_dir, args.max_slices_per_volume, crop_size=args.crop_size)
    fn = partial(METHODS[name], **method_kwargs)
    out = Path(args.output_dir) / f"{args.acceleration}x"
    with torch.no_grad():
        return evaluate(fn, loader, out, name.replace("_", "-"), args.acceleration, None, args.n_examples)
