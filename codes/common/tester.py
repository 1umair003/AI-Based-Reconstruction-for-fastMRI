"""Test a trained deep learning model on held-out volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.cli import add_data_args, get_device, resolve_files, seed_everything
from common.data import make_loader
from common.evaluation import evaluate
from common.models import build_model, count_parameters


def test_parser(model_name: str, default_ckpt: Path, default_output: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Test {model_name}: SSIM, PSNR, NMSE, SNR, params and time.")
    add_data_args(p, ("test",))
    p.add_argument("--checkpoint", default=str(default_ckpt))
    p.add_argument("--output_dir", default=str(default_output))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_examples", type=int, default=3)
    return p


def load_model(checkpoint: str | Path, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], **ckpt["kwargs"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def run_test(model_name: str, args) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"{args.checkpoint} not found. Train the model first or pass --checkpoint.")
    model, ckpt = load_model(args.checkpoint, device)
    if ckpt["model"] != model_name:
        print(f"note: checkpoint is a {ckpt['model']} model")
    files = resolve_files(args, "test")
    loader = make_loader(files, args.acceleration, args.batch_size, False, args.mask_type, args.workers,
                         args.maps_dir, args.max_slices_per_volume, crop_size=args.crop_size)
    out = Path(args.output_dir) / f"{args.acceleration}x"
    return evaluate(lambda b: model(b["zero_filled"], b["kspace"], b["mask"], b["maps"]), loader, out,
                    model_name.replace("_", "-"), args.acceleration, count_parameters(model),
                    args.n_examples, device)
