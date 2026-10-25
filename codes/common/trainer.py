"""Training loop shared by every deep learning model (Sec. IV-F).

Adam (lr 1e-4, weight decay 1e-5), 50 epochs, batch size 4, learning rate halved
after 10 epochs without validation improvement, combined loss (Eq. 10).
The checkpoint with the best validation SSIM is kept.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import config, metrics
from common.cli import add_data_args, get_device, resolve_files, seed_everything
from common.data import make_loader
from common.losses import ReconstructionLoss
from common.models import build_model, count_parameters


def training_parser(model_name: str, default_output: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Train {model_name} for accelerated MRI reconstruction.")
    add_data_args(p, ("train", "val"))
    g = p.add_argument_group("training")
    g.add_argument("--output_dir", default=str(default_output))
    g.add_argument("--epochs", type=int, default=config.EPOCHS)
    g.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    g.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    g.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    g.add_argument("--loss", choices=["combined", "l2"], default="combined")
    g.add_argument("--no_pretrained_vgg", action="store_true",
                   help="Random VGG16 for the perceptual loss (offline testing only)")
    g.add_argument("--no_augment", action="store_true")
    if model_name == "UNet_DC":
        m = p.add_argument_group("U-Net + DC (ablation switches)")
        m.add_argument("--iterations", type=int, default=config.UNET_DC["iterations"], help="K unrolled iterations")
        m.add_argument("--base_channels", type=int, default=config.UNET_DC["base_channels"])
        m.add_argument("--no_dc", action="store_true", help="Remove the data-consistency layers")
        m.add_argument("--no_kspace", action="store_true", help="Remove the k-space (cross-domain) branch")
    return p


def model_kwargs_from_args(model_name: str, args) -> dict:
    if model_name != "UNet_DC":
        return {}
    return {"iterations": args.iterations, "base": args.base_channels,
            "use_dc": not args.no_dc, "use_kspace": not args.no_kspace}


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    rows = []
    for b in loader:
        recon = model(b["zero_filled"].to(device), b["kspace"].to(device), b["mask"].to(device), b["maps"].to(device))
        gt, pred = b["target"].abs().numpy(), recon.abs().float().cpu().numpy()
        rows += [metrics.all_metrics(gt[i], pred[i]) for i in range(len(gt))]
    model.train()
    return {k: v["mean"] for k, v in metrics.summarise(rows).items()}


def _plot_history(df: pd.DataFrame, out: Path, name: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.epoch, df.train_loss, label="training loss", lw=2)
    ax[0].set_xlabel("Epoch"), ax[0].set_ylabel("Loss"), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[1].plot(df.epoch, df.val_SSIM, label="validation SSIM", lw=2, color="#2a7")
    ax[1].set_xlabel("Epoch"), ax[1].set_ylabel("SSIM"), ax[1].legend(), ax[1].grid(alpha=0.3)
    fig.suptitle(f"{name}: training history")
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=140)
    plt.close()


def run_training(model_name: str, args) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_files, val_files = resolve_files(args, "train"), resolve_files(args, "val")
    train_dl = make_loader(train_files, args.acceleration, args.batch_size, True, args.mask_type, args.workers,
                           args.maps_dir, args.max_slices_per_volume, augment=not args.no_augment,
                           crop_size=args.crop_size)
    val_dl = make_loader(val_files, args.acceleration, args.batch_size, False, args.mask_type, args.workers,
                         args.maps_dir, args.max_slices_per_volume, crop_size=args.crop_size)

    kwargs = model_kwargs_from_args(model_name, args)
    model = build_model(model_name, **kwargs).to(device)
    n_params = count_parameters(model)
    loss_fn = ReconstructionLoss(args.loss, pretrained_vgg=not args.no_pretrained_vgg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=config.LR_PLATEAU_FACTOR,
                                                       patience=config.LR_PLATEAU_PATIENCE)
    print(f"{model_name}: {n_params / 1e6:.2f}M parameters | {len(train_dl.dataset)} train / "
          f"{len(val_dl.dataset)} val slices | {args.acceleration}x | device {device}")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        for b in train_dl:
            zf, k, m, s, tgt = (b[x].to(device) for x in ("zero_filled", "kspace", "mask", "maps", "target"))
            recon = model(zf, k, m, s)
            loss, _ = loss_fn(recon, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val = validate(model, val_dl, device)
        sched.step(val["SSIM"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               **{f"val_{k}": v for k, v in val.items()}, "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.5f} | val SSIM {val['SSIM']:.4f} "
              f"PSNR {val['PSNR']:.2f} NMSE {val['NMSE']:.4f} | lr {row['lr']:.1e} | {row['seconds']}s")

        ckpt = {"model": model_name, "kwargs": kwargs, "state_dict": model.state_dict(),
                "acceleration": args.acceleration, "epoch": epoch, "val": val}
        torch.save(ckpt, out / "last_model.pt")
        if val["SSIM"] > best:
            best = val["SSIM"]
            torch.save(ckpt, out / "best_model.pt")

    _plot_history(pd.DataFrame(history), out, model_name)
    result = {"model": model_name, "params_M": round(n_params / 1e6, 3), "best_val_SSIM": best,
              "kwargs": kwargs, "args": {k: v for k, v in vars(args).items()}}
    (out / "training_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"Done. Best validation SSIM {best:.4f}. Checkpoint: {out / 'best_model.pt'}")
    return result
