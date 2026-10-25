"""Shared evaluation: per-slice metrics, summary, timing and example figures."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import metrics


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().abs().float().cpu().numpy()


def save_example_figure(sample: dict, recon: np.ndarray, name: str, path: Path, acceleration: int) -> None:
    """Ground truth | sampling mask | zero-filled | reconstruction | residual (like Fig. 3)."""
    gt = sample["gt"]
    zf = sample["zf"]
    mask = np.tile(sample["mask"][None, :], (gt.shape[0], 1))
    residual = np.abs(gt - recon)
    m = metrics.all_metrics(gt, recon)
    vmax = gt.max()
    panels = [
        ("Fully sampled", gt, "gray", (0, vmax)),
        (f"{acceleration}x mask (ACS centre)", mask, "viridis", (0, 1)),
        ("Zero-filled", zf, "gray", (0, vmax)),
        (f"{name}\nSSIM {m['SSIM']:.3f}  PSNR {m['PSNR']:.1f} dB", recon, "gray", (0, vmax)),
        ("Residual |GT - recon| (x5)", np.clip(residual * 5, 0, vmax), "inferno", (0, vmax)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.3))
    for ax, (title, img, cmap, lim) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


@torch.no_grad()
def evaluate(recon_fn: Callable[[dict], torch.Tensor], loader, out_dir: Path, name: str,
             acceleration: int, n_params: int | None = None, n_examples: int = 3,
             device: torch.device | None = None) -> dict:
    """Run `recon_fn(batch) -> complex (B,H,W)` over a loader and write results to out_dir."""
    out_dir = Path(out_dir)
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    best_per_volume = {}  # volume -> (score, sample, prediction); score = mean signal of the ground truth

    for batch in loader:
        if device is not None:
            for k in ("kspace", "mask", "maps", "target", "zero_filled"):
                batch[k] = batch[k].to(device)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        recon = recon_fn(batch)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000 / recon.shape[0])

        gt, pred, zf = _to_np(batch["target"]), _to_np(recon), _to_np(batch["zero_filled"])
        masks = batch["mask"].cpu().numpy().reshape(recon.shape[0], -1)
        for i in range(pred.shape[0]):
            row = {"volume": batch["fname"][i], "slice": int(batch["slice"][i]),
                   **metrics.all_metrics(gt[i], pred[i])}
            row.update({f"zero_filled_{k}": v for k, v in metrics.all_metrics(gt[i], zf[i]).items()})
            rows.append(row)
            # Edge slices of a knee volume are mostly noise: keep the most anatomy-rich slice per volume
            score = float(gt[i].mean())
            vol = batch["fname"][i]
            if n_examples and (vol not in best_per_volume or score > best_per_volume[vol][0]):
                best_per_volume[vol] = (score, {"gt": gt[i], "zf": zf[i], "mask": masks[i]}, pred[i])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_slice_metrics.csv", index=False)
    # skip warm-up batch for timing when possible
    t = np.array(times[1:] if len(times) > 1 else times)
    summary = {
        "method": name,
        "acceleration": acceleration,
        "n_slices": int(len(df)),
        "n_volumes": int(df["volume"].nunique()) if len(df) else 0,
        **{k: v for k, v in metrics.summarise(rows).items()},
        "zero_filled": metrics.summarise([{k.replace("zero_filled_", ""): r[k] for k in r
                                           if k.startswith("zero_filled_")} for r in rows]),
        "params_M": round(n_params / 1e6, 3) if n_params else None,
        "inference_ms_per_slice": float(t.mean()) if t.size else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    examples = sorted(best_per_volume.values(), key=lambda e: -e[0])[:n_examples]
    for j, (_, sample, pred) in enumerate(examples):
        save_example_figure(sample, pred, name, out_dir / "examples" / f"example_{j + 1}.png", acceleration)

    print(f"\n{name} @ {acceleration}x on {summary['n_slices']} slices")
    for k in ("SSIM", "PSNR", "NMSE", "SNR"):
        print(f"  {k:<5} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    if summary["inference_ms_per_slice"] is not None:
        print(f"  time  {summary['inference_ms_per_slice']:.1f} ms / slice")
    print(f"Results written to {out_dir}")
    return summary
