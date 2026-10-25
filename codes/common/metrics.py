"""Reconstruction metrics (Sec. V-A): SSIM, PSNR, NMSE and SNR.

All functions take real magnitude images (numpy). Following the fastMRI
convention, the data range for SSIM and PSNR is the maximum of the ground truth.

    NMSE = ||gt - pred||^2 / ||gt||^2
    PSNR = 10 log10(max(gt)^2 / MSE)
    SNR  = 10 log10(||gt||^2 / ||gt - pred||^2)  (= -10 log10 NMSE)
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(gt - pred) ** 2 / max(np.linalg.norm(gt) ** 2, 1e-20))


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(gt.max() ** 2 / mse))


def snr(gt: np.ndarray, pred: np.ndarray) -> float:
    err = float(np.sum((gt - pred) ** 2))
    if err == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(gt ** 2) / err))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt, pred = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
    return float(structural_similarity(gt, pred, data_range=gt.max() or 1.0))


def all_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    return {"SSIM": ssim(gt, pred), "PSNR": psnr(gt, pred), "NMSE": nmse(gt, pred), "SNR": snr(gt, pred)}


def summarise(rows: list[dict], keys=("SSIM", "PSNR", "NMSE", "SNR")) -> dict:
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if np.isfinite(r[k])], dtype=float)
        out[k] = {"mean": float(vals.mean()) if vals.size else float("nan"),
                  "std": float(vals.std()) if vals.size else float("nan")}
    return out
