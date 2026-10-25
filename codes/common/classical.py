"""Classical reconstruction baselines (Sec. IV-B, IV-C).

* zero_filled : A^H y (inverse FFT of the under-sampled k-space, SENSE-combined)
* cs_tv       : compressed sensing with total variation, solved with FISTA (Eqs. 4-5)
* sense       : CG-SENSE, Tikhonov-regularised least squares (Eq. 6)
* grappa      : k-space interpolation with weights fitted on the ACS lines (Eq. 7)

All functions take one slice (batch size 1) as torch tensors from the data loader
and return the complex SENSE-combined image (1, H, W).
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

from common.mri_ops import adjoint_op, forward_op, ifft2c_np


# ------------------------------------------------------------------ zero-filled
def zero_filled(b: dict) -> torch.Tensor:
    return adjoint_op(b["kspace"], b["maps"], b["mask"])


# ------------------------------------------------------------------ CS-TV
def _prox_tv(x: torch.Tensor, weight: float) -> torch.Tensor:
    z = x[0].cpu().numpy()
    re = denoise_tv_chambolle(z.real, weight=weight)
    im = denoise_tv_chambolle(z.imag, weight=weight)
    return torch.from_numpy((re + 1j * im).astype(np.complex64))[None].to(x.device)


def cs_tv(b: dict, lam: float = 0.003, iterations: int = 50, step: float = 1.0) -> torch.Tensor:
    """FISTA for  min_x 1/2 ||Ax - y||^2 + lam * TV(x).

    ||A|| <= 1 with normalised sensitivity maps, so step = 1 satisfies alpha < 2/||A||^2.
    """
    y, maps, mask = b["kspace"], b["maps"], b["mask"]
    x = adjoint_op(y, maps, mask)
    z, t = x.clone(), 1.0
    for _ in range(iterations):
        grad = adjoint_op(forward_op(z, maps, mask) - y, maps, mask)
        x_new = _prox_tv(z - step * grad, lam * step)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


# ------------------------------------------------------------------ SENSE
def sense(b: dict, reg: float = 1e-3, iterations: int = 30, tol: float = 1e-6) -> torch.Tensor:
    """Conjugate-gradient solution of (A^H A + reg I) x = A^H y."""
    y, maps, mask = b["kspace"], b["maps"], b["mask"]
    normal = lambda v: adjoint_op(forward_op(v, maps, mask), maps, mask) + reg * v
    rhs = adjoint_op(y, maps, mask)
    x = torch.zeros_like(rhs)
    r = rhs - normal(x)
    p = r.clone()
    rs = (r.conj() * r).sum().real
    for _ in range(iterations):
        ap = normal(p)
        alpha = rs / (p.conj() * ap).sum().real.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r.conj() * r).sum().real
        if rs_new.sqrt() < tol * rhs.abs().norm():
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ------------------------------------------------------------------ GRAPPA
def _grappa_fill(k: np.ndarray, mask: np.ndarray, acs: slice, kx: int = 3, reg: float = 1e-4) -> np.ndarray:
    """Fill missing phase-encoding columns of multi-coil k-space (C, H, W).

    For each missing column j the source points are the nearest acquired columns
    on either side (p < j < n) over a kx-wide readout window in every coil.
    Weights are fitted on the ACS block separately for each (j - p, n - j)
    geometry, so equispaced and random Cartesian masks are both supported.
    """
    c, h, w = k.shape
    acquired = np.flatnonzero(mask)
    out = k.copy()
    half = kx // 2
    pad = np.pad(k, ((0, 0), (half, half), (0, 0)))
    acs_cols = np.arange(acs.start, acs.stop)
    weights = {}

    def sources(kpad, cols_a, cols_b):
        # (H, C * kx * 2) source matrix for every readout row
        blocks = []
        for col in (cols_a, cols_b):
            for d in range(kx):
                blocks.append(kpad[:, d:d + h, col].transpose(1, 0))  # (H, C)
        return np.concatenate(blocks, axis=1)

    for j in range(w):
        if mask[j]:
            continue
        left, right = acquired[acquired < j], acquired[acquired > j]
        if not len(left) or not len(right):
            continue
        p, n = left[-1], right[0]
        key = (j - p, n - j)
        if key not in weights:
            src, tgt = [], []
            for t in acs_cols:
                if t - key[0] >= acs.start and t + key[1] < acs.stop:
                    src.append(sources(pad, t - key[0], t + key[1]))
                    tgt.append(k[:, :, t].T)
            if not src:
                weights[key] = None
            else:
                s_mat, t_mat = np.concatenate(src), np.concatenate(tgt)
                ahs = s_mat.conj().T @ s_mat
                lam = reg * np.trace(ahs).real / ahs.shape[0]
                weights[key] = np.linalg.solve(ahs + lam * np.eye(ahs.shape[0]), s_mat.conj().T @ t_mat)
        wts = weights[key]
        if wts is not None:
            out[:, :, j] = (sources(pad, p, n) @ wts).T
    return out


def grappa(b: dict, acs_lines: int = 24, kx: int = 3) -> torch.Tensor:
    y = b["kspace"][0].cpu().numpy()
    mask = b["mask"].reshape(-1).cpu().numpy() > 0
    w = y.shape[-1]
    start = (w - acs_lines + 1) // 2
    filled = _grappa_fill(y, mask, slice(start, start + acs_lines), kx)
    maps = b["maps"][0].cpu().numpy()
    img = (np.conj(maps) * ifft2c_np(filled)).sum(0)
    return torch.from_numpy(img.astype(np.complex64))[None]


METHODS = {"Zero_Filled": zero_filled, "CS_TV": cs_tv, "SENSE": sense, "GRAPPA": grappa}
