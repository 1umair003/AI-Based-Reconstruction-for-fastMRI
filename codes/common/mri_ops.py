"""MRI forward model and helpers (Eqs. 1-3, 9).

Conventions
-----------
* k-space / coil images: complex tensors of shape (B, C, H, W)
* coil-combined image x: complex tensor (B, H, W)
* sensitivity maps S: complex (B, C, H, W)
* mask P: real {0, 1} tensor broadcastable to (B, 1, 1, W)

Forward operator   A x  = P F (S x)              (Eq. 3)
Adjoint            A^H y = sum_c conj(S_c) F^H(P y_c)

All FFTs are centred and orthonormal, as in the fastMRI reference code.
"""

from __future__ import annotations

import numpy as np
import torch

# ------------------------------------------------------------------ FFT (torch)
def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    k = torch.fft.ifft2(k, norm="ortho")
    return torch.fft.fftshift(k, dim=(-2, -1))


# ------------------------------------------------------------------ FFT (numpy)
def fft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c_np(k: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


# ------------------------------------------------------------------ coil operators
def expand(x: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Image (B,H,W) -> coil images (B,C,H,W)."""
    return maps * x.unsqueeze(1)


def reduce(coil_imgs: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Coil images (B,C,H,W) -> SENSE-combined image (B,H,W)."""
    return (maps.conj() * coil_imgs).sum(dim=1)


def forward_op(x: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask * fft2c(expand(x, maps))


def adjoint_op(y: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return reduce(ifft2c(mask * y), maps)


def rss(coil_imgs: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.sqrt((coil_imgs.abs() ** 2).sum(dim=dim))


def data_consistency(x: torch.Tensor, y: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor,
                     lam: torch.Tensor | float | None) -> torch.Tensor:
    """Closed-form data consistency in k-space (Eq. 9).

    At sampled locations k <- (lam * k + y) / (lam + 1); unsampled locations keep
    the network's prediction. lam = None gives hard DC (measured lines replaced).
    """
    k = fft2c(expand(x, maps))
    if lam is None:
        k_dc = (1 - mask) * k + mask * y
    else:
        k_dc = (1 - mask) * k + mask * (lam * k + y) / (lam + 1)
    return reduce(ifft2c(k_dc), maps)


# ------------------------------------------------------------------ complex <-> channels
def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    """(B,H,W) complex -> (B,2,H,W) real."""
    return torch.stack([x.real, x.imag], dim=1)


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B,2,H,W) real -> (B,H,W) complex."""
    return torch.complex(x[:, 0], x[:, 1])


def kspace_to_chan(k: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) complex -> (B,2C,H,W) real."""
    return torch.cat([k.real, k.imag], dim=1)


def chan_to_kspace(k: torch.Tensor) -> torch.Tensor:
    c = k.shape[1] // 2
    return torch.complex(k[:, :c], k[:, c:])


# ------------------------------------------------------------------ sensitivity maps (numpy)
def center_crop(x: np.ndarray, size: int) -> np.ndarray:
    h, w = x.shape[-2:]
    s = min(size, h), min(size, w)
    top, left = (h - s[0]) // 2, (w - s[1]) // 2
    return x[..., top:top + s[0], left:left + s[1]]


def acs_sensitivity_maps(kspace: np.ndarray, acs_lines: int, eps: float = 1e-8) -> np.ndarray:
    """Low-resolution sensitivity maps from the ACS region (fast, no extra dependencies).

    kspace: (C, H, W) fully sampled or under-sampled with a full ACS block.
    Returns (C, H, W) complex maps normalised so that sum |S_c|^2 = 1.
    """
    c, h, w = kspace.shape
    if c == 1:
        return np.ones_like(kspace)
    start = (w - acs_lines + 1) // 2
    low = np.zeros_like(kspace)
    low[..., start:start + acs_lines] = kspace[..., start:start + acs_lines]
    # Hann window along the readout and phase directions to suppress ringing
    win = np.outer(np.hanning(h), np.hanning(acs_lines + 2)[1:-1])
    low[..., start:start + acs_lines] *= win
    imgs = ifft2c_np(low)
    norm = np.sqrt((np.abs(imgs) ** 2).sum(0, keepdims=True)) + eps
    return (imgs / norm).astype(np.complex64)


def espirit_maps(kspace: np.ndarray, acs_lines: int) -> np.ndarray:
    """ESPIRiT sensitivity maps via SigPy (pip install sigpy)."""
    import sigpy.mri as mr

    if kspace.shape[0] == 1:
        return np.ones_like(kspace)
    return mr.app.EspiritCalib(kspace, calib_width=acs_lines, show_pbar=False).run().astype(np.complex64)
