"""Combined reconstruction loss (Eq. 10):

    L = alpha * L2 + beta * L_SSIM + gamma * L_perceptual,   alpha=0.6, beta=0.3, gamma=0.1

L2 is computed on the complex image (real + imaginary), L_SSIM = 1 - SSIM and
L_perceptual (VGG16 features) on the magnitude image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import config


class SSIMLoss(nn.Module):
    """1 - SSIM with a 7x7 uniform window (fastMRI reference implementation)."""

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size, self.k1, self.k2 = win_size, k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        np_ = win_size ** 2
        self.cov_norm = np_ / (np_ - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B,1,H,W) magnitudes; data range = per-image max of target
        data_range = y.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        c1, c2 = (self.k1 * data_range) ** 2, (self.k2 * data_range) ** 2
        ux, uy = F.conv2d(x, self.w), F.conv2d(y, self.w)
        uxx, uyy, uxy = F.conv2d(x * x, self.w), F.conv2d(y * y, self.w), F.conv2d(x * y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
        return 1 - s.mean()


class PerceptualLoss(nn.Module):
    """MSE between VGG16 relu3_3 features of the magnitude images."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = x / x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std

    def forward(self, x, y):
        return F.mse_loss(self.features(self._prep(x)), self.features(self._prep(y)))


class ReconstructionLoss(nn.Module):
    """`kind='l2'` (ablation baseline) or `kind='combined'` (Eq. 10)."""

    def __init__(self, kind: str = "combined", alpha: float = config.LOSS_WEIGHTS["alpha"],
                 beta: float = config.LOSS_WEIGHTS["beta"], gamma: float = config.LOSS_WEIGHTS["gamma"],
                 pretrained_vgg: bool = True):
        super().__init__()
        if kind not in ("l2", "combined"):
            raise ValueError("loss must be 'l2' or 'combined'")
        self.kind, self.alpha, self.beta, self.gamma = kind, alpha, beta, gamma
        if kind == "combined":
            self.ssim = SSIMLoss()
            self.perceptual = PerceptualLoss(pretrained_vgg) if gamma > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """pred, target: complex (B,H,W)."""
        l2 = F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))
        if self.kind == "l2":
            return l2, {"l2": l2.item()}
        pm, tm = pred.abs().unsqueeze(1), target.abs().unsqueeze(1)
        l_ssim = self.ssim(pm, tm)
        total = self.alpha * l2 + self.beta * l_ssim
        parts = {"l2": l2.item(), "ssim": l_ssim.item()}
        if self.perceptual is not None:
            l_p = self.perceptual(pm, tm)
            total = total + self.gamma * l_p
            parts["perceptual"] = l_p.item()
        return total, parts
