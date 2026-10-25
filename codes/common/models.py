"""Reconstruction networks compared in the paper (Table II).

Every model has the same interface:

    model(zero_filled, kspace, mask, maps) -> complex image (B, H, W)

* Cascade_Net : deep cascade of CNNs with data consistency (Schlemper et al. [10])
* KIKI_Net    : alternating k-space / image-domain CNNs (Eo et al. [8])
* UNet        : residual U-Net on the zero-filled image, no data consistency [16]
* PD_Net      : learned primal-dual unrolled reconstruction (Table II "PD-Net")
* UNet_DC     : proposed U-Net with K iterative data-consistency layers (Eqs. 8-9)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import config
from common.mri_ops import (
    adjoint_op, chan_to_complex, complex_to_chan, data_consistency, fft2c, forward_op, ifft2c,
)


# ================================================================== building blocks
def _norm_input(x: torch.Tensor):
    """Per-sample normalisation of a (B,C,H,W) real tensor."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, mean, std


class ConvBlock(nn.Module):
    """(3x3 conv -> GroupNorm -> LeakyReLU) x 2."""

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        g = min(groups, cout)
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UNetCore(nn.Module):
    """U-Net with `levels` encoder/decoder levels, 3x3 convs, GroupNorm and LeakyReLU."""

    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 32, levels: int = 4):
        super().__init__()
        self.down = nn.ModuleList([ConvBlock(in_ch, base)])
        ch = base
        for _ in range(levels - 1):
            self.down.append(ConvBlock(ch, ch * 2))
            ch *= 2
        self.bottleneck = ConvBlock(ch, ch * 2)
        self.up_t, self.up = nn.ModuleList(), nn.ModuleList()
        ch *= 2
        for _ in range(levels):
            self.up_t.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.up.append(ConvBlock(ch, ch // 2))
            ch //= 2
        self.out = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottleneck(x)
        for t, blk in zip(self.up_t, self.up):
            x = t(x)
            s = skips.pop()
            if x.shape[-2:] != s.shape[-2:]:
                x = F.pad(x, (0, s.shape[-1] - x.shape[-1], 0, s.shape[-2] - x.shape[-2]))
            x = blk(torch.cat([x, s], dim=1))
        return self.out(x)


class ImageDenoiser(nn.Module):
    """Residual complex-image denoiser f_theta built on the U-Net."""

    def __init__(self, base: int, levels: int):
        super().__init__()
        self.unet = UNetCore(2, 2, base, levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = complex_to_chan(x)
        n, mean, std = _norm_input(c)
        return x + chan_to_complex(self.unet(n) * std)  # residual, rescaled to the input


def plain_cnn(cin: int, cout: int, filters: int, layers: int) -> nn.Sequential:
    mods = [nn.Conv2d(cin, filters, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(layers - 2):
        mods += [nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(inplace=True)]
    mods.append(nn.Conv2d(filters, cout, 3, padding=1))
    return nn.Sequential(*mods)


class KSpaceRefiner(nn.Module):
    """Small CNN acting on the k-space of the current image estimate (cross-domain branch)."""

    def __init__(self, filters: int = 32, layers: int = 5):
        super().__init__()
        self.net = plain_cnn(2, 2, filters, layers)

    def forward(self, x):
        k = fft2c(x)
        c = complex_to_chan(k)
        n, mean, std = _norm_input(c)
        return ifft2c(k + chan_to_complex(self.net(n) * std))


# ================================================================== models
class UNetRecon(nn.Module):
    """U-Net baseline: zero-filled image -> residual U-Net, no data consistency."""

    def __init__(self, base: int = config.UNET_DC["base_channels"], levels: int = config.UNET_DC["levels"]):
        super().__init__()
        self.denoiser = ImageDenoiser(base, levels)

    def forward(self, zero_filled, kspace, mask, maps):
        return self.denoiser(zero_filled)


class UNetDC(nn.Module):
    """Proposed U-Net + iterative data consistency (Eqs. 8-9).

    x^{k+1} = DC( f_theta(x^k), y, A, lambda ),  k = 0..K-1, shared weights (MoDL-style).

    Ablation switches (Table IV):
        use_dc      - closed-form DC layer after every iteration
        use_kspace  - k-space CNN branch before the image U-Net (cross-domain, "+KD")
    """

    def __init__(self, iterations: int = config.UNET_DC["iterations"], base: int = config.UNET_DC["base_channels"],
                 levels: int = config.UNET_DC["levels"], use_dc: bool = True, use_kspace: bool = True,
                 initial_lambda: float = config.UNET_DC["initial_lambda"]):
        super().__init__()
        self.iterations, self.use_dc, self.use_kspace = iterations, use_dc, use_kspace
        self.denoiser = ImageDenoiser(base, levels)
        self.kspace = KSpaceRefiner() if use_kspace else None
        # lambda = softplus(raw) > 0, initialised to `initial_lambda`
        self.raw_lambda = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(initial_lambda)))))

    @property
    def lam(self):
        return F.softplus(self.raw_lambda)

    def forward(self, zero_filled, kspace, mask, maps):
        x = zero_filled
        for _ in range(self.iterations):
            if self.kspace is not None:
                x = self.kspace(x)
            x = self.denoiser(x)
            if self.use_dc:
                x = data_consistency(x, kspace, maps, mask, self.lam)
        return x


class CascadeNet(nn.Module):
    """Deep cascade (D5-C5): 5 CNNs of 5 conv layers, each followed by data consistency."""

    def __init__(self, cascades: int = 5, layers: int = 5, filters: int = 118, use_dc: bool = True):
        super().__init__()
        self.cnns = nn.ModuleList(plain_cnn(2, 2, filters, layers) for _ in range(cascades))
        self.use_dc = use_dc

    def forward(self, zero_filled, kspace, mask, maps):
        x = zero_filled
        for cnn in self.cnns:
            c = complex_to_chan(x)
            n, mean, std = _norm_input(c)
            x = x + chan_to_complex(cnn(n) * std)
            if self.use_dc:
                x = data_consistency(x, kspace, maps, mask, None)
        return x


class KIKINet(nn.Module):
    """KIKI-Net: [k-space CNN -> image CNN -> DC] repeated `iterations` times."""

    def __init__(self, iterations: int = 2, layers: int = 6, filters: int = 128):
        super().__init__()
        self.k_nets = nn.ModuleList(plain_cnn(2, 2, filters, layers) for _ in range(iterations))
        self.i_nets = nn.ModuleList(plain_cnn(2, 2, filters, layers) for _ in range(iterations))

    def forward(self, zero_filled, kspace, mask, maps):
        x = zero_filled
        for kn, inet in zip(self.k_nets, self.i_nets):
            k = fft2c(x)
            c = complex_to_chan(k)
            n, _, std = _norm_input(c)
            x = ifft2c(k + chan_to_complex(kn(n) * std))
            c = complex_to_chan(x)
            n, _, std = _norm_input(c)
            x = x + chan_to_complex(inet(n) * std)
            x = data_consistency(x, kspace, maps, mask, None)
        return x


class PDNet(nn.Module):
    """Learned primal-dual reconstruction with the multi-coil operator A = P F S.

    Dual update (per coil, in k-space):  h <- h + Gamma([h, A f, y])
    Primal update (image domain):        f <- f + Lambda([f, A^H h])
    """

    def __init__(self, iterations: int = 10, filters: int = 60, layers: int = 4):
        super().__init__()
        self.iterations = iterations
        self.dual = nn.ModuleList(plain_cnn(6, 2, filters, layers) for _ in range(iterations))
        self.primal = nn.ModuleList(plain_cnn(4, 2, filters, layers) for _ in range(iterations))

    @staticmethod
    def _per_coil(t):  # (B,C,H,W) complex -> (B*C,2,H,W)
        b, c, h, w = t.shape
        return torch.stack([t.real, t.imag], dim=2).reshape(b * c, 2, h, w)

    def forward(self, zero_filled, kspace, mask, maps):
        b, c, h, w = kspace.shape
        f = zero_filled
        hdual = torch.zeros_like(kspace)
        scale = kspace.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        for d_net, p_net in zip(self.dual, self.primal):
            af = forward_op(f, maps, mask)
            inp = torch.cat([self._per_coil(hdual / scale), self._per_coil(af / scale),
                             self._per_coil(kspace / scale)], dim=1)
            upd = d_net(inp).reshape(b, c, 2, h, w)
            hdual = hdual + mask * torch.complex(upd[:, :, 0], upd[:, :, 1]) * scale
            ah = adjoint_op(hdual, maps, mask)
            fs = f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            inp = torch.cat([complex_to_chan(f / fs), complex_to_chan(ah / fs)], dim=1)
            f = f + chan_to_complex(p_net(inp)) * fs
        return f


# ================================================================== factory
def build_model(name: str, **kwargs) -> nn.Module:
    builders = {
        "Cascade_Net": CascadeNet,
        "KIKI_Net": KIKINet,
        "UNet": UNetRecon,
        "PD_Net": PDNet,
        "UNet_DC": UNetDC,
    }
    if name not in builders:
        raise ValueError(f"Unknown model {name}; choose from {list(builders)}")
    return builders[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
