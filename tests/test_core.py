"""Fast checks of the core pieces: masks, MRI operators, metrics, models and baselines.

Run from the repository root:
    python tests/test_core.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
sys.path.insert(0, str(CODES))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import metrics  # noqa: E402
from common.classical import METHODS  # noqa: E402
from common.data import FastMRISliceDataset, list_volumes  # noqa: E402
from common.masks import make_mask  # noqa: E402
from common.models import build_model, count_parameters  # noqa: E402
from common.mri_ops import adjoint_op, data_consistency, forward_op  # noqa: E402


def check_masks():
    for kind in ("random", "equispaced"):
        for acc in (2, 4, 8):
            m = make_mask(kind, 320, acc, 24, seed_key="test")
            assert m[148:172].all(), "ACS block must be fully sampled"
            if kind == "random":
                assert m.sum() == round(320 / acc)
    assert (make_mask("random", 320, 4, 24, "a") == make_mask("random", 320, 4, 24, "a")).all()


def check_metrics():
    rng = np.random.default_rng(0)
    gt = rng.random((64, 64))
    noisy = gt + rng.normal(0, 0.05, gt.shape)
    assert abs(metrics.ssim(gt, gt) - 1) < 1e-9
    assert metrics.nmse(gt, gt) == 0
    assert metrics.ssim(gt, noisy) < 1
    assert abs(metrics.snr(gt, noisy) + 10 * np.log10(metrics.nmse(gt, noisy))) < 1e-9


def check_operators(batch):
    x, y, s, m = batch["target"][None], batch["kspace"][None], batch["maps"][None], batch["mask"][None]
    a, b = torch.randn_like(x), torch.randn_like(y)
    lhs = (forward_op(a, s, m).conj() * b).sum()
    rhs = (a.conj() * adjoint_op(b, s, m)).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-4, "A and A^H must be adjoint"
    out = data_consistency(torch.zeros_like(x), y, s, m, None)
    assert torch.allclose(forward_op(out, s, m), y * m, atol=1e-2 * y.abs().max())


def check_models(batch):
    b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in batch.items()}
    for name in ("Cascade_Net", "KIKI_Net", "UNet", "PD_Net", "UNet_DC"):
        kw = {"iterations": 2} if name == "UNet_DC" else {}
        model = build_model(name, **kw).eval()
        with torch.no_grad():
            out = model(b["zero_filled"], b["kspace"], b["mask"], b["maps"])
        assert out.shape == b["target"].shape and out.is_complex(), name
        assert torch.isfinite(out.abs()).all(), name
        assert count_parameters(model) > 0


def check_augmentation_keeps_maps_aligned(batch):
    """Flips / rotations must be applied identically to coil images and sensitivity maps."""
    import random

    maps = batch["maps"].numpy()
    coils = maps * batch["target"].numpy()[None]
    for seed in range(4):
        c, m = FastMRISliceDataset._augment([coils, maps], random.Random(seed))
        x = (np.conj(m) * c).sum(0)
        assert np.linalg.norm(c - m * x) / np.linalg.norm(c) < 1e-3, "maps misaligned after augmentation"


def check_baselines(batch):
    b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in batch.items()}
    gt = b["target"][0].abs().numpy()
    zf = metrics.nmse(gt, METHODS["Zero_Filled"](b)[0].abs().numpy())
    for name in ("SENSE", "GRAPPA"):
        err = metrics.nmse(gt, METHODS[name](b)[0].abs().numpy())
        assert err < zf, f"{name} should beat zero-filling ({err:.4f} vs {zf:.4f})"


def main():
    check_masks()
    check_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(CODES / "data_preparation" / "make_synthetic_volumes.py"),
                        "--output_dir", tmp, "--n_volumes", "1", "--n_slices", "1"], check=True)
        ds = FastMRISliceDataset(list_volumes(tmp), 4, "equispaced", crop_size=64)
        batch = ds[0]
        check_operators(batch)
        check_models(batch)
        check_augmentation_keeps_maps_aligned(batch)
        check_baselines(batch)
    print("All core checks passed.")


if __name__ == "__main__":
    main()
