"""Create small synthetic multi-coil volumes in fastMRI HDF5 format.

Useful for checking that the whole pipeline runs before downloading the real
fastMRI dataset (and used by the repository's automated tests). The images are
Shepp-Logan phantoms with smooth simulated coil sensitivities; they are NOT
knee MRI and results on them mean nothing clinically.

Usage (from codes/):
    python data_preparation/make_synthetic_volumes.py --output_dir data/synthetic --n_volumes 6
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def coil_maps(n_coils: int, h: int, w: int, rng) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    maps = []
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cy, cx = h / 2 + 0.6 * h * np.sin(angle), w / 2 + 0.6 * w * np.cos(angle)
        mag = np.exp(-(((yy - cy) / (0.7 * h)) ** 2 + ((xx - cx) / (0.7 * w)) ** 2))
        phase = np.exp(1j * (angle + 0.002 * rng.normal() * (xx + yy)))
        maps.append(mag * phase)
    maps = np.stack(maps)
    return maps / np.sqrt((np.abs(maps) ** 2).sum(0, keepdims=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_volumes", type=int, default=6)
    p.add_argument("--n_slices", type=int, default=4)
    p.add_argument("--n_coils", type=int, default=4)
    p.add_argument("--height", type=int, default=400, help="Readout size (cropped to 320 when loaded)")
    p.add_argument("--width", type=int, default=336, help="Phase-encoding size")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = shepp_logan_phantom()
    for v in range(args.n_volumes):
        ks = []
        for s in range(args.n_slices):
            img = resize(base, (int(args.height * 0.75), int(args.width * 0.75)), anti_aliasing=True)
            img = np.roll(img, rng.integers(-5, 6, size=2), axis=(0, 1)) * (0.8 + 0.4 * rng.random())
            canvas = np.zeros((args.height, args.width))
            t, l = (args.height - img.shape[0]) // 2, (args.width - img.shape[1]) // 2
            canvas[t:t + img.shape[0], l:l + img.shape[1]] = img
            phase = np.exp(1j * 0.3 * np.sin(np.linspace(0, np.pi, args.width)))[None]
            coils = coil_maps(args.n_coils, args.height, args.width, rng) * (canvas * phase)[None]
            k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coils, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
            k += 1e-4 * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape))
            ks.append(k)
        with h5py.File(out / f"synthetic_{v:03d}.h5", "w") as h:
            h.create_dataset("kspace", data=np.stack(ks).astype(np.complex64))
            h.attrs["acquisition"] = "SYNTHETIC"
    print(f"Wrote {args.n_volumes} synthetic volumes to {out}")


if __name__ == "__main__":
    main()
