"""fastMRI knee data loading (Sec. IV-F, V-A).

Each fastMRI HDF5 volume stores raw k-space under the key `kspace`:
    multi-coil  : (slices, coils, H, W) complex64
    single-coil : (slices, H, W)        complex64

Per slice this loader:
 1. inverse-FFTs each coil, centre-crops to 320 x 320, and FFTs back, giving a
    fully sampled k-space that matches the cropped field of view;
 2. (train only) augments with random flips, 90-degree rotations and additive
    complex Gaussian noise;
 3. estimates coil sensitivity maps from the 24-line ACS region
    (or loads ESPIRiT maps written by data_preparation/precompute_espirit_maps.py);
 4. builds the SENSE-combined complex target image;
 5. applies a Cartesian mask (random for training, fixed per slice for val/test);
 6. normalises everything by the maximum of the zero-filled image.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import config
from common.masks import make_mask
from common.mri_ops import acs_sensitivity_maps, center_crop, fft2c_np, ifft2c_np


def load_split(splits_file: str | Path, split: str) -> list[str]:
    data = json.loads(Path(splits_file).read_text())
    if split not in data:
        raise KeyError(f"split '{split}' not in {splits_file} (has {list(data)})")
    return data[split]


def list_volumes(path: str | Path) -> list[str]:
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        raise ValueError("Pass a folder of .h5 files, or use --splits with --split.")
    files = sorted(str(f) for f in p.rglob("*.h5")) if p.is_dir() else [str(p)]
    if not files:
        raise FileNotFoundError(f"No .h5 volumes found in {path}")
    return files


class FastMRISliceDataset(Dataset):
    def __init__(self, files: list[str], acceleration: int = config.DEFAULT_ACCELERATION,
                 mask_type: str = "random", acs_lines: int = config.ACS_LINES,
                 crop_size: int = config.CROP_SIZE, train: bool = False, augment: bool = True,
                 maps_dir: str | None = None, max_slices_per_volume: int | None = None):
        self.files = list(files)
        self.acceleration = acceleration
        self.mask_type = mask_type
        self.acs_lines = acs_lines
        self.crop_size = crop_size
        self.train = train
        self.augment = augment and train
        self.maps_dir = Path(maps_dir) if maps_dir else None
        self.index = []
        for f in self.files:
            with h5py.File(f, "r") as h:
                n = h["kspace"].shape[0]
            slices = range(n)
            if max_slices_per_volume and n > max_slices_per_volume:
                mid = n // 2
                half = max_slices_per_volume // 2
                slices = range(mid - half, mid - half + max_slices_per_volume)
            self.index += [(f, s) for s in slices]

    def __len__(self):
        return len(self.index)

    # -------------------------------------------------------------- helpers
    def _load_kspace(self, fname: str, sl: int) -> np.ndarray:
        with h5py.File(fname, "r") as h:
            k = h["kspace"][sl]
        if k.ndim == 2:  # single-coil
            k = k[None]
        return k.astype(np.complex64)

    def _stored_maps(self, fname: str, sl: int) -> np.ndarray | None:
        """Pre-computed ESPIRiT maps for this slice, or None to estimate from the ACS lines."""
        if self.maps_dir is not None:
            mf = self.maps_dir / Path(fname).name
            if mf.exists():
                with h5py.File(mf, "r") as h:
                    maps = h["maps"][sl].astype(np.complex64)
                return center_crop(maps, self.crop_size)
        return None

    @staticmethod
    def _augment(arrays: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
        """Apply the same random flips / 90-degree rotation to every array (images and maps)."""
        flip_x = rng.random() < config.AUG_FLIP_PROB
        flip_y = rng.random() < config.AUG_FLIP_PROB
        h, w = arrays[0].shape[-2:]
        rot = rng.choice([1, 3]) if (rng.random() < config.AUG_ROT90_PROB and h == w) else 0
        out = []
        for a in arrays:
            if flip_x:
                a = a[..., ::-1]
            if flip_y:
                a = a[..., ::-1, :]
            if rot:
                a = np.rot90(a, k=rot, axes=(-2, -1))
            out.append(np.ascontiguousarray(a))
        return out

    # -------------------------------------------------------------- item
    def __getitem__(self, i):
        fname, sl = self.index[i]
        rng = random.Random() if self.train else random.Random(f"{Path(fname).name}:{sl}")

        coil_imgs = center_crop(ifft2c_np(self._load_kspace(fname, sl)), self.crop_size)
        maps = self._stored_maps(fname, sl)
        if self.augment:
            if maps is not None:
                coil_imgs, maps = self._augment([coil_imgs, maps], rng)
            else:
                (coil_imgs,) = self._augment([coil_imgs], rng)
        k_full = fft2c_np(coil_imgs).astype(np.complex64)
        if self.augment and config.AUG_NOISE_STD > 0:
            sigma = config.AUG_NOISE_STD * np.abs(k_full).mean()
            k_full = k_full + (sigma * (np.random.randn(*k_full.shape)
                                        + 1j * np.random.randn(*k_full.shape))).astype(np.complex64)
            coil_imgs = ifft2c_np(k_full)

        if maps is None:
            maps = acs_sensitivity_maps(k_full, self.acs_lines)
        target = (np.conj(maps) * coil_imgs).sum(0)

        width = k_full.shape[-1]
        seed_key = None if self.train else f"{Path(fname).name}:{sl}"
        mask = make_mask(self.mask_type, width, self.acceleration, self.acs_lines, seed_key).astype(np.float32)
        k_us = k_full * mask[None, None, :]
        zf = (np.conj(maps) * ifft2c_np(k_us)).sum(0)

        scale = float(np.abs(zf).max()) or 1.0
        return {
            "kspace": torch.from_numpy((k_us / scale).astype(np.complex64)),
            "mask": torch.from_numpy(mask).view(1, 1, -1),
            "maps": torch.from_numpy(maps.astype(np.complex64)),
            "target": torch.from_numpy((target / scale).astype(np.complex64)),
            "zero_filled": torch.from_numpy((zf / scale).astype(np.complex64)),
            "fname": Path(fname).name,
            "slice": sl,
        }


def make_loader(files, acceleration, batch_size, train, mask_type="random", workers=2,
                maps_dir=None, max_slices_per_volume=None, augment=True,
                crop_size: int = config.CROP_SIZE) -> DataLoader:
    ds = FastMRISliceDataset(files, acceleration, mask_type, crop_size=crop_size, train=train, augment=augment,
                             maps_dir=maps_dir, max_slices_per_volume=max_slices_per_volume)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      drop_last=train and len(ds) > batch_size, pin_memory=torch.cuda.is_available())
