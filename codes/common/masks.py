"""Cartesian k-space under-sampling masks.

Masks select phase-encoding columns (last axis). Both types keep a fully
sampled centre of `acs_lines` columns (24 in the paper).

* random      - remaining columns chosen at random (paper: "4x/8x Random Cartesian mask")
* equispaced  - every R-th column (needed by GRAPPA)

The overall fraction of sampled columns is 1 / acceleration.
"""

from __future__ import annotations

import hashlib

import numpy as np

from common import config


def _rng_for(seed_key: str | int | None) -> np.random.Generator:
    if seed_key is None:
        return np.random.default_rng()
    if isinstance(seed_key, str):
        seed_key = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed_key)


def acs_slice(width: int, acs_lines: int) -> slice:
    start = (width - acs_lines + 1) // 2
    return slice(start, start + acs_lines)


def random_mask(width: int, acceleration: int, acs_lines: int = config.ACS_LINES,
                seed_key=None) -> np.ndarray:
    rng = _rng_for(seed_key)
    mask = np.zeros(width, dtype=bool)
    mask[acs_slice(width, acs_lines)] = True
    target = int(round(width / acceleration))
    remaining = max(target - int(mask.sum()), 0)
    candidates = np.flatnonzero(~mask)
    if remaining:
        mask[rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)] = True
    return mask


def equispaced_mask(width: int, acceleration: int, acs_lines: int = config.ACS_LINES,
                    seed_key=None) -> np.ndarray:
    rng = _rng_for(seed_key)
    mask = np.zeros(width, dtype=bool)
    offset = int(rng.integers(0, acceleration))
    mask[offset::acceleration] = True
    mask[acs_slice(width, acs_lines)] = True
    return mask


MASK_FUNCS = {"random": random_mask, "equispaced": equispaced_mask}


def make_mask(kind: str, width: int, acceleration: int, acs_lines: int = config.ACS_LINES,
              seed_key=None) -> np.ndarray:
    if kind not in MASK_FUNCS:
        raise ValueError(f"mask type must be one of {list(MASK_FUNCS)}")
    return MASK_FUNCS[kind](width, acceleration, acs_lines, seed_key)
