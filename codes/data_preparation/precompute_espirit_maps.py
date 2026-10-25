"""Pre-compute ESPIRiT coil sensitivity maps for every volume (optional, Sec. IV-D / Fig. 2).

By default the loaders estimate maps from the 24-line ACS region on the fly,
which is fast and needs no extra packages. ESPIRiT maps are more accurate;
compute them once with this script and pass `--maps_dir` to any training,
testing or baseline script.

Requires: pip install sigpy

Usage (from codes/):
    python data_preparation/precompute_espirit_maps.py --splits splits.json --output_dir data/espirit_maps
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from common import config  # noqa: E402
from common.mri_ops import center_crop, espirit_maps, fft2c_np, ifft2c_np  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", help="splits.json (all splits are processed)")
    p.add_argument("--data_dir", help="...or a folder of .h5 volumes")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--acs_lines", type=int, default=config.ACS_LINES)
    args = p.parse_args()

    if args.splits:
        files = [f for v in json.loads(Path(args.splits).read_text()).values() for f in v]
    elif args.data_dir:
        files = sorted(str(f) for f in Path(args.data_dir).rglob("*.h5"))
    else:
        sys.exit("Provide --splits or --data_dir")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        target = out / Path(f).name
        if target.exists():
            continue
        with h5py.File(f, "r") as h:
            kspace = h["kspace"][()]
        if kspace.ndim == 3:
            kspace = kspace[:, None]
        maps = []
        for sl in kspace:
            k = fft2c_np(center_crop(ifft2c_np(sl), config.CROP_SIZE))
            maps.append(espirit_maps(k, args.acs_lines))
        with h5py.File(target, "w") as h:
            h.create_dataset("maps", data=np.stack(maps).astype(np.complex64), compression="gzip")
        print(f"[{i}/{len(files)}] {target.name}")
    print(f"ESPIRiT maps written to {out}")


if __name__ == "__main__":
    main()
