"""SENSE (CG-SENSE) baseline.

Inverts the coil-sensitivity encoding (Eq. 6) by solving
(A^H A + reg I) x = A^H y with conjugate gradients.

Usage (from this folder):
    python run_sense.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.baseline_runner import baseline_parser, run_baseline  # noqa: E402

NAME = "SENSE"

if __name__ == "__main__":
    p = baseline_parser(NAME, HERE / "results", default_mask="random")
    p.add_argument("--reg", type=float, default=1e-3, help="Tikhonov regularisation")
    p.add_argument("--iterations", type=int, default=30, help="Conjugate-gradient iterations")
    args = p.parse_args()
    run_baseline(NAME, args, reg=args.reg, iterations=args.iterations)
