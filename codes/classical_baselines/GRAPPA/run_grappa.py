"""GRAPPA baseline.

Interpolates missing k-space lines from neighbouring acquired lines in all
coils, with weights fitted on the 24 ACS lines (Eq. 7). Uses an equispaced
mask by default (classic GRAPPA); random masks also work.

Usage (from this folder):
    python run_grappa.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.baseline_runner import baseline_parser, run_baseline  # noqa: E402

NAME = "GRAPPA"

if __name__ == "__main__":
    p = baseline_parser(NAME, HERE / "results", default_mask="equispaced")
    p.add_argument("--kernel_kx", type=int, default=3, help="Readout width of the GRAPPA kernel")
    args = p.parse_args()
    run_baseline(NAME, args, acs_lines=24, kx=args.kernel_kx)
