"""Compressed Sensing with Total Variation (CS-TV) baseline.

Solves  x = argmin 1/2 ||Ax - y||^2 + lambda TV(x)  (Eq. 4) with FISTA,
alternating a gradient step on the data term and a TV proximal step (Eq. 5).

Usage (from this folder):
    python run_cs_tv.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.baseline_runner import baseline_parser, run_baseline  # noqa: E402

NAME = "CS_TV"

if __name__ == "__main__":
    p = baseline_parser(NAME, HERE / "results", default_mask="random")
    p.add_argument("--lam", type=float, default=0.003, help="TV weight lambda (Eq. 4)")
    p.add_argument("--iterations", type=int, default=50, help="FISTA iterations (Eq. 5)")
    args = p.parse_args()
    run_baseline(NAME, args, lam=args.lam, iterations=args.iterations)
