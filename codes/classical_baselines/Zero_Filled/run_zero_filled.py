"""Zero-Filled IFFT baseline.

Inverse FFT of the under-sampled k-space with missing lines set to zero,
combined over coils with the sensitivity maps (A^H y).

Usage (from this folder):
    python run_zero_filled.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.baseline_runner import baseline_parser, run_baseline  # noqa: E402

NAME = "Zero_Filled"

if __name__ == "__main__":
    p = baseline_parser(NAME, HERE / "results", default_mask="random")
    args = p.parse_args()
    run_baseline(NAME, args)
