"""Train PD-Net.

Learned primal-dual network: 10 unrolled iterations of dual (k-space) and primal (image) CNN updates.

Usage (from this folder):
    python train_pd_net.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.trainer import run_training, training_parser  # noqa: E402

MODEL_NAME = "PD_Net"

if __name__ == "__main__":
    args = training_parser(MODEL_NAME, HERE / "outputs").parse_args()
    run_training(MODEL_NAME, args)
