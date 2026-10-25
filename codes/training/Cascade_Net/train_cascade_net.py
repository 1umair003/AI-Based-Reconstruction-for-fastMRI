"""Train Cascade-Net.

Deep cascade of 5 CNNs (5 conv layers each) with data consistency after every cascade [10].

Usage (from this folder):
    python train_cascade_net.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # -> codes/

from common.trainer import run_training, training_parser  # noqa: E402

MODEL_NAME = "Cascade_Net"

if __name__ == "__main__":
    args = training_parser(MODEL_NAME, HERE / "outputs").parse_args()
    run_training(MODEL_NAME, args)
