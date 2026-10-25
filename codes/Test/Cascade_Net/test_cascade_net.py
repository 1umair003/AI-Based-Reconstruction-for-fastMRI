"""Test Cascade-Net: SSIM, PSNR, NMSE, SNR, parameters and inference time.

Loads codes/training/Cascade_Net/outputs/best_model.pt by default.

Usage (from this folder):
    python test_cascade_net.py --splits ../../splits.json --acceleration 4
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parents[1]
sys.path.insert(0, str(CODES))

from common.tester import run_test, test_parser  # noqa: E402

MODEL_NAME = "Cascade_Net"
CHECKPOINT = CODES / "training" / MODEL_NAME / "outputs" / "best_model.pt"

if __name__ == "__main__":
    args = test_parser(MODEL_NAME, CHECKPOINT, HERE / "results").parse_args()
    run_test(MODEL_NAME, args)
