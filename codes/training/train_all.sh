#!/usr/bin/env bash
# Train all five networks.
# Usage: bash train_all.sh ../splits.json [acceleration]
set -euo pipefail
SPLITS="${1:?Usage: bash train_all.sh /path/to/splits.json [acceleration]}"
ACC="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/Cascade_Net/train_cascade_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/KIKI_Net/train_kiki_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/UNet/train_unet.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/PD_Net/train_pd_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/UNet_DC/train_unet_dc.py" --splits "$SPLITS" --acceleration "$ACC"
