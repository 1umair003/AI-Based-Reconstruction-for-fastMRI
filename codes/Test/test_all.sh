#!/usr/bin/env bash
# Test all five networks.
# Usage: bash test_all.sh ../splits.json [acceleration]
set -euo pipefail
SPLITS="${1:?Usage: bash test_all.sh /path/to/splits.json [acceleration]}"
ACC="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/Cascade_Net/test_cascade_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/KIKI_Net/test_kiki_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/UNet/test_unet.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/PD_Net/test_pd_net.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/UNet_DC/test_unet_dc.py" --splits "$SPLITS" --acceleration "$ACC"
