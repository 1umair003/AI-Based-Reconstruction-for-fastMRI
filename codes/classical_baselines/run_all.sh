#!/usr/bin/env bash
# Run all classical baselines.
# Usage: bash run_all.sh ../splits.json [acceleration]
set -euo pipefail
SPLITS="${1:?Usage: bash run_all.sh /path/to/splits.json [acceleration]}"
ACC="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/Zero_Filled/run_zero_filled.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/CS_TV/run_cs_tv.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/SENSE/run_sense.py" --splits "$SPLITS" --acceleration "$ACC"
python "$HERE/GRAPPA/run_grappa.py" --splits "$SPLITS" --acceleration "$ACC"
