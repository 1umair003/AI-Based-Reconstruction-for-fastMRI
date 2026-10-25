"""Collect every test summary into one Table II-style table.

Reads Test/<Model>/results/<acc>x/summary.json and
classical_baselines/<Method>/results/<acc>x/summary.json.

Usage (from codes/):
    python Test/summarize_results.py --acceleration 4
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
sys.path.insert(0, str(CODES))

import pandas as pd  # noqa: E402

from common import config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acceleration", type=int, default=config.DEFAULT_ACCELERATION)
    args = p.parse_args()

    sources = [(CODES / "classical_baselines" / c / "results", "Classical") for c in config.CLASSICAL]
    sources += [(HERE / m / "results", "DL+DC" if m == "UNet_DC" else "DL") for m in config.MODELS]
    rows = []
    for folder, kind in sources:
        f = folder / f"{args.acceleration}x" / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        rows.append({
            "Method": s["method"], "SSIM": round(s["SSIM"]["mean"], 3), "PSNR (dB)": round(s["PSNR"]["mean"], 1),
            "SNR (dB)": round(s["SNR"]["mean"], 2), "NMSE": round(s["NMSE"]["mean"], 4),
            "Params (M)": s.get("params_M"),
            "Time (ms)": round(s["inference_ms_per_slice"], 1) if s.get("inference_ms_per_slice") else None,
            "Type": kind,
        })
    if not rows:
        sys.exit("No results found. Run the test / baseline scripts first.")
    df = pd.DataFrame(rows).sort_values("SSIM")
    out = HERE / f"results_table_{args.acceleration}x"
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))
    print(f"\nSaved {out}.csv and {out}.md")


if __name__ == "__main__":
    main()
