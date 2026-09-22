"""Classify bilinear-only missing pixels versus holes shared with nearest-Z."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from PIL import Image

from experiments.renderer_interpolation_ablation.run_batch import write_csv, write_json


def analyze(root):
    root = Path(root)
    with (root / "renderer_regression_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    records = []
    for case in cases:
        directory = root / Path(case["source"]).stem / case["preset"]
        with Image.open(directory / "nearest_z" / "valid_mask.png") as image:
            nearest = np.asarray(image) > 0
        with Image.open(directory / "bilinear" / "valid_mask.png") as image:
            bilinear = np.asarray(image) > 0
        bilinear_only_missing = nearest & ~bilinear
        shared_missing = ~nearest & ~bilinear
        bilinear_only_covered = ~nearest & bilinear
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            bilinear_only_missing.astype(np.uint8), connectivity=8)
        sizes = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=int)
        records.append({
            "source": case["source"], "preset": case["preset"],
            "bilinear_only_missing_ratio": float(bilinear_only_missing.mean()),
            "shared_hole_ratio": float(shared_missing.mean()),
            "bilinear_only_covered_ratio": float(bilinear_only_covered.mean()),
            "bilinear_only_missing_component_count": count - 1,
            "largest_new_hole_component_pixels": int(sizes.max()) if len(sizes) else 0,
            "p99_new_hole_component_pixels": float(np.percentile(sizes, 99)) if len(sizes) else 0,
        })
    write_csv(root / "coverage_hole_characterization.csv", records)
    summary = {"case_count": len(records),
               "largest_new_hole_component_pixels_max": max(row["largest_new_hole_component_pixels"] for row in records),
               "bilinear_only_covered_ratio_max": max(row["bilinear_only_covered_ratio"] for row in records),
               "worst_coverage_cases": sorted(records, key=lambda row: row["bilinear_only_missing_ratio"], reverse=True)[:10]}
    write_json(root / "coverage_hole_characterization.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(analyze(args.output_dir))


if __name__ == "__main__":
    main()
