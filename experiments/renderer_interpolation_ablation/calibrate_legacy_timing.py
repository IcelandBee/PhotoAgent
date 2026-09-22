"""Match old production and research nearest renderer on identical warmed inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from experiments.depth_pointcloud_sketch.run_batch import camera_state
from experiments.renderer_interpolation_ablation.run_batch import load_presets, write_json
from experiments.renderer_interpolation_ablation.renderers import ResearchPointRenderer
from photography_viewpoint_agent.camera_rendering.depth_renderer import DepthRenderer, prepare_point_cloud
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics


def measure(call, repeats=5):
    call()  # warm-up
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return result, {"samples_seconds": samples, "median_seconds": statistics.median(samples),
                    "mean_seconds": statistics.mean(samples)}


def run(source_path, depth_path, output_path, preset="crop_forward"):
    source = Image.open(source_path).convert("RGB")
    depth = np.load(depth_path, allow_pickle=False)
    k = CameraIntrinsics.from_horizontal_fov(*source.size, fov_deg=60)
    geometry = prepare_point_cloud(source, depth, k)
    state = camera_state(k, load_presets()[preset])
    production = DepthRenderer(1, renderer='nearest_z')
    research = ResearchPointRenderer("nearest_z")
    old, old_stats = measure(lambda: production.render(source, state, depth, geometry=geometry))
    new, new_stats = measure(lambda: research.render(source, state, depth, geometry=geometry))
    exact_rgb = np.array_equal(np.asarray(old.image), np.asarray(new.image))
    exact_mask = np.array_equal(old.valid_mask, new.valid_mask)
    record = {"source": str(Path(source_path).resolve()), "depth": str(Path(depth_path).resolve()),
              "preset": preset, "same_source_size": list(source.size), "same_geometry": True,
              "same_state": True, "production": old_stats, "research_nearest": new_stats,
              "production_over_research_ratio": old_stats["median_seconds"] / new_stats["median_seconds"],
              "rgb_exact": exact_rgb, "mask_exact": exact_mask,
              "implementation_difference": (
                  "Production splats() generator materializes 3x3 candidate arrays in each of three passes; "
                  "research nearest_z materializes the same nine arrays once and reuses them for all passes."),
              "timing_boundary": "Both are warmed render calls with the same PreparedPointCloud and CameraState; no I/O."}
    write_json(Path(output_path), record)
    if not exact_rgb or not exact_mask:
        raise AssertionError("Production and research nearest outputs differ")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=list(load_presets()), default="crop_forward")
    args = parser.parse_args(argv)
    record = run(args.source, args.depth, args.output, args.preset)
    print(record)


if __name__ == "__main__":
    main()
