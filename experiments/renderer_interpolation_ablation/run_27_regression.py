"""27-image nearest-Z versus unchanged bilinear regression and unified timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageOps

from experiments.depth_pointcloud_sketch.run_batch import camera_state
from experiments.renderer_interpolation_ablation.renderers import ResearchPointRenderer
from experiments.renderer_interpolation_ablation.run_batch import (
    DEFAULT_PRESETS, bar_chart, diff_image, image_metrics, labelled_grid,
    load_presets, measure_renderer, write_csv, write_json,
)
from photography_viewpoint_agent.camera_rendering.camera_state import CameraState
from photography_viewpoint_agent.camera_rendering.depth_renderer import DepthRenderer, prepare_point_cloud
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics
from photography_viewpoint_agent.depth_estimation.monocular import MonocularDepthEstimator
from photography_viewpoint_agent.depth_estimation.storage import load_depth, normalize_depth
from photography_viewpoint_agent.schemas.video import FrameInfo


RENDERERS = ("nearest_z", "bilinear")
PRESETS = load_presets(DEFAULT_PRESETS)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
STAGES = (
    "image_load_seconds", "model_load_seconds", "depth_preprocess_seconds",
    "depth_inference_seconds", "depth_postprocess_seconds",
    "depth_mapping_seconds", "intrinsics_seconds", "prepare_point_cloud_seconds",
    "renderer_seconds", "save_seconds",
)


def select_sources(input_dir):
    files = sorted((path for path in Path(input_dir).iterdir()
                    if path.is_file() and path.suffix.lower() in EXTENSIONS),
                   key=lambda path: path.name.casefold())
    print(f"discovered image count: {len(files)}", flush=True)
    print(f"selected image count: {len(files)}", flush=True)
    print("selected filenames: " + ", ".join(path.name for path in files), flush=True)
    if len(files) != 27 or len({path.stem for path in files}) != 27:
        raise ValueError(f"Expected exactly 27 uniquely named source images, found {len(files)}")
    return files


def frame_for(path, source):
    return FrameInfo(frame_id=path.stem, frame_index=0, timestamp=0,
                     path=str(path), width=source.width, height=source.height)


def _save_mask(path, mask):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _save_formal(result, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    result.image.save(directory / "result.png")
    _save_mask(directory / "valid_mask.png", result.valid_mask)
    write_json(directory / "metadata.json", metadata)


def _save_identity(source, result, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    result.image.save(directory / "identity_result.png")
    _save_mask(directory / "valid_mask.png", result.valid_mask)
    diff_image(source, result.image).save(directory / "identity_diff.png")
    labelled_grid([(source, "SOURCE", ""), (result.image, metadata["renderer"],
                   f"MAE {metadata['rgb_mae']:.4f} | PSNR {metadata['psnr_db']:.2f} dB")],
                  2, title="Identity reconstruction").save(directory / "identity_compare.png")
    write_json(directory / "metadata.json", metadata)


def _grid_for_preset(root, stem, preset, rows_by_key, source=None):
    source = source or Image.open(root / stem / "source.png").convert("RGB")
    cells = [(source, "SOURCE", "")]
    for name in RENDERERS:
        row = rows_by_key[(stem + ".png", preset, name)]
        with Image.open(root / stem / preset / name / "result.png") as opened:
            cells.append((opened.copy(), name,
                          f"valid {row['valid_pixel_ratio']:.1%} | hole {row['hole_pixel_ratio']:.1%}\n"
                          f"median {row['render_median_seconds']:.3f}s"))
    return labelled_grid(cells, 3, title=f"{stem} / {preset}")


def _render_bar(output, values, title, label):
    bar_chart(values, output, title, label)


def _breakdown_chart(output, medians, max_seconds, title):
    # Same x-axis limit is passed to both charts.
    from PIL import ImageDraw
    sheet = Image.new("RGB", (1050, 500), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), title, fill="black", font_size=25)
    labels = (
        ("Depth preprocess", "depth_preprocess_seconds"),
        ("Depth inference", "depth_inference_seconds"),
        ("Depth postprocess", "depth_postprocess_seconds"),
        ("Pseudo-Z mapping", "depth_mapping_seconds"),
        ("Geometry", "prepare_point_cloud_seconds"),
        ("Renderer", "renderer_seconds"),
        ("Save", "save_seconds"),
    )
    for index, (label, key) in enumerate(labels):
        y = 65 + index * 58
        value = medians[key]
        draw.text((20, y + 4), label, fill="black", font_size=18)
        draw.rectangle((220, y, 220 + max(2, int(value / max_seconds * 680)), y + 26),
                       fill="#4c78a8" if key != "renderer_seconds" else "#f58518")
        draw.text((920, y + 4), f"{value:.3f}s", fill="black", font_size=18)
    draw.text((20, 475), f"Shared horizontal scale: 0 to {max_seconds:.3f}s", fill="#555555", font_size=15)
    sheet.save(output)


def depth_from_raw(raw, size):
    """Production alpha=.10 mapping, split out solely to instrument E2E."""
    lo, hi = np.percentile(raw, [2, 98])
    inverse = .1 + .9 * np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
    depth, _ = normalize_depth(1 / inverse, size)
    return depth


def trace_raw_depth(estimator, frame):
    """Time raw model stages without requiring changes to the depth module."""
    if hasattr(estimator, 'estimate_raw_trace'):
        return estimator.estimate_raw_trace(frame)
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    load_seconds = 0.0
    if estimator.model is None:
        started = time.perf_counter()
        estimator.processor = AutoImageProcessor.from_pretrained(
            estimator.model_name, use_fast=False)
        estimator.model = AutoModelForDepthEstimation.from_pretrained(
            estimator.model_name, use_safetensors=True).to(estimator.device).eval()
        if str(estimator.device).startswith('cuda'):
            torch.cuda.synchronize(estimator.device)
        load_seconds = time.perf_counter() - started
        if (estimator.model.config.model_type != 'depth_anything'
                or getattr(estimator.model.config, 'depth_estimation_type', 'relative') != 'relative'):
            raise ValueError('Expected relative inverse-depth Depth Anything weights')

    started = time.perf_counter()
    with Image.open(frame.path) as opened:
        inputs = estimator.processor(images=opened.convert('RGB'), return_tensors='pt').to(estimator.device)
    preprocess_seconds = time.perf_counter() - started
    if str(estimator.device).startswith('cuda'):
        torch.cuda.synchronize(estimator.device)
    started = time.perf_counter()
    with torch.inference_mode():
        predicted = estimator.model(**inputs).predicted_depth
    if str(estimator.device).startswith('cuda'):
        torch.cuda.synchronize(estimator.device)
    inference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    raw = torch.nn.functional.interpolate(
        predicted[:, None], size=(frame.height, frame.width),
        mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
    native = predicted[0].detach().float().cpu().numpy()
    resize_seconds = time.perf_counter() - started
    if not np.isfinite(raw).all():
        raise ValueError('Model returned nonfinite inverse depth')
    return native, raw, {
        'model_load_seconds': load_seconds,
        'depth_preprocess_seconds': preprocess_seconds,
        'depth_inference_seconds': inference_seconds,
        'depth_resize_postprocess_seconds': resize_seconds,
    }


def single_sketch_timed(source_path, output_dir, estimator, renderer, *, mode,
                        cached=None, repeat=1):
    """Time source-file to saved sketch; cached mode times state/render/save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = {key: 0.0 for key in STAGES}
    wall_started = time.perf_counter()
    if cached is None:
        started = time.perf_counter()
        with Image.open(source_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        stage["image_load_seconds"] = time.perf_counter() - started
        frame = frame_for(source_path, source)
        raw_started = time.perf_counter()
        _, raw, raw_meta = trace_raw_depth(estimator, frame)
        raw_total = time.perf_counter() - raw_started
        stage["model_load_seconds"] = raw_meta["model_load_seconds"]
        stage["depth_preprocess_seconds"] = raw_meta["depth_preprocess_seconds"]
        stage["depth_inference_seconds"] = raw_meta["depth_inference_seconds"]
        stage["depth_postprocess_seconds"] = raw_meta["depth_resize_postprocess_seconds"]
        # Torch/import and tiny orchestration overhead is accounted as a named
        # unassigned interval below instead of being hidden inside inference.
        stage["raw_trace_unassigned_seconds"] = max(0.0, raw_total - sum(
            stage[key] for key in ("model_load_seconds", "depth_preprocess_seconds",
                                   "depth_inference_seconds", "depth_postprocess_seconds")))
        started = time.perf_counter()
        depth = depth_from_raw(raw, source.size)
        stage["depth_mapping_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        k = CameraIntrinsics.from_horizontal_fov(*source.size, fov_deg=60)
        state = camera_state(k, PRESETS["crop_forward"])
        stage["intrinsics_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        geometry = prepare_point_cloud(source, depth, k)
        stage["prepare_point_cloud_seconds"] = time.perf_counter() - started
    else:
        source, depth, geometry, k = cached
        started = time.perf_counter()
        state = camera_state(k, PRESETS["crop_forward"])
        stage["intrinsics_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    result = renderer.render(source, state, depth, (128, 128, 128), geometry=geometry)
    stage["renderer_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    result.image.save(output_dir / "target_sketch.png")
    _save_mask(output_dir / "valid_mask.png", result.valid_mask)
    # Metadata is part of the timed single-sketch path.  Timings are written
    # afterward to avoid recursive timing of the metadata save itself.
    write_json(output_dir / "metadata.json", {"source": source_path.name,
               "renderer": renderer.name, "preset": "crop_forward", "mode": mode,
               "valid_pixel_ratio": float(result.valid_mask.mean())})
    stage["save_seconds"] = time.perf_counter() - started
    total = time.perf_counter() - wall_started
    accounted = sum(stage.values())
    row = {"renderer": renderer.name, "source": source_path.name, "mode": mode,
           "repeat": repeat, **stage, "e2e_seconds": total,
           "stage_sum_seconds": accounted, "unaccounted_seconds": total - accounted,
           "stage_sum_gap_ratio": abs(total - accounted) / max(total, 1e-9)}
    write_json(output_dir / "timing.json", row)
    return row, (source, depth, geometry, k)


def run_e2e(root, input_dir, model, device="cpu"):
    representatives = [Path(input_dir) / f"video{number}_final_01_gpt.png"
                       for number in (4, 5, 23)]
    if not all(path.is_file() for path in representatives):
        raise FileNotFoundError("Representative E2E sources missing")
    rows = []
    for renderer_name in RENDERERS:
        renderer = ResearchPointRenderer(renderer_name)
        estimator = MonocularDepthEstimator(str(Path(model).resolve()), device)
        base = root / "e2e" / renderer_name
        cold, cached = single_sketch_timed(
            representatives[0], base / "cold" / representatives[0].stem,
            estimator, renderer, mode="cold", repeat=1)
        rows.append(cold)
        for path in representatives:
            for repeat in range(1, 6):
                warm, cached = single_sketch_timed(
                    path, base / "warm" / path.stem / f"repeat_{repeat}",
                    estimator, renderer, mode="warm", repeat=repeat)
                rows.append(warm)
            for repeat in range(1, 6):
                rerender, _ = single_sketch_timed(
                    path, base / "same_source" / path.stem / f"repeat_{repeat}",
                    estimator, renderer, mode="same_source", cached=cached, repeat=repeat)
                rows.append(rerender)
    write_csv(root / "e2e_timing.csv", rows)
    write_json(root / "e2e_timing.json", rows)
    warm_medians = {name: statistics.median(row["e2e_seconds"] for row in rows
                    if row["renderer"] == name and row["mode"] == "warm") for name in RENDERERS}
    _render_bar(root / "e2e_latency_compare.png", warm_medians,
                "Warm single-image to target sketch", "Median end-to-end seconds")
    stage_medians = {name: {stage: statistics.median(row[stage] for row in rows
                      if row["renderer"] == name and row["mode"] == "warm")
                      for stage in STAGES} for name in RENDERERS}
    maximum = max(value for record in stage_medians.values() for value in record.values()) * 1.1
    for name in RENDERERS:
        _breakdown_chart(root / f"pipeline_breakdown_{'nearest' if name == 'nearest_z' else name}.png",
                         stage_medians[name], maximum, f"Warm pipeline: {name}")
    return rows, warm_medians, stage_medians


def _overview(root, sources, paired):
    # At most nine source rows per JPEG to keep viewers responsive.
    overview = root / "overview"
    overview.mkdir(exist_ok=True)
    names = list(PRESETS) + ["identity"]
    for preset in names:
        for part, offset in enumerate(range(0, len(sources), 9), 1):
            cells = []
            for path in sources[offset:offset + 9]:
                for name in RENDERERS:
                    if preset == "identity":
                        result_path = root / path.stem / "identity" / name / "identity_result.png"
                        label = paired["identity"][(path.name, name)]
                        line = f"MAE {label['rgb_mae']:.3f}"
                    else:
                        result_path = root / path.stem / preset / name / "result.png"
                        row = paired["formal"][(path.name, preset, name)]
                        line = f"valid {row['valid_pixel_ratio']:.1%} | {row['render_median_seconds']:.3f}s"
                    with Image.open(result_path) as opened:
                        cells.append((opened.copy(), f"{path.stem} / {name}", line))
            labelled_grid(cells, 2, cell_size=(300, 410), title=f"{preset} part {part}").save(
                overview / f"{preset}_compare_part_{part:03d}.jpg", quality=91)
    worst = sorted(paired["comparison"], key=lambda row: row["delta_valid"])[:10]
    _largest_regression_grid(root, worst).save(
        overview / "largest_coverage_regressions.jpg", quality=91)


def _largest_regression_grid(root, worst):
    from PIL import ImageDraw
    width, row_height, header = 720, 370, 55
    sheet = Image.new("RGB", (width, header + row_height * len(worst)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Largest bilinear coverage regressions", fill="black", font_size=23)
    for index, row in enumerate(worst, 1):
        y = header + (index - 1) * row_height
        draw.text((10, y), f"#{index}  {row['source']} / {row['preset']}  delta {row['delta_valid']:.2%}",
                  fill="black", font_size=17)
        for column, name in enumerate(RENDERERS):
            with Image.open(root / Path(row["source"]).stem / row["preset"] / name / "result.png") as opened:
                thumb = ImageOps.contain(opened.convert("RGB"), (344, 300))
                x = column * 360 + (360 - thumb.width) // 2
                sheet.paste(thumb, (x, y + 26))
            draw.text((column * 360 + 12, y + 330),
                      f"{name}: {row['nearest_valid' if name == 'nearest_z' else 'bilinear_valid']:.1%} valid",
                      fill="#333333", font_size=17)
    return sheet


def _summary_stats(formal, identity, comparison, e2e, warm_medians):
    result = {"identity": {}, "formal": {}, "renderer_only": {}, "e2e": {}, "coverage": {}}
    for name in RENDERERS:
        ids = [row for row in identity if row["renderer"] == name]
        cases = [row for row in formal if row["renderer"] == name]
        maes = [row["rgb_mae"] for row in ids]
        all_samples = [sample for row in cases for sample in row["render_samples_seconds"]]
        result["identity"][name] = {
            "mae_mean": statistics.mean(maes), "mae_median": statistics.median(maes),
            "mae_p90": float(np.percentile(maes, 90)), "mae_max": max(maes),
            "psnr_mean": statistics.mean(row["psnr_db"] for row in ids),
            "valid_mean": statistics.mean(row["valid_pixel_ratio"] for row in ids),
        }
        result["formal"][name] = {
            "valid_mean": statistics.mean(row["valid_pixel_ratio"] for row in cases),
            "valid_median": statistics.median(row["valid_pixel_ratio"] for row in cases),
            "hole_mean": statistics.mean(row["hole_pixel_ratio"] for row in cases),
        }
        result["renderer_only"][name] = {
            "mean_seconds": statistics.mean(all_samples),
            "median_seconds": statistics.median(all_samples),
            "p90_seconds": float(np.percentile(all_samples, 90)),
        }
        result["e2e"][name] = {
            "cold_seconds": next(row["e2e_seconds"] for row in e2e
                                 if row["renderer"] == name and row["mode"] == "cold"),
            "warm_median_seconds": warm_medians[name],
            "same_source_median_seconds": statistics.median(
                row["e2e_seconds"] for row in e2e
                if row["renderer"] == name and row["mode"] == "same_source"),
        }
    deltas = [row["delta_valid"] for row in comparison]
    result["coverage"] = {
        "mean": statistics.mean(deltas), "median": statistics.median(deltas),
        "p10": float(np.percentile(deltas, 10)), "p90": float(np.percentile(deltas, 90)),
        "min": min(deltas), "max": max(deltas),
        "below_minus_3pp": sum(value < -.03 for value in deltas),
        "below_minus_5pp": sum(value < -.05 for value in deltas),
        "below_minus_10pp": sum(value < -.10 for value in deltas),
    }
    result["bilinear_identity_worst_5"] = sorted(
        (row for row in identity if row["renderer"] == "bilinear"),
        key=lambda row: row["rgb_mae"], reverse=True)[:5]
    result["coverage_worst_10"] = sorted(comparison, key=lambda row: row["delta_valid"])[:10]
    result["bilinear_renderer_speedup_ratio"] = (
        result["renderer_only"]["nearest_z"]["median_seconds"] /
        result["renderer_only"]["bilinear"]["median_seconds"])
    return result


def run(input_dir, output_dir, model, device="cpu"):
    input_dir, root = Path(input_dir).resolve(), Path(output_dir).resolve()
    sources = select_sources(input_dir)
    if root.exists():
        raise FileExistsError(f"Output directory exists: {root}")
    root.mkdir(parents=True)
    model = Path(model).resolve()
    estimator = MonocularDepthEstimator(str(model), device)
    renderers = {name: ResearchPointRenderer(name) for name in RENDERERS}
    production = DepthRenderer(1, renderer='nearest_z')
    summary = {"experiment": "27-image_renderer_bilinear_regression",
               "status": "running", "input_dir": str(input_dir), "output_dir": str(root),
               "discovered_image_count": len(sources), "selected_image_count": len(sources),
               "selected_filenames": [path.name for path in sources],
               "fixed": {"depth_model": str(model), "depth_mapping_alpha": .10,
                         "source_horizontal_fov_deg": 60, "depth_rel_tol": .01,
                         "renderers": list(RENDERERS), "presets": PRESETS,
                         "renderer_device": "cpu", "warmups": 1, "timed_repeats": 5},
               "depth_inference_calls": 0, "geometry_builds": 0,
               "formal_cases_expected": 216, "identity_cases_expected": 54,
               "baseline_regression_exact_count": 0}
    write_json(root / "experiment_summary.json", summary)
    formal, identity, comparison = [], [], []
    started = time.perf_counter()
    for index, path in enumerate(sources, 1):
        source_dir = root / path.stem
        source_dir.mkdir()
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        source.save(source_dir / "source.png")
        frame = frame_for(path, source)
        print(f"[{index}/27] {path.name}: depth once / geometry once", flush=True)
        observation = estimator.estimate(frame, source_dir / "depth" / "depth.npy")
        summary["depth_inference_calls"] += 1
        depth = load_depth(observation, source.size)
        depth.setflags(write=False)
        k = CameraIntrinsics.from_horizontal_fov(*source.size, fov_deg=60)
        geometry = prepare_point_cloud(source, depth, k)
        summary["geometry_builds"] += 1
        identity_state = CameraState(source_intrinsics=k, target_intrinsics=k)
        for name in RENDERERS:
            renderer = renderers[name]
            result = renderer.render(source, identity_state, depth, geometry=geometry)
            mae, psnr = image_metrics(source, result.image)
            row = {"source": path.name, "renderer": name, "rgb_mae": mae,
                   "psnr_db": psnr, "valid_pixel_ratio": float(result.valid_mask.mean())}
            _save_identity(source, result, source_dir / "identity" / name, row)
            identity.append(row)
        for preset, parameters in PRESETS.items():
            state = camera_state(k, parameters)
            # The independent production oracle is used on every formal case.
            oracle = production.render(source, state, depth, geometry=geometry)
            local = {}
            for name in RENDERERS:
                renderer = renderers[name]
                result, timing = measure_renderer(
                    lambda r=renderer, s=state: r.render(source, s, depth, geometry=geometry),
                    repetitions=5, warmups=1)
                ratio = float(result.valid_mask.mean())
                row = {"source": path.name, "preset": preset, "renderer": name,
                       "valid_pixel_ratio": ratio, "hole_pixel_ratio": 1 - ratio,
                       "render_seconds": timing["render_median_seconds"],
                       "project_seconds": timing["project_median_seconds"],
                       "splat_seconds": timing["rasterization_median_seconds"],
                       **{f"repeat_{j}": value for j, value in enumerate(timing["render_samples_seconds"], 1)},
                       **timing}
                if name == "nearest_z":
                    exact_rgb = np.array_equal(np.asarray(result.image), np.asarray(oracle.image))
                    exact_mask = np.array_equal(result.valid_mask, oracle.valid_mask)
                    row.update(production_rgb_exact=exact_rgb, production_mask_exact=exact_mask)
                    if not exact_rgb or not exact_mask:
                        raise AssertionError(f"Production regression failed: {path.name}/{preset}")
                    summary["baseline_regression_exact_count"] += 1
                _save_formal(result, source_dir / preset / name, row)
                local[name] = row
                formal.append(row)
            near, bilinear = local["nearest_z"], local["bilinear"]
            comparison.append({"source": path.name, "preset": preset,
                               "nearest_valid": near["valid_pixel_ratio"],
                               "bilinear_valid": bilinear["valid_pixel_ratio"],
                               "delta_valid": bilinear["valid_pixel_ratio"] - near["valid_pixel_ratio"],
                               "nearest_hole": near["hole_pixel_ratio"],
                               "bilinear_hole": bilinear["hole_pixel_ratio"],
                               "delta_hole": bilinear["hole_pixel_ratio"] - near["hole_pixel_ratio"],
                               "nearest_render_median": near["render_median_seconds"],
                               "bilinear_render_median": bilinear["render_median_seconds"],
                               "render_speedup_ratio": near["render_median_seconds"] /
                               bilinear["render_median_seconds"]})
            row_lookup = {(row["source"], row["preset"], row["renderer"]): row
                          for row in formal if row["source"] == path.name}
            grid = _grid_for_preset(root, path.stem, preset, row_lookup, source)
            (source_dir / "comparisons").mkdir(exist_ok=True)
            grid.save(source_dir / preset / "renderer_compare.jpg", quality=93)
            grid.save(source_dir / "comparisons" / f"{preset}_renderer_compare.jpg", quality=93)
        cells = []
        rows_for_source = {(row["source"], row["preset"], row["renderer"]): row
                           for row in formal if row["source"] == path.name}
        for preset in PRESETS:
            cells.append((source, f"{preset} / SOURCE", ""))
            for name in RENDERERS:
                row = rows_for_source[(path.name, preset, name)]
                with Image.open(source_dir / preset / name / "result.png") as opened:
                    cells.append((opened.copy(), name,
                                  f"valid {row['valid_pixel_ratio']:.1%} | hole {row['hole_pixel_ratio']:.1%}\n"
                                  f"median {row['render_median_seconds']:.3f}s"))
        labelled_grid(cells, 3, title=path.stem).save(
            source_dir / "all_renderer_results_grid.jpg", quality=93)
        summary.update(formal_case_count=len(formal), identity_case_count=len(identity),
                       completed_source_count=index, elapsed_seconds=time.perf_counter() - started)
        write_json(root / "experiment_summary.json", summary)
        write_json(root / "timing_summary.json", formal)
        write_csv(root / "renderer_regression_metrics.csv", comparison)
        write_csv(root / "identity_metrics.csv", _identity_pairs(identity))
    print("Formal regression complete; E2E calibration starting", flush=True)
    e2e, warm_medians, stage_medians = run_e2e(root, input_dir, model, device)
    paired = {"formal": {(row["source"], row["preset"], row["renderer"]): row for row in formal},
              "identity": {(row["source"], row["renderer"]): row for row in identity},
              "comparison": comparison}
    _overview(root, sources, paired)
    from experiments.renderer_interpolation_ablation.analyze_coverage import analyze as analyze_coverage
    coverage_holes = analyze_coverage(root)
    global_stats = _summary_stats(formal, identity, comparison, e2e, warm_medians)
    _render_bar(root / "renderer_latency_compare.png",
                {name: global_stats["renderer_only"][name]["median_seconds"] for name in RENDERERS},
                "Renderer-only latency: 27 images x 4 presets", "Median of 540 timed calls per renderer")
    summary.update(status="completed", formal_case_count=len(formal), identity_case_count=len(identity),
                   paired_case_count=len(comparison), baseline_regression_all_exact=(
                       summary["baseline_regression_exact_count"] == 108),
                   depth_inference_calls_batch=summary["depth_inference_calls"],
                   geometry_builds_batch=summary["geometry_builds"],
                   e2e_depth_inference_calls=sum(row["mode"] != "same_source" for row in e2e),
                   coverage_hole_characterization=coverage_holes,
                   statistics=global_stats, warm_stage_medians=stage_medians,
                   elapsed_seconds=time.perf_counter() - started)
    write_json(root / "experiment_summary.json", summary)
    write_csv(root / "timing_summary.csv", formal)
    return summary


def _identity_pairs(identity):
    by_source = {}
    for row in identity:
        by_source.setdefault(row["source"], {})[row["renderer"]] = row
    return [{"source": source,
             "nearest_mae": rows["nearest_z"]["rgb_mae"],
             "bilinear_mae": rows["bilinear"]["rgb_mae"],
             "nearest_psnr": rows["nearest_z"]["psnr_db"],
             "bilinear_psnr": rows["bilinear"]["psnr_db"],
             "nearest_valid": rows["nearest_z"]["valid_pixel_ratio"],
             "bilinear_valid": rows["bilinear"]["valid_pixel_ratio"]}
            for source, rows in by_source.items() if len(rows) == 2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path,
                        default=Path("workdir/depth_models/depth-anything-v2-small"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run(args.input_dir, args.output_dir, args.depth_model, args.device)


if __name__ == "__main__":
    main()
