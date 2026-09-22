"""Run a controlled nearest/bilinear/Gaussian point-rasterization ablation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.depth_pointcloud_sketch.run_batch import camera_state
from experiments.renderer_interpolation_ablation.renderers import RendererConfig, create_renderers
from photography_viewpoint_agent.camera_rendering.camera_state import CameraState
from photography_viewpoint_agent.camera_rendering.depth_renderer import DepthRenderer, prepare_point_cloud
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics
from photography_viewpoint_agent.depth_estimation.monocular import MonocularDepthEstimator
from photography_viewpoint_agent.depth_estimation.storage import load_depth, save_visualization
from photography_viewpoint_agent.schemas.video import FrameInfo


SOURCE_NAMES = (
    "video4_final_01_gpt.png",
    "video5_final_01_gpt.png",
    "video23_final_01_gpt.png",
)
RENDERER_NAMES = ("nearest_z", "bilinear", "gaussian")
DEFAULT_PRESETS = Path(__file__).with_name("presets.json")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_presets(path=DEFAULT_PRESETS):
    presets = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if tuple(presets) != (
        "crop_forward", "expand_backward", "view_up_pitch_down", "view_right_yaw_left"
    ):
        raise ValueError("This controlled ablation requires exactly the four specified presets")
    return presets


def timing_stats(samples):
    values = [float(value) for value in samples]
    return {
        "timing_repetitions": len(values),
        "render_samples_seconds": values,
        "render_mean_seconds": statistics.fmean(values),
        "render_median_seconds": statistics.median(values),
        "render_min_seconds": min(values),
        "render_max_seconds": max(values),
        "render_p90_seconds": float(np.percentile(values, 90)),
    }


def measure_renderer(render_call, repetitions=5, warmups=1):
    """Measure only render_call. Saving and metadata work happen at the caller."""
    for _ in range(warmups):
        render_call()
    samples, result, internals = [], None, []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = render_call()
        samples.append(time.perf_counter() - started)
        internals.append({
            "project_seconds": result.metadata.get("project_seconds"),
            "rasterization_seconds": result.metadata.get("rasterization_seconds"),
        })
    stats = timing_stats(samples)
    for field in ("project_seconds", "rasterization_seconds"):
        values = [row[field] for row in internals if row[field] is not None]
        stats[field.replace("seconds", "median_seconds")] = statistics.median(values) if values else None
    return result, stats


def image_metrics(source, result):
    a = np.asarray(source, dtype=np.float32)
    b = np.asarray(result, dtype=np.float32)
    mae = float(np.abs(a - b).mean())
    mse = float(np.square(a - b).mean())
    # A finite cap keeps JSON/CSV portable while preserving the exact-match case.
    psnr = 100.0 if mse == 0 else float(20 * math.log10(255) - 10 * math.log10(mse))
    return mae, psnr


def labelled_grid(cells, columns, cell_size=(300, 410), title=None):
    """Cells are (PIL image, line1, line2); labels sit below images."""
    cell_w, cell_h = cell_size
    image_h = cell_h - 76
    rows = math.ceil(len(cells) / columns)
    top = 40 if title else 0
    sheet = Image.new("RGB", (columns * cell_w, top + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((10, 9), title, fill="black", font_size=22)
    for index, (image, line1, line2) in enumerate(cells):
        x, y = index % columns * cell_w, top + index // columns * cell_h
        thumb = ImageOps.contain(image.convert("RGB"), (cell_w - 12, image_h - 8))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        draw.text((x + 8, y + image_h + 5), line1, fill="black", font_size=18)
        if line2:
            draw.text((x + 8, y + image_h + 31), line2, fill="#333333", font_size=14)
    return sheet


def save_result(result, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result.image.save(case_dir / "result.png")
    save_seconds = time.perf_counter() - started
    Image.fromarray(result.valid_mask.astype(np.uint8) * 255).save(case_dir / "valid_mask.png")
    np.save(case_dir / "projected_depth.npy", result.debug_info["projected_depth"], allow_pickle=False)
    return save_seconds


def diff_image(source, result, gain=4):
    difference = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(result, dtype=np.int16))
    return Image.fromarray(np.clip(difference * gain, 0, 255).astype(np.uint8))


def bar_chart(values, output, title, ylabel):
    width, height = 900, 600
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((35, 20), title, fill="black", font_size=26)
    draw.text((35, 55), ylabel, fill="#444444", font_size=18)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    maximum = max(values.values()) * 1.12 or 1
    baseline, chart_h = 520, 390
    for index, (name, value) in enumerate(values.items()):
        x0 = 120 + index * 250
        bar_h = int(chart_h * value / maximum)
        draw.rectangle((x0, baseline - bar_h, x0 + 130, baseline), fill=colors[index])
        draw.text((x0, baseline + 15), name, fill="black", font_size=18)
        draw.text((x0, baseline - bar_h - 30), f"{value:.3f}s", fill="black", font_size=18)
    draw.line((70, baseline, 840, baseline), fill="black", width=2)
    sheet.save(output)


def _source_paths(input_dir):
    paths = [Path(input_dir) / name for name in SOURCE_NAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing controlled input(s): " + ", ".join(missing))
    return paths


def _depth_timing(observation):
    timing = observation.metadata.get("timing", {})
    return {
        "depth_preprocess_seconds": float(timing.get("depth_preprocess_seconds", 0)),
        "depth_inference_seconds": float(timing.get("depth_inference_seconds", 0)),
        "depth_postprocess_seconds": float(timing.get("depth_resize_postprocess_seconds", 0)),
        "depth_model_load_seconds_excluded": float(timing.get("model_load_seconds", 0)),
    }


def run_batch(input_dir, output_dir, *, model, device="cpu", repetitions=5,
              depth_rel_tol=0.01, gaussian_sigma=0.75, fill_color=(128, 128, 128)):
    input_dir, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if repetitions != 5:
        raise ValueError("Controlled run requires exactly five measured repetitions")
    output.mkdir(parents=True)
    (output / "overview").mkdir()
    presets = load_presets()
    config = RendererConfig(depth_rel_tol=depth_rel_tol, gaussian_sigma=gaussian_sigma)
    renderers = create_renderers(config)
    estimator = MonocularDepthEstimator(str(Path(model).resolve()), device)
    metric_rows, timing_rows, identities, regressions, image_records = [], [], [], [], []
    started = time.perf_counter()
    summary = {
        "experiment": "renderer_interpolation_ablation",
        "status": "running",
        "input_dir": str(input_dir), "output_dir": str(output),
        "sources": list(SOURCE_NAMES), "presets": presets,
        "renderers": {
            "nearest_z": {"radius": 1, "policy": "production three-pass nearest-Z and nearest-point winner"},
            "bilinear": {"footprint": "floor/ceil four pixels", "depth_rel_tol": depth_rel_tol},
            "gaussian": {"radius": 1, "sigma": gaussian_sigma, "depth_rel_tol": depth_rel_tol},
        },
        "fixed": {"depth_model": "Depth Anything V2 Small", "mapping_alpha": 0.10,
                  "source_horizontal_fov_deg": 60, "device": device,
                  "warmups": 1, "measured_repetitions": repetitions,
                  "fill_color": list(fill_color)},
        "depth_inference_calls": 0, "geometry_builds": 0,
        "formal_cases_expected": 36, "identity_cases_expected": 9,
    }
    write_json(output / "experiment_summary.json", summary)

    source_data = {}
    for source_index, source_path in enumerate(_source_paths(input_dir), 1):
        stem, source_dir = source_path.stem, output / source_path.stem
        source_dir.mkdir()
        (source_dir / "comparisons").mkdir()
        with Image.open(source_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        source.save(source_dir / "source.png")
        frame = FrameInfo(frame_id=stem, frame_index=0, timestamp=0,
                          path=str(source_path), width=source.width, height=source.height)
        print(f"[{source_index}/3] {source_path.name}: depth once", flush=True)
        observation = estimator.estimate(frame, source_dir / "depth" / "depth.npy")
        summary["depth_inference_calls"] += 1
        depth = load_depth(observation, source.size)
        save_visualization(depth, source_dir / "depth" / "depth_vis.png")
        scene_reference = float(np.median(depth[depth > 0]))
        depth = depth / scene_reference if observation.metric else depth
        depth.setflags(write=False)
        k = CameraIntrinsics.from_horizontal_fov(*source.size, fov_deg=60)
        geometry_started = time.perf_counter()
        geometry = prepare_point_cloud(source, depth, k)
        geometry_seconds = time.perf_counter() - geometry_started
        summary["geometry_builds"] += 1
        shared = {**_depth_timing(observation), "prepare_point_cloud_seconds": geometry_seconds}
        source_record = {"source": source_path.name, "image_size": list(source.size),
                         "point_count": len(geometry.ids), "shared_timing": shared,
                         "depth_model": observation.metadata.get("model")}
        image_records.append(source_record)
        source_data[stem] = {"source": source, "results": {}, "identity": {}, "dir": source_dir}

        # Identity is measured with the same method because its latency is shown in its grid.
        identity_state = CameraState(source_intrinsics=k, target_intrinsics=k)
        for renderer_name in RENDERER_NAMES:
            renderer = renderers[renderer_name]
            print(f"  identity/{renderer_name}: warmup + {repetitions}", flush=True)
            result, timing = measure_renderer(
                lambda r=renderer: r.render(source, identity_state, depth, fill_color, geometry=geometry),
                repetitions)
            identity_dir = source_dir / "identity" / renderer_name
            save_seconds = save_result(result, identity_dir)
            result.image.save(identity_dir / "identity_result.png")
            mae, psnr = image_metrics(source, result.image)
            diff = diff_image(source, result.image)
            diff.save(identity_dir / "identity_diff.png")
            compare = labelled_grid([
                (source, "SOURCE", ""),
                (result.image, renderer_name, f"MAE {mae:.3f} | PSNR {psnr:.2f} dB"),
            ], 2, title=f"{stem} identity")
            compare.save(identity_dir / "identity_compare.png")
            row = {
                "source": source_path.name, "kind": "identity", "preset": "identity",
                "renderer": renderer_name, "status": "success", "global_rgb_mae": mae,
                "psnr_db": psnr, "valid_pixel_ratio": float(result.valid_mask.mean()),
                "hole_pixel_ratio": float(1 - result.valid_mask.mean()),
                "output_save_seconds": save_seconds, **timing, **result.metadata,
            }
            write_json(identity_dir / "metadata.json", row)
            identities.append(row)
            timing_rows.append(row.copy())
            source_data[stem]["identity"][renderer_name] = (result, row, diff)

        for preset_name, preset in presets.items():
            state = camera_state(k, preset)
            source_data[stem]["results"][preset_name] = {}
            # Direct production result is the independent regression oracle.
            baseline = DepthRenderer(1, renderer='nearest_z').render(source, state, depth, fill_color, geometry=geometry)
            for renderer_name in RENDERER_NAMES:
                renderer = renderers[renderer_name]
                print(f"  {preset_name}/{renderer_name}: warmup + {repetitions}", flush=True)
                result, timing = measure_renderer(
                    lambda r=renderer, s=state: r.render(source, s, depth, fill_color, geometry=geometry),
                    repetitions)
                case_dir = source_dir / preset_name / renderer_name
                save_seconds = save_result(result, case_dir)
                ratio = float(result.valid_mask.mean())
                row = {
                    "source": source_path.name, "kind": "formal", "preset": preset_name,
                    "renderer": renderer_name, "status": "success",
                    "valid_pixel_ratio": ratio, "hole_pixel_ratio": 1 - ratio,
                    "output_save_seconds": save_seconds, **timing, **result.metadata,
                }
                write_json(case_dir / "metadata.json", row)
                metric_rows.append(row)
                timing_rows.append(row.copy())
                source_data[stem]["results"][preset_name][renderer_name] = (result, row)
                if renderer_name == "nearest_z":
                    rgb_exact = np.array_equal(np.asarray(result.image), np.asarray(baseline.image))
                    mask_exact = np.array_equal(result.valid_mask, baseline.valid_mask)
                    regression = {"source": source_path.name, "preset": preset_name,
                                  "rgb_exact_match": rgb_exact, "valid_mask_exact_match": mask_exact}
                    regressions.append(regression)
                    if not rgb_exact or not mask_exact:
                        raise AssertionError(f"nearest_z production regression failed: {stem}/{preset_name}")

            cells = [(source, "SOURCE", "")]
            for renderer_name in RENDERER_NAMES:
                result, row = source_data[stem]["results"][preset_name][renderer_name]
                cells.append((result.image, renderer_name,
                              f"valid {row['valid_pixel_ratio']:.1%} | {row['render_median_seconds']:.3f}s"))
            compare_grid = labelled_grid(cells, 4, title=f"{stem} / {preset_name}")
            compare_grid.save(source_dir / "comparisons" / f"{preset_name}_renderer_compare.jpg", quality=94)
            compare_grid.save(source_dir / preset_name / "renderer_compare.jpg", quality=94)

        # Per-source formal and identity grids.
        formal_cells = []
        for preset_name in presets:
            formal_cells.append((source, f"{preset_name}: SOURCE", ""))
            for renderer_name in RENDERER_NAMES:
                result, row = source_data[stem]["results"][preset_name][renderer_name]
                formal_cells.append((result.image, renderer_name,
                                     f"valid {row['valid_pixel_ratio']:.1%} | {row['render_median_seconds']:.3f}s"))
        labelled_grid(formal_cells, 4, title=f"{stem}: renderer interpolation ablation").save(
            source_dir / "all_renderer_results_grid.jpg", quality=94)
        identity_cells = [(source, "SOURCE", "")]
        diff_cells = []
        for renderer_name in RENDERER_NAMES:
            result, row, diff = source_data[stem]["identity"][renderer_name]
            identity_cells.append((result.image, renderer_name,
                                   f"MAE {row['global_rgb_mae']:.3f} | PSNR {row['psnr_db']:.2f}\nvalid {row['valid_pixel_ratio']:.1%} | {row['render_median_seconds']:.3f}s"))
            diff_cells.append((diff, renderer_name, "abs diff x4"))
        labelled_grid(identity_cells, 4, title=f"{stem}: identity").save(
            source_dir / "identity_renderer_compare.jpg", quality=94)
        labelled_grid(diff_cells, 3, title=f"{stem}: identity absolute difference x4").save(
            source_dir / "identity_diff_grid.jpg", quality=94)
        write_json(output / "experiment_summary.json", {**summary, "images": image_records,
                                                         "baseline_regression": regressions})

    # Cross-scene overviews.
    for preset_name in presets:
        cells = []
        for stem, data in source_data.items():
            for renderer_name in RENDERER_NAMES:
                result, row = data["results"][preset_name][renderer_name]
                cells.append((result.image, f"{stem} / {renderer_name}",
                              f"valid {row['valid_pixel_ratio']:.1%} | {row['render_median_seconds']:.3f}s"))
        labelled_grid(cells, 3, title=f"Cross-scene: {preset_name}").save(
            output / "overview" / f"{preset_name}_compare.jpg", quality=94)
    cells = []
    for stem, data in source_data.items():
        for renderer_name in RENDERER_NAMES:
            result, row, _ = data["identity"][renderer_name]
            cells.append((result.image, f"{stem} / {renderer_name}",
                          f"MAE {row['global_rgb_mae']:.3f} | PSNR {row['psnr_db']:.2f}"))
    labelled_grid(cells, 3, title="Cross-scene identity").save(
        output / "overview" / "identity_compare.jpg", quality=94)

    # Aggregate quality, latency, and estimated end-to-end costs.
    quality_rows, e2e_values, latency_values = [], {}, {}
    shared_by_source = {record["source"]: record["shared_timing"] for record in image_records}
    for renderer_name in RENDERER_NAMES:
        formal = [row for row in metric_rows if row["renderer"] == renderer_name]
        identity = [row for row in identities if row["renderer"] == renderer_name]
        all_samples = [sample for row in formal for sample in row["render_samples_seconds"]]
        case_medians = [row["render_median_seconds"] for row in formal]
        shared_frontend = statistics.mean(
            sum(value for key, value in shared.items() if not key.endswith("excluded"))
            for shared in shared_by_source.values())
        output_save = statistics.mean(row["output_save_seconds"] for row in formal)
        render_median = statistics.median(case_medians)
        estimated_e2e = shared_frontend + render_median + output_save
        quality = {
            "renderer": renderer_name,
            "identity_mae_mean": statistics.mean(row["global_rgb_mae"] for row in identity),
            "identity_mae_median": statistics.median(row["global_rgb_mae"] for row in identity),
            "identity_psnr_mean": statistics.mean(row["psnr_db"] for row in identity),
            "identity_valid_ratio_mean": statistics.mean(row["valid_pixel_ratio"] for row in identity),
            "valid_ratio_mean": statistics.mean(row["valid_pixel_ratio"] for row in formal),
            "hole_ratio_mean": statistics.mean(row["hole_pixel_ratio"] for row in formal),
            "render_median_seconds": render_median,
            "render_mean_seconds": statistics.mean(all_samples),
            "render_p90_seconds": float(np.percentile(all_samples, 90)),
            "shared_frontend_seconds": shared_frontend,
            "output_save_mean_seconds": output_save,
            "estimated_e2e_seconds": estimated_e2e,
        }
        quality_rows.append(quality)
        latency_values[renderer_name] = render_median
        e2e_values[renderer_name] = estimated_e2e

    write_csv(output / "renderer_metrics.csv", metric_rows)
    write_json(output / "renderer_metrics.json", metric_rows)
    write_csv(output / "timing_summary.csv", timing_rows)
    write_json(output / "timing_summary.json", timing_rows)
    write_csv(output / "renderer_quality_summary.csv", quality_rows)
    bar_chart(latency_values, output / "renderer_latency_bar.png",
              "Renderer latency (12 formal cases)", "Median of per-case median render seconds")
    bar_chart(e2e_values, output / "estimated_e2e_bar.png",
              "Estimated single-sketch end-to-end latency", "Shared depth/geometry/save + renderer median")
    summary.update({
        "status": "completed", "images": image_records,
        "formal_case_count": len(metric_rows), "identity_case_count": len(identities),
        "success_case_count": len(metric_rows), "failed_case_count": 0,
        "baseline_regression": regressions,
        "baseline_regression_all_exact": all(row["rgb_exact_match"] and row["valid_mask_exact_match"] for row in regressions),
        "quality_summary": quality_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "PreparedPointCloud+CameraState to RGB+valid mask+projected depth; excludes all file I/O.",
        "depth_postprocess_scope": "Instrumented depth postprocess is model-output resize; pseudo-Z mapping overhead is not separately instrumented and is excluded from this estimate.",
        "e2e_formula": "shared mean(depth preprocess + inference + resize/mapping postprocess + geometry) + renderer median + renderer-specific mean result PNG save",
    })
    write_json(output / "experiment_summary.json", summary)
    print(f"completed: {len(metric_rows)} formal, {len(identities)} identity", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path,
                        default=Path("workdir/depth_models/depth-anything-v2-small"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--depth-rel-tol", type=float, default=0.01)
    parser.add_argument("--gaussian-sigma", type=float, default=0.75)
    parser.add_argument("--blank-color", type=int, nargs=3, default=(128, 128, 128))
    args = parser.parse_args(argv)
    run_batch(args.input_dir, args.output_dir, model=args.depth_model, device=args.device,
              repetitions=args.repetitions, depth_rel_tol=args.depth_rel_tol,
              gaussian_sigma=args.gaussian_sigma, fill_color=tuple(args.blank_color))


if __name__ == "__main__":
    main()
