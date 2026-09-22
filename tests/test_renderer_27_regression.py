"""Batch invariants and timing boundaries for the 27-image research regression."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.renderer_interpolation_ablation import run_27_regression as regression
from experiments.renderer_interpolation_ablation.renderers import ResearchPointRenderer, RendererConfig
from photography_viewpoint_agent.depth_estimation.storage import save_observation


def test_previous_bilinear_math_and_defaults_are_unchanged():
    renderer = ResearchPointRenderer("bilinear")
    assert renderer.config == RendererConfig()
    assert renderer.config.depth_rel_tol == .01
    assert renderer.config.gaussian_sigma == .75


def test_selection_requires_exactly_27_images(tmp_path):
    for index in range(26):
        (tmp_path / f"image_{index:02d}.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="exactly 27"):
        regression.select_sources(tmp_path)
    (tmp_path / "image_26.png").write_bytes(b"x")
    assert len(regression.select_sources(tmp_path)) == 27


class FakeEstimator:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.raw_calls = 0
        self.instances.append(self)

    def estimate(self, frame, output_path):
        self.calls += 1
        return save_observation(np.ones((frame.height, frame.width), np.float32),
                                frame, output_path, {"source": "fake"})

    def estimate_raw_trace(self, frame):
        self.raw_calls += 1
        raw = np.arange(frame.width * frame.height, dtype=np.float32).reshape(frame.height, frame.width)
        timing = {"model_load_seconds": 0.0, "depth_preprocess_seconds": 0.0,
                  "depth_inference_seconds": 0.0, "depth_resize_postprocess_seconds": 0.0}
        return raw, raw, timing


def test_e2e_includes_saves_and_stage_sum_is_close(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (12, 10), "red").save(source_path)
    original_save = Image.Image.save
    calls = {"save": 0}

    def counted_save(self, *args, **kwargs):
        calls["save"] += 1
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", counted_save)
    row, cached = regression.single_sketch_timed(
        source_path, tmp_path / "out", FakeEstimator(), ResearchPointRenderer("bilinear"), mode="warm")
    assert calls["save"] >= 2
    assert (tmp_path / "out" / "target_sketch.png").is_file()
    assert (tmp_path / "out" / "valid_mask.png").is_file()
    assert (tmp_path / "out" / "metadata.json").is_file()
    assert row["save_seconds"] > 0
    assert row["stage_sum_gap_ratio"] < .1
    row2, _ = regression.single_sketch_timed(
        source_path, tmp_path / "out2", FakeEstimator(), ResearchPointRenderer("bilinear"),
        mode="same_source", cached=cached)
    assert row2["depth_inference_seconds"] == 0


def test_full_synthetic_27_image_batch_counts_and_outputs(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for index in range(27):
        name = (f"video{index:02d}_final_01_gpt.png" if index not in (4, 5, 23)
                else f"video{index}_final_01_gpt.png")
        Image.new("RGB", (12, 10), (index * 7 % 255, 30, 80)).save(input_dir / name)
    FakeEstimator.instances.clear()
    monkeypatch.setattr(regression, "MonocularDepthEstimator", FakeEstimator)
    out = tmp_path / "results"
    summary = regression.run(input_dir, out, tmp_path / "unused-model")
    assert summary["status"] == "completed"
    assert summary["selected_image_count"] == 27
    assert summary["depth_inference_calls_batch"] == 27
    assert summary["geometry_builds_batch"] == 27
    assert FakeEstimator.instances[0].calls == 27
    assert summary["formal_case_count"] == 216
    assert summary["identity_case_count"] == 54
    assert summary["paired_case_count"] == 108
    assert summary["baseline_regression_exact_count"] == 108
    assert summary["baseline_regression_all_exact"]
    for filename in ("renderer_regression_metrics.csv", "identity_metrics.csv",
                     "e2e_timing.csv", "timing_summary.json", "timing_summary.csv",
                     "experiment_summary.json", "renderer_latency_compare.png",
                     "e2e_latency_compare.png", "pipeline_breakdown_nearest.png",
                     "pipeline_breakdown_bilinear.png"):
        assert (out / filename).is_file(), filename
    assert len(list((out / "overview").glob("*_compare_part_*.jpg"))) == 15
    assert (out / "overview" / "largest_coverage_regressions.jpg").is_file()
