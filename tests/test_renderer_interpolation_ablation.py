"""Synthetic checks for the research-only renderer interpolation ablation."""
import numpy as np
import pytest
from PIL import Image

from experiments.depth_pointcloud_sketch.run_batch import camera_state
from experiments.renderer_interpolation_ablation.renderers import (
    ResearchPointRenderer,
    RendererConfig,
    bilinear_footprint,
    create_renderers,
    gaussian_footprint,
)
from experiments.renderer_interpolation_ablation.run_batch import (
    load_presets,
    measure_renderer,
)
from photography_viewpoint_agent.camera_rendering.camera_state import CameraState
from photography_viewpoint_agent.camera_rendering.depth_renderer import (
    DepthRenderer,
    PreparedPointCloud,
    prepare_point_cloud,
    project_cloud,
)
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics
from photography_viewpoint_agent.config.settings import AgentConfig


def scene(width=16, height=12):
    rng = np.random.default_rng(8)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    depth = np.ones((height, width), np.float64)
    depth[:, width // 2:] = 2
    image = Image.fromarray(pixels)
    k = CameraIntrinsics.from_horizontal_fov(width, height)
    return image, depth, k, prepare_point_cloud(image, depth, k)


def test_nearest_reference_matches_research_baseline():
    image, depth, k, cloud = scene()
    state = CameraState(source_intrinsics=k, target_intrinsics=k,
                        translation={"x": .03}, rotation={"yaw_deg": -5})
    production = DepthRenderer(renderer='nearest_z').render(image, state, depth, geometry=cloud)
    research = ResearchPointRenderer("nearest_z").render(image, state, depth, geometry=cloud)
    assert DepthRenderer().radius == 1
    np.testing.assert_array_equal(np.asarray(research.image), np.asarray(production.image))
    np.testing.assert_array_equal(research.valid_mask, production.valid_mask)
    np.testing.assert_array_equal(research.debug_info["projected_depth"],
                                  production.debug_info["projected_depth"])


def test_production_defaults_to_bilinear_and_reconstructs_identity():
    image, depth, k, cloud = scene()
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    result = DepthRenderer().render(image, state, depth, geometry=cloud)
    assert AgentConfig().point_renderer == 'bilinear'
    assert result.metadata['point_renderer'] == 'bilinear'
    assert result.valid_mask.all()
    assert np.abs(np.asarray(image).astype(float) - np.asarray(result.image)).mean() < .05


def test_renderer_selection_and_invalid_name():
    image, depth, k, cloud = scene()
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    nearest = DepthRenderer(renderer='nearest_z').render(image, state, depth, geometry=cloud)
    assert nearest.metadata['point_renderer'] == 'nearest_z'
    with pytest.raises(ValueError, match='Unsupported renderer: gaussian'):
        DepthRenderer(renderer='gaussian')
    with pytest.raises(ValueError):
        AgentConfig(point_renderer='gaussian')


def test_production_bilinear_depth_gate_preserves_front_color():
    k = CameraIntrinsics.from_horizontal_fov(8, 8)
    cloud = PreparedPointCloud(
        ids=np.array([0, 1]),
        vertices=np.array([[0., 0., 1.], [0., 0., 10.]]),
        colors=np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8),
        source_intrinsics=k,
    )
    result = DepthRenderer().render(
        Image.new('RGB', k.size), CameraState(source_intrinsics=k, target_intrinsics=k),
        np.ones((8, 8)), geometry=cloud)
    np.testing.assert_array_equal(np.asarray(result.image)[4, 4], [255, 0, 0])


@pytest.mark.parametrize("renderer_name", ["bilinear", "gaussian"])
def test_weighted_identity_does_not_change_projected_geometry(renderer_name):
    image, depth, k, cloud = scene()
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    before = project_cloud(cloud, state)
    ResearchPointRenderer(renderer_name).render(image, state, depth, geometry=cloud)
    after = project_cloud(cloud, state)
    for left, right in zip(before[:3], after[:3]):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("renderer_name", ["nearest_z", "bilinear", "gaussian"])
def test_all_renderers_return_rgb_mask_and_depth(renderer_name):
    image, depth, k, cloud = scene()
    state = CameraState(source_intrinsics=k, target_intrinsics=k, translation={"z": .05})
    result = ResearchPointRenderer(renderer_name).render(image, state, depth, geometry=cloud)
    assert result.image.mode == "RGB" and result.image.size == image.size
    assert result.valid_mask.shape == depth.shape and result.valid_mask.dtype == bool
    assert result.debug_info["projected_depth"].shape == depth.shape


def test_bilinear_weights_sum_to_one():
    u, v = np.array([2.2, 7.75]), np.array([3.4, 1.1])
    footprint = bilinear_footprint(u, v)
    np.testing.assert_allclose(sum(weight for _, _, weight in footprint), 1)


def test_gaussian_weights_are_positive():
    footprint = gaussian_footprint(np.array([2.2]), np.array([3.4]), radius=1, sigma=.75)
    assert len(footprint) == 9
    assert all(np.all(weight > 0) for _, _, weight in footprint)


@pytest.mark.parametrize("renderer_name", ["bilinear", "gaussian"])
def test_depth_gate_prevents_foreground_background_color_bleeding(renderer_name):
    k = CameraIntrinsics.from_horizontal_fov(8, 8)
    # Two points on the optical axis land at exactly the same target pixel.
    cloud = PreparedPointCloud(
        ids=np.array([0, 1]),
        vertices=np.array([[0., 0., 1.], [0., 0., 10.]]),
        colors=np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8),
        source_intrinsics=k,
    )
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    result = ResearchPointRenderer(renderer_name).render(
        Image.new("RGB", k.size), state, np.ones((8, 8)), geometry=cloud)
    np.testing.assert_array_equal(np.asarray(result.image)[4, 4], [255, 0, 0])


@pytest.mark.parametrize("renderer_name", ["nearest_z", "bilinear", "gaussian"])
def test_three_renderers_run_all_four_presets(renderer_name):
    image, depth, k, cloud = scene()
    for preset in load_presets().values():
        result = ResearchPointRenderer(renderer_name).render(
            image, camera_state(k, preset), depth, geometry=cloud)
        assert result.valid_mask.any()


def test_timing_has_five_repetitions_and_never_saves(monkeypatch):
    image, depth, k, cloud = scene(8, 6)
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    renderer = ResearchPointRenderer("bilinear")
    calls = {"count": 0}

    def render():
        calls["count"] += 1
        return renderer.render(image, state, depth, geometry=cloud)

    def forbidden_save(*args, **kwargs):
        raise AssertionError("file saving entered renderer timing")

    monkeypatch.setattr(Image.Image, "save", forbidden_save)
    _, timing = measure_renderer(render, repetitions=5, warmups=1)
    assert calls["count"] == 6
    assert timing["timing_repetitions"] == 5
    assert len(timing["render_samples_seconds"]) == 5


def test_renderer_configuration_is_recorded():
    image, depth, k, cloud = scene()
    config = RendererConfig(depth_rel_tol=.02, gaussian_sigma=.6)
    result = create_renderers(config)["gaussian"].render(
        image, CameraState(source_intrinsics=k, target_intrinsics=k), depth, geometry=cloud)
    assert result.metadata["depth_rel_tol"] == .02
    assert result.metadata["gaussian_sigma"] == .6
    assert result.metadata["splat_radius"] == 1
