"""Projection oracles and production workflow checks, without downloading models."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.camera_rendering import CameraRenderer, CameraState, CameraIntrinsics
from photography_viewpoint_agent.camera_rendering.depth_renderer import DepthRenderer, prepare_point_cloud, project_cloud
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.depth_estimation.storage import save_observation
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch
from experiments.depth_pointcloud_sketch.run_batch import load_presets, DEFAULT_PRESETS, camera_state


class CountingDepth:
    def __init__(self, metric=False):
        self.calls, self.metric = 0, metric

    def estimate(self, frame, output_path):
        self.calls += 1
        depth = np.ones((frame.height, frame.width), np.float32)
        depth[:, frame.width // 2:] = 2
        return save_observation(depth * (7 if self.metric else 1), frame, output_path,
                                {'source': 'synthetic'}, metric=self.metric)


def make_video(path, count=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    try:
        for i in range(count):
            writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    finally:
        writer.release()


def preset_target(preset):
    return TargetState.model_validate({'viewpoint': {'mode': 'camera', **preset['rotation'],
        **{'translation_' + k: v for k, v in preset['translation'].items()}},
        'framing': {'focal_scale': preset['focal_scale']}})


@pytest.mark.parametrize('name', list(load_presets(DEFAULT_PRESETS)))
def test_all_experiment_presets_match_production(frame, tmp_path, name):
    pixels = np.random.default_rng(12).integers(0, 256, (frame.height, frame.width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(frame.path)
    preset = load_presets(DEFAULT_PRESETS)[name]
    target = preset_target(preset)
    obs = CountingDepth().estimate(frame, tmp_path / 'depth.npy')
    renderer = TargetSketchRenderer(100, 100)
    path, meta = renderer.render(frame, None, target, tmp_path / 'target.png', reference_depth=obs)
    k = CameraIntrinsics.from_horizontal_fov(100, 100)
    source = Image.open(frame.path).convert('RGB')
    baseline = DepthRenderer().render(source, camera_state(k, preset), np.load(obs.depth_path))
    np.testing.assert_array_equal(np.asarray(Image.open(path)), np.asarray(baseline.image))
    np.testing.assert_array_equal(np.asarray(Image.open(meta.valid_mask_path)) > 0, baseline.valid_mask)
    assert validate_sketch(target, meta, path, AgentConfig(target_width=100, target_height=100)).passed


@pytest.mark.parametrize('intent', [{}, {'viewpoint': {'mode': 'rotation', 'yaw_deg': 5}},
    {'framing': {'reference_viewport': [-.2, -.2, 1.2, 1.2], 'focal_scale': .9}},
    {'viewpoint': {'mode': 'camera', 'translation_y': .05, 'pitch_deg': -8}}])
def test_workflow_always_estimates_once_and_projects(tmp_path, intent):
    video = tmp_path / 'input.avi'
    make_video(video)
    estimator = CountingDepth()
    config = AgentConfig(target_width=64, target_height=48, frame_sample_interval=3,
                         work_dir=str(tmp_path / 'run'))
    graph = build_workflow(config, depth_estimator=estimator)
    result = graph.invoke({'video_path': str(video), 'manual_target_state': intent})
    assert result.get('error') is None
    assert result['validation'].passed and estimator.calls == 1
    assert result['current_frame'].frame_index == 7
    assert result['reference_frame'].frame_index in (0, 3, 6, 7)
    assert 'reference_subject' not in result and 'detect_reference_subject' not in graph.get_graph().nodes
    warp = result['render_meta'].viewpoint_warp
    assert warp['renderer'] == 'point_cloud' and not warp['identity_copy_fast_path']
    assert Path(result['target_sketch_path']).suffix == '.png'


def test_depth_failure_does_not_fallback(tmp_path):
    class Broken:
        def estimate(self, *args):
            raise RuntimeError('depth model unavailable')
    video = tmp_path / 'input.avi'
    make_video(video)
    result = build_workflow(AgentConfig(work_dir=str(tmp_path / 'run')), depth_estimator=Broken()).invoke(
        {'video_path': str(video), 'manual_target_state': {}})
    assert 'estimate_reference_depth' in result['error']
    assert 'target_sketch_path' not in result


@pytest.mark.parametrize('field,value', [('viewpoint_backend', 'homography'),
    ('viewpoint_backend', 'depth_mesh'), ('viewpoint_border_mode', 'replicate'), ('mesh_stride', 2)])
def test_removed_config_rejected(field, value):
    with pytest.raises(ValueError):
        AgentConfig(**{field: value})


def test_subject_edits_and_invalid_motion_rejected():
    for payload in ({'subject': {'mode': 'reposition', 'bbox': [.1, .2, .5, .8]}},
                    {'viewpoint': {'translation_x': .05}},
                    {'viewpoint': {'mode': 'camera', 'translation_y': .21}},
                    {'framing': {'focal_scale': float('nan')}},
                    {'framing': {'reference_viewport': [0, 0, 0, 1]}}):
        with pytest.raises(ValueError):
            TargetState.model_validate(payload)


def test_projection_translation_signs_and_parallax():
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    depth = np.ones((48, 64)); depth[:, 32:] = 2
    cloud = prepare_point_cloud(Image.new('RGB', k.size), depth, k)
    for axis in ('x', 'y', 'z'):
        state = CameraState(source_intrinsics=k, target_intrinsics=k, translation={axis: .05})
        ids, uv, *_ = project_cloud(cloud, state)
        x, y, z = ids % 64, ids // 64, depth.ravel()[ids]
        if axis == 'x':
            np.testing.assert_allclose(uv[0], x - k.fx * .05 / z)
        elif axis == 'y':
            np.testing.assert_allclose(uv[1], y + k.fy * .05 / z)
        else:
            np.testing.assert_allclose(uv[0], k.cx + (x - k.cx) * z / (z - .05))
    for field, expected in [('yaw_deg', -1), ('pitch_deg', 1)]:
        state = CameraState(source_intrinsics=k, target_intrinsics=k, rotation={field: 8})
        ids, uv, *_ = project_cloud(cloud, state)
        mid = np.flatnonzero(ids == 24 * 64 + 32)[0]
        index, center = (0, k.cx) if field == 'yaw_deg' else (1, k.cy)
        assert np.sign(uv[index, mid] - center) == expected


def test_identity_uses_projection_and_nearest_surface_occlusion():
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    pixels = np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    depth = np.ones((48, 64))
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    result = DepthRenderer(0).render(image, state, depth)
    np.testing.assert_array_equal(pixels, np.asarray(result.image))
    assert not result.metadata['identity_copy_fast_path']
    # Two source rays collapse into one pixel under narrow focal scaling;
    # the front surface wins even if the farther point has a closer splat center.
    depth[:] = 0; depth[24, 32] = 1; depth[24, 33] = 2
    pixels[24, 32] = [255, 0, 0]; pixels[24, 33] = [0, 0, 255]
    small = k.model_copy(update={'fx': k.fx * .1, 'fy': k.fy * .1})
    result = DepthRenderer(0).render(Image.fromarray(pixels),
        CameraState(source_intrinsics=k, target_intrinsics=small), depth)
    np.testing.assert_array_equal(np.asarray(result.image)[24, 32], [255, 0, 0])


def test_cached_camera_depth_and_geometry(tmp_path):
    estimator = CountingDepth()
    renderer = CameraRenderer(work_dir=tmp_path, depth_estimator=estimator)
    image = Image.new('RGB', (64, 48), 'red')
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    cloud = None
    for params in ({}, {'rotation': {'pitch_deg': -8}}, {'translation': {'y': .05}}):
        result = renderer.render(image, CameraState(source_intrinsics=k, target_intrinsics=k, **params))
        assert result.renderer_type == 'depth_3d'
        if cloud is not None:
            assert renderer._geometry is cloud
        cloud = renderer._geometry
    assert estimator.calls == 1
    renderer.render(Image.new('RGB', (64, 48), 'blue'), CameraState(source_intrinsics=k, target_intrinsics=k))
    assert estimator.calls == 2 and renderer._geometry is not cloud


def test_framing_intrinsics_preserve_aspect_without_post_resize():
    from photography_viewpoint_agent.camera_rendering.intent import camera_from_target
    target = TargetState.model_validate({'framing': {'reference_viewport': [-.2, .1, 1.1, .9], 'focal_scale': 1.1},
        'viewpoint': {'mode': 'camera', 'translation_z': .05}})
    state, fitted = camera_from_target(target, (80, 100), (120, 80))
    cloud = prepare_point_cloud(Image.new('RGB', (80, 100)), np.ones((100, 80)), state.source_intrinsics)
    ids, uv, *_ = project_cloud(cloud, state)
    x, y = ids % 80, ids // 80
    # First translate the camera, then express the resulting rays in the fitted
    # viewport and apply the focal scale about the target canvas center.
    source_x = 40 + (x - 40) / .95
    source_y = 50 + (y - 50) / .95
    expected_x = 60 + (((source_x / 80 - fitted[0]) / (fitted[2] - fitted[0])) * 120 - 60) * 1.1
    expected_y = 40 + (((source_y / 100 - fitted[1]) / (fitted[3] - fitted[1])) * 80 - 40) * 1.1
    np.testing.assert_allclose(uv, np.stack((expected_x, expected_y)), atol=1e-10)
    assert state.target_intrinsics.fx == pytest.approx(state.target_intrinsics.fy)


@pytest.mark.parametrize('depth', [np.zeros((48, 64)), np.full((48, 64), np.nan), np.ones((47, 64))])
def test_invalid_depth_never_renders(depth):
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    with pytest.raises(ValueError):
        DepthRenderer().render(Image.new('RGB', k.size), CameraState(source_intrinsics=k, target_intrinsics=k), depth)


def test_metric_normalization_and_metadata_tampering(frame, tmp_path):
    target = TargetState.model_validate({'viewpoint': {'mode': 'camera', 'translation_y': .05}})
    renderer = TargetSketchRenderer(100, 100)
    results = []
    for metric in (False, True):
        obs = CountingDepth(metric).estimate(frame, tmp_path / f'depth_{metric}.npy')
        results.append(renderer.render(frame, None, target, tmp_path / f'result_{metric}.png', reference_depth=obs))
    np.testing.assert_array_equal(np.asarray(Image.open(results[0][0])), np.asarray(Image.open(results[1][0])))
    path, meta = results[0]
    config = AgentConfig(target_width=100, target_height=100)
    assert validate_sketch(target, meta, path, config).passed
    for update in ({'camera_translation': (0, 0, 0)}, {'hole_pixel_ratio': .99}, {'viewpoint_warp': {}},
                   {'focal_scale': 1.1}, {'valid_mask_path': str(tmp_path / 'missing.png')}):
        assert not validate_sketch(target, meta.model_copy(update=update), path, config).passed


def test_cli_precomputed_and_output_protection(tmp_path):
    from photography_viewpoint_agent.app import main
    video = tmp_path / 'input.avi'; make_video(video)
    depth = tmp_path / 'depth.npy'; np.save(depth, np.ones((48, 64), np.float32))
    target = tmp_path / 'target.json'
    target.write_text(json.dumps({'viewpoint': {'mode': 'camera', 'translation_z': .05}}))
    args = ['--video', str(video), '--target-state', str(target), '--depth-path', str(depth),
            '--work-dir', str(tmp_path / 'run'), '--target-width', '64', '--target-height', '48']
    assert main(args) == 0
    assert main(args) == 1
    output = json.loads((tmp_path / 'run/result.json').read_text())
    assert output['render_meta']['camera_translation'] == [0, 0, .05]
    assert output['validation']['passed']
