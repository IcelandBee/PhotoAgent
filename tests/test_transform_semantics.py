"""Semantic boundaries and operation ordering, independent of renderer quality."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.renderer.viewpoint_warp import RotationViewpointWarper
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.renderer.mesh_warp import MeshViewpointWarper
from photography_viewpoint_agent.schemas.target import TargetState, ViewpointTarget
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.tools.plan_validator import validate_plan
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch
from photography_viewpoint_agent.depth_estimation.storage import save_observation
from .helpers.mesh_oracle import CpuTriangleOracle
from .test_depth3d import make_video


def target(yaw=0, shifted=False, subject=False):
    return TargetState.model_validate({
        'viewpoint': {'mode': 'rotation' if yaw else 'none', 'yaw_deg': yaw},
        'framing': {'reference_viewport': [.1, 0, 1.1, 1] if shifted else [0, 0, 1, 1]},
        'subject': {'mode': 'reposition', 'bbox': [.65, .25, .85, .85]} if subject else {'mode': 'follow_reference'},
    })


@pytest.fixture
def scene(tmp_path):
    y, x = np.indices((80, 100))
    pixels = np.stack((x * 2, y * 3, x + y), axis=-1).astype(np.uint8)
    pixels[20:60, 40:60] = (20, 250, 20)
    image = Image.fromarray(pixels)
    path = tmp_path / 'reference.png'
    image.save(path)
    mask = np.zeros((80, 100), np.uint8)
    mask[20:60, 40:60] = 255
    mask_path = tmp_path / 'mask.png'
    Image.fromarray(mask).save(mask_path)
    frame = FrameInfo(frame_id='reference', path=str(path), frame_index=0, timestamp=0, width=100, height=80)
    subject = SubjectObservation(bbox=(.4, .25, .6, .75), mask_path=str(mask_path), confidence=1)
    return image, frame, subject


@pytest.mark.parametrize('backend', ['homography', 'depth_3d', 'depth_mesh'])
@pytest.mark.parametrize('subject,shifted', [(False, True), (True, False), (True, True)])
def test_zero_rotation_never_uses_viewpoint_or_depth(scene, tmp_path, backend, subject, shifted):
    _, frame, observation = scene
    wanted = target(shifted=shifted, subject=subject)
    expected, _ = TargetSketchRenderer(100, 80).render(frame, observation if subject else None, wanted, tmp_path / 'expected.png')
    with patch('photography_viewpoint_agent.renderer.target_sketch.load_depth', side_effect=AssertionError('depth loaded')), \
         patch.object(RotationViewpointWarper, 'warp', side_effect=AssertionError('rotation called')), \
         patch('photography_viewpoint_agent.renderer.target_sketch.PointCloudViewpointWarper.warp', side_effect=AssertionError('point called')), \
         patch.object(MeshViewpointWarper, 'warp', side_effect=AssertionError('mesh called')):
        path, meta = TargetSketchRenderer(100, 80, viewpoint_backend=backend).render(
            frame, observation if subject else None, wanted, tmp_path / 'actual.png')
    assert Path(path).read_bytes() == Path(expected).read_bytes()
    assert meta.viewpoint_warp is None and not meta.viewpoint_applied
    assert meta.viewpoint_backend == 'none' and meta.camera_translation == (0, 0, 0)
    assert meta.subject_transform_applied == subject


@pytest.mark.parametrize('shifted', [False, True])
def test_yaw_precedes_framing_pixel_result(scene, tmp_path, shifted):
    image, frame, _ = scene
    wanted = target(yaw=5, shifted=shifted)
    path, meta = TargetSketchRenderer(100, 80).render(frame, None, wanted, tmp_path / 'actual.png')
    rotated, _ = RotationViewpointWarper().warp(image, yaw_deg=5)
    expected, _ = ViewportRenderer(100, 80).transform_background_by_viewport(rotated, wanted.framing.reference_viewport)
    np.testing.assert_array_equal(np.asarray(Image.open(path)), np.asarray(expected))
    if shifted:
        framed, _ = ViewportRenderer(100, 80).transform_background_by_viewport(image, wanted.framing.reference_viewport)
        wrong_order, _ = RotationViewpointWarper().warp(framed, yaw_deg=5)
        assert np.abs(np.asarray(expected).astype(float) - np.asarray(wrong_order)).mean() > 1
    assert meta.viewpoint_backend == 'homography' and meta.viewpoint_applied
    assert meta.viewpoint_rotation == {'yaw_deg': 5, 'pitch_deg': 0, 'roll_deg': 0}


@pytest.mark.parametrize('yaw,shifted', [(0, False), (5, False), (5, True)])
def test_subject_layout_is_last_and_bbox_is_final(scene, tmp_path, monkeypatch, yaw, shifted):
    import photography_viewpoint_agent.renderer.target_sketch as module
    image, frame, observation = scene
    wanted = target(yaw=yaw, shifted=shifted, subject=True)
    events = []
    rotate = RotationViewpointWarper.warp
    viewport = ViewportRenderer.transform_background_by_viewport
    place = module.transform_subject_by_bbox
    def wrapped_rotate(*args, **kwargs):
        events.append('rotation')
        return rotate(*args, **kwargs)
    def wrapped_viewport(*args, **kwargs):
        events.append('framing')
        return viewport(*args, **kwargs)
    def wrapped_place(*args, **kwargs):
        events.append('subject')
        return place(*args, **kwargs)
    monkeypatch.setattr(RotationViewpointWarper, 'warp', wrapped_rotate)
    monkeypatch.setattr(ViewportRenderer, 'transform_background_by_viewport', wrapped_viewport)
    monkeypatch.setattr(module, 'transform_subject_by_bbox', wrapped_place)
    path, meta = TargetSketchRenderer(100, 80).render(frame, observation, wanted, tmp_path / 'actual.png')
    assert events[-1] == 'subject'
    # A separate natural-reference projection is also computed for metadata.
    if yaw:
        assert events[-3:] == ['rotation', 'framing', 'subject']
    else:
        assert 'rotation' not in events
        assert events[-2:] == ['framing', 'subject']
    box = meta.rendered_subject_bbox
    assert (box[0] + box[2]) / 2 == pytest.approx(.75, abs=.01)
    assert box[3] == pytest.approx(.85, abs=.015)
    assert box[0] >= .65 - .01 and box[2] <= .85 + .01
    assert validate_sketch(wanted, meta, path, AgentConfig(target_width=100, target_height=80)).passed
    if not yaw and not shifted:
        # A region away from the removed/repositioned person stays fixed.
        np.testing.assert_array_equal(np.asarray(Image.open(path))[:10, :20], np.asarray(image)[:10, :20])


@pytest.mark.parametrize('backend', ['homography', 'depth_3d', 'depth_mesh'])
@pytest.mark.parametrize('yaw', [0, 5])
@pytest.mark.parametrize('subject', [False, True])
def test_depth_route_requires_active_rotation(tmp_path, backend, yaw, subject):
    video = tmp_path / 'video.avi'
    make_video(video)
    class Depth:
        calls = 0
        def estimate(self, frame, output):
            self.calls += 1
            return save_observation(np.ones((frame.height, frame.width)), frame, output, {})
    class Detector:
        def detect(self, frame, output):
            mask = np.zeros((frame.height, frame.width), np.uint8)
            mask[10:30, 20:30] = 255
            Image.fromarray(mask).save(output)
            return SubjectObservation(bbox=(20/64, 10/48, 30/64, 30/48), mask_path=str(output), confidence=1)
    depth = Depth()
    mesh = MeshViewpointWarper(rasterizer=CpuTriangleOracle())
    renderer = TargetSketchRenderer(64, 48, viewpoint_backend=backend, mesh_warper=mesh)
    config = AgentConfig(work_dir=str(tmp_path / 'run'), target_width=64, target_height=48, viewpoint_backend=backend)
    updates = list(build_workflow(config, renderer=renderer, depth_estimator=depth, detector=Detector()).stream({
        'video_path': str(video), 'manual_target_state': target(yaw=yaw, shifted=True, subject=subject)}, stream_mode='updates'))
    names = [key for update in updates for key in update]
    assert not any(value.get('error') for update in updates for value in update.values())
    needs_depth = bool(yaw and backend != 'homography')
    assert depth.calls == int(needs_depth)
    assert ('estimate_reference_depth' in names) == needs_depth
    meta = next(update['render_target_sketch']['render_meta'] for update in updates if 'render_target_sketch' in update)
    assert meta.camera_translation == (0, 0, 0)
    if needs_depth:
        assert meta.viewpoint_warp['camera_center_source'] == [0, 0, 0]
        assert all(meta.viewpoint_warp['parameters'][key] == 0 for key in ('translation_x', 'translation_y', 'translation_z'))
    validation = next(update['validate_sketch']['validation'] for update in updates if 'validate_sketch' in update)
    assert validation.passed, validation.messages


@pytest.mark.parametrize('field', ['translation_x', 'translation_y', 'translation_z'])
@pytest.mark.parametrize('value', [0, .03])
def test_public_target_and_planner_reject_even_zero_translation(field, value):
    data = target().model_dump()
    data['viewpoint'][field] = value
    with pytest.raises(ValidationError, match=field):
        TargetState.model_validate(data)
    with pytest.raises(ValidationError, match=field):
        validate_plan(data)


@pytest.mark.parametrize('field,value', [('mode', 'depth_3d'), ('mode', 'depth_mesh'),
                                       ('horizontal_fov_deg', 60), ('border_mode', 'constant')])
def test_renderer_options_not_semantic_fields(field, value):
    with pytest.raises(ValidationError):
        ViewpointTarget.model_validate({field: value})


def test_metadata_tampering_fails_validation(scene, tmp_path):
    _, frame, _ = scene
    wanted = target(yaw=5)
    config = AgentConfig(target_width=100, target_height=80)
    path, meta = TargetSketchRenderer(100, 80).render(frame, None, wanted, tmp_path / 'output.png')
    for change in ({'viewpoint_applied': False}, {'viewpoint_backend': 'depth_mesh'},
                   {'camera_translation': (.1, 0, 0)}, {'viewpoint_rotation': {'yaw_deg': -5}}):
        assert not validate_sketch(wanted, meta.model_copy(update=change), path, config).passed


def test_research_translation_runner_retains_motion(scene, tmp_path):
    from experiments.camera_translation.run import main
    _, frame, _ = scene
    depth = tmp_path / 'depth.npy'
    np.save(depth, np.ones((80, 100)))
    output = tmp_path / 'research'
    assert main(['--reference', frame.path, '--depth', str(depth), '--parameters',
                 'experiments/camera_translation/examples/51_depth3d__translate_right_small.json',
                 '--output-dir', str(output)]) == 0
    meta = json.loads((output / 'research_metadata.json').read_text(encoding='utf-8'))
    assert meta['research_only']
    assert meta['warp']['camera_center_source'][0] == .03


@pytest.mark.parametrize('backend', ['homography', 'depth_3d', 'depth_mesh'])
def test_rotation_mode_with_zero_angles_is_inactive(scene, tmp_path, backend):
    _, frame, _ = scene
    wanted = target().model_copy(update={'viewpoint': ViewpointTarget(mode='rotation')})
    config = AgentConfig(viewpoint_backend=backend)
    assert not config.needs_reference_depth(wanted.viewpoint)
    with patch.object(RotationViewpointWarper, 'warp', side_effect=AssertionError('warp')), \
         patch.object(MeshViewpointWarper, 'warp', side_effect=AssertionError('mesh')):
        _, meta = TargetSketchRenderer(100, 80, viewpoint_backend=backend).render(frame, None, wanted, tmp_path / 'zero.png')
    assert not meta.viewpoint_applied and meta.viewpoint_backend == 'none'


@pytest.mark.parametrize('field', ['translation_x', 'translation_y', 'translation_z'])
def test_integrated_workflow_rejects_translation_before_generator(tmp_path, field):
    from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow
    class Never:
        def invoke(self, *args):
            pytest.fail('Invalid semantics reached a graph')
    data = target().model_dump()
    data['viewpoint'][field] = 0
    with pytest.raises(ValidationError, match=field):
        build_integrated_workflow(target_graph=Never(), guide_graph=Never()).invoke({
            'video_path': 'unused.avi', 'manual_target_state': data, 'session_directory': str(tmp_path / 'session')})
