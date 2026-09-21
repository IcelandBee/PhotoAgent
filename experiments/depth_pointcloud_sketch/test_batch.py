"""Small synthetic integration checks; no downloaded models or production edits."""
import json
import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.depth_estimation.storage import save_observation
from photography_viewpoint_agent.camera_rendering.depth_renderer import project_explicit_intrinsics
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics
from experiments.depth_pointcloud_sketch import run_batch as batch


class CountingDepth:
    def __init__(self):
        self.calls = 0

    def estimate(self, frame, output_path):
        self.calls += 1
        depth = np.ones((frame.height, frame.width), np.float32)
        depth[:, frame.width // 2:] = 2
        return save_observation(depth, frame, output_path, {'model': 'synthetic'})


def inputs(tmp_path):
    directory = tmp_path / 'input'
    directory.mkdir()
    pixels = np.random.default_rng(2).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(directory / 'a.PNG')
    return directory


def test_all_presets_share_depth_and_only_point_cloud(tmp_path, monkeypatch):
    estimator = CountingDepth()
    output = tmp_path / 'out'
    summary = batch.run_batch(inputs(tmp_path), output, depth_estimator=estimator, save_debug=True)
    assert summary['status'] == 'completed'
    assert summary['success_case_count'] == 10 and summary['failed_case_count'] == 0
    assert estimator.calls == 1
    hashes = set()
    for record in summary['cases']:
        path = output / record['result_path']
        assert Image.open(path).size == (64, 48)
        assert Image.open(output / record['compare_path']).size == (128, 208)
        meta = json.loads(path.with_name('metadata.json').read_text())
        assert any(meta['translation'].values()) and meta['renderer'] == 'point_cloud'
        assert meta['renderer_debug']['geometry_reused']
        assert meta['renderer_debug']['implementation'] == 'explicit_intrinsics_point_splat_adapter'
        assert meta['valid_pixel_ratio'] + meta['hole_pixel_ratio'] == pytest.approx(1)
        hashes.add(meta['depth_sha256'])
        ks, kt = np.array(meta['source_intrinsics']), np.array(meta['target_intrinsics'])
        assert kt[0, 0] == pytest.approx(ks[0, 0] * meta['focal_scale'])
        assert kt[0, 2] == ks[0, 2]
    assert len(hashes) == 1
    directory = output / summary['images'][0]['directory']
    assert (directory / 'summary/all_results_grid.jpg').exists()
    assert (directory / 'summary/all_comparisons_grid.jpg').exists()
    assert summary['images'][0]['identity']['projection_max_error_px'] < 1e-6
    assert not summary['images'][0]['identity']['identity_copy_fast_path']


def test_failed_image_and_case_do_not_stop_batch(tmp_path, monkeypatch):
    directory = inputs(tmp_path)
    (directory / 'broken.webp').write_bytes(b'not an image')
    estimator = CountingDepth()
    original = batch.DepthRenderer.render
    def fail_one(self, image, state, depth, fill_color, **kwargs):
        if state.translation.z > 0 and state.rotation.pitch_deg < 0:
            raise RuntimeError('injected render failure')
        return original(self, image, state, depth, fill_color, **kwargs)
    monkeypatch.setattr(batch.DepthRenderer, 'render', fail_one)
    summary = batch.run_batch(directory, tmp_path / 'out', depth_estimator=estimator)
    assert summary['success_case_count'] == 9 and summary['failed_case_count'] == 11
    assert estimator.calls == 1
    assert summary['status'] == 'completed_with_errors'
    assert all(r['error'] for r in summary['cases'] if r['status'] == 'failed')


def test_presets_reject_zero_translation(tmp_path):
    presets = batch.load_presets(batch.DEFAULT_PRESETS)
    presets['crop_forward']['translation'] = {'x': 0, 'y': 0, 'z': 0}
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(presets))
    with pytest.raises(ValueError, match='nonzero'):
        batch.load_presets(path)


def test_forward_and_target_focal_geometric_oracle():
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    depth = np.ones((48, 64), np.float32)
    presets = batch.load_presets(batch.DEFAULT_PRESETS)
    for name, tz, scale in [('crop_forward', .05, 1), ('crop_forward_zoom_in', .05, 1.1),
                            ('expand_backward_wide', -.05, .9)]:
        state = batch.camera_state(k, presets[name])
        ids, uv, *_ = project_explicit_intrinsics(depth, state)
        expected_x = k.cx + (ids % 64 - k.cx) * scale / (1-tz)
        np.testing.assert_allclose(uv[0], expected_x, atol=1e-10)
