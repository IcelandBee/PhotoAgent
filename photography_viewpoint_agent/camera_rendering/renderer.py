"""Depth-only camera renderer with single-source depth and point-cloud caching."""
import hashlib
from pathlib import Path
import numpy as np
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.video import FrameInfo
from .camera_state import CameraState
from .depth_renderer import DepthRenderer, prepare_point_cloud


class CameraRenderer:
    def __init__(self, *, config=None, work_dir='workdir/pointcloud_depth',
                 depth_estimator=None):
        self.config = config or AgentConfig()
        self.work_dir = Path(work_dir)
        self._estimator = depth_estimator
        self._cache = None
        self.depth_estimate_calls = 0
        self._geometry = None

    def _depth(self, image):
        from photography_viewpoint_agent.depth_estimation.factory import create_depth_estimator, resolve_backend
        from photography_viewpoint_agent.depth_estimation.storage import load_depth, save_visualization
        key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        hit = self._cache is not None and self._cache[0] == key
        if not hit:
            directory = self.work_dir / key
            directory.mkdir(parents=True, exist_ok=True)
            source_path = directory / 'source.png'
            image.save(source_path)
            frame = FrameInfo(frame_id=key, path=str(source_path.resolve()), frame_index=0,
                              timestamp=0, width=image.width, height=image.height)
            if self._estimator is None:
                self._estimator = create_depth_estimator(self.config, 'depth_3d')
            self.depth_estimate_calls += 1
            observation = self._estimator.estimate(frame, directory / 'depth.npy')
            depth = load_depth(observation, image.size)
            save_visualization(depth, directory / 'depth.png')
            scene = float(np.median(depth[depth > 0]))
            render_depth = depth / scene if observation.metric else depth
            render_depth.setflags(write=False)
            self._cache = key, render_depth, observation, scene
            self._geometry = None
        _, depth, observation, scene = self._cache
        return depth, {
            'depth_backend': observation.metadata.get('depth_backend', resolve_backend(self.config, 'depth_3d')),
            'depth_estimator': type(self._estimator).__name__,
            'depth_observation': observation.model_dump(mode='json'),
            'depth_cache_hit': hit, 'depth_source_sha256': key, 'scene_reference_depth': scene,
            'depth_metric': observation.metric,
            'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest(),
            'intrinsics_policy': 'Explicit CameraState K; estimator focal metadata never overrides it',
        }

    def render(self, source_image, camera_state):
        # Revalidate even model_copy updates; fail before any inference or disk writes.
        state = CameraState.model_validate(camera_state.model_dump() if isinstance(camera_state, CameraState) else camera_state)
        image = source_image.convert('RGB')
        if image.size != state.source_intrinsics.size:
            raise ValueError('Source image dimensions do not match source intrinsics')
        norm = state.translation.norm
        depth, depth_metadata = self._depth(image)
        if self._geometry is None or self._geometry.source_intrinsics != state.source_intrinsics:
            self._geometry = prepare_point_cloud(image, depth, state.source_intrinsics)
        result = DepthRenderer(self.config.splat_radius, renderer=self.config.point_renderer).render(
            image, state, depth, self.config.fill_color, geometry=self._geometry)
        result.metadata.update(depth_metadata)
        r = state.rotation_matrix()
        center = state.camera_center()
        result.metadata.update({
            'source_intrinsics': state.source_intrinsics.matrix().tolist(),
            'target_intrinsics': state.target_intrinsics.matrix().tolist(), 'rotation': r.tolist(),
            'requested_translation': state.translation.model_dump(),
            'camera_center_source_normalized': center.tolist(),
            'extrinsic_translation_normalized': (-r @ center).tolist(),
            'translation_units': 'camera displacement / scene reference depth; not meters',
            'translation_norm': norm, 'translation_treated_as_zero': norm == 0,
            'renderer_backend': result.renderer_type, 'fill_color': list(self.config.fill_color),
            'border_mode': 'constant', 'valid_fraction': float(result.valid_mask.mean()),
            'image_size': list(result.image.size), 'depth_estimate_calls_total': self.depth_estimate_calls,
        })
        return result
