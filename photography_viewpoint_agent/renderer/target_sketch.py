"""Whole-scene depth + point-cloud target sketches; no independent subject edits."""
from pathlib import Path
import numpy as np
from PIL import Image
from photography_viewpoint_agent.camera_rendering.depth_renderer import DepthRenderer, prepare_point_cloud
from photography_viewpoint_agent.camera_rendering.intent import camera_from_target
from photography_viewpoint_agent.depth_estimation.storage import load_depth
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.tools.plan_validator import validate_plan


class TargetSketchRenderer:
    def __init__(self, width=1280, height=720, fill_color=(128, 128, 128), debug=False,
                 splat_radius=1, viewpoint_horizontal_fov_deg=60, renderer='bilinear'):
        self.size = (width, height)
        self.fill_color, self.debug = fill_color, debug
        self.viewpoint_fov = viewpoint_horizontal_fov_deg
        self.renderer = DepthRenderer(splat_radius, renderer=renderer)

    def render(self, reference_frame, reference_subject, target_state, output_path, *, reference_depth=None):
        target = validate_plan(target_state)
        if reference_depth is None:
            raise ValueError('Every target sketch requires reference_depth; no renderer fallback')
        if reference_subject is not None:
            raise ValueError('Independent subject processing is unsupported; render the whole scene')
        with Image.open(reference_frame.path) as image:
            source = image.convert('RGB')
        if source.size != (reference_frame.width, reference_frame.height):
            raise ValueError('Reference frame dimensions do not match RGB')
        depth = load_depth(reference_depth, source.size)
        scene = float(np.median(depth[depth > 0]))
        normalized = depth / scene if reference_depth.metric else depth
        camera, viewport = camera_from_target(target, source.size, self.size, self.viewpoint_fov)
        geometry = prepare_point_cloud(source, normalized, camera.source_intrinsics)
        rendered = self.renderer.render(source, camera, normalized, self.fill_color, geometry=geometry)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.image.save(output_path)
        mask_path = output_path.with_name(output_path.stem + '_valid_mask.png')
        Image.fromarray(rendered.valid_mask.astype(np.uint8) * 255).save(mask_path)
        coverage = float(rendered.valid_mask.mean())
        center, rotation = camera.camera_center(), camera.rotation_matrix()
        warp = {**rendered.metadata, 'backend': 'depth_3d', 'renderer': 'point_cloud',
            'parameters': camera.warp_parameters().model_dump(),
            'camera_state': camera.model_dump(mode='json'),
            'source_intrinsics': camera.source_intrinsics.matrix().tolist(),
            'target_intrinsics': camera.target_intrinsics.matrix().tolist(),
            'rotation': rotation.tolist(), 'camera_center_source_normalized': center.tolist(),
            'extrinsic_translation_normalized': (-rotation @ center).tolist(),
            'translation_units': 'fraction of scene median depth; not meters',
            'scene_reference_depth': scene, 'metric_depth': reference_depth.metric,
            'depth_observation': reference_depth.model_dump(mode='json'),
            'depth_backend': reference_depth.metadata.get('model', reference_depth.metadata.get('source', 'injected')),
            'valid_fraction': coverage, 'hole_fraction': 1 - coverage,
            'point_count': len(geometry.ids), 'source_geometry_builds': 1,
            'fill_color': list(self.fill_color), 'no_subject_edit': True, 'no_post_render_framing': True,
            'intrinsics_policy': 'configured source FOV; framing/focal scale applied in target K'}
        if self.debug:
            np.save(output_path.with_name('projected_depth.npy'), rendered.debug_info['projected_depth'], allow_pickle=False)
        meta = RenderMeta(viewpoint_applied=bool(target.viewpoint.active),
            viewpoint_rotation=target.viewpoint.rotation, camera_translation=target.viewpoint.translation,
            viewpoint_warp=warp, requested_viewport=target.framing.reference_viewport,
            rendered_viewport=viewport, focal_scale=target.framing.focal_scale,
            reference_size=source.size, valid_pixel_ratio=coverage, hole_pixel_ratio=1 - coverage,
            valid_mask_path=str(mask_path.resolve()))
        output_path.with_suffix('.json').write_text(meta.model_dump_json(indent=2), encoding='utf-8')
        return str(output_path.resolve()), meta
