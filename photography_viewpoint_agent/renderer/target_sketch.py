from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.renderer.viewpoint_warp import RotationViewpointWarper, warp_bbox
from photography_viewpoint_agent.renderer.point_cloud_warp import PointCloudViewpointWarper, repair_background_depth
from photography_viewpoint_agent.renderer.mesh_warp import MeshViewpointWarper, camera_intrinsics
from photography_viewpoint_agent.depth_estimation.monocular import load_depth
import numpy as np
from photography_viewpoint_agent.tools.subject_utils import (
    load_subject_mask, extract_subject, build_background, transform_subject_by_bbox, composite,
)
from photography_viewpoint_agent.tools.geometry import (
    transform_bbox_by_viewport, clip_bbox_to_canvas, expected_subject_bbox, subject_is_noop,
)


class TargetSketchRenderer:
    def __init__(self, width=1280, height=720, fill_color=(128, 128, 128), debug=False,
                 subject_noop_position_threshold=0.001, subject_noop_scale_threshold=0.001,splat_radius=1,
                 mesh_stride=2,mesh_depth_edge_threshold=.12,mesh_device='cuda',mesh_warper=None):
        self.viewport_renderer = ViewportRenderer(width, height, fill_color)
        self.size, self.debug = (width, height), debug
        self.noop_position = subject_noop_position_threshold
        self.noop_scale = subject_noop_scale_threshold
        self.point_warper = PointCloudViewpointWarper(splat_radius)
        self.mesh_warper = mesh_warper if mesh_warper is not None else MeshViewpointWarper(mesh_stride,mesh_depth_edge_threshold,mesh_device)

    def preflight(self,mode):
        if mode=='depth_mesh':self.mesh_warper.check_available()

    def render(self, reference_frame: FrameInfo, reference_subject: SubjectObservation | None,
               target_state: TargetState, output_path: Path, *, reference_depth=None):
        with Image.open(reference_frame.path) as image:
            source = image.convert("RGB")
        viewpoint = target_state.viewpoint
        if viewpoint.mode in ('depth_3d','depth_mesh'):
            return self._render_depth(source,reference_subject,reference_depth,target_state,output_path)
        warp_meta = None
        warped_source = source
        def rotate(image):
            return RotationViewpointWarper().warp(image,
                **viewpoint.model_dump(exclude={'mode','translation_x','translation_y','translation_z'}),fill_color=self.viewport_renderer.fill_color)
        if viewpoint.active:
            warped_source, warp_meta = rotate(source)
        projected_box = reference_subject.bbox if reference_subject is not None else None
        if projected_box is not None and warp_meta is not None:
            projected_box = warp_bbox(projected_box,source.size,np.array(warp_meta['homography']))
        canvas, meta = self.viewport_renderer.transform_background_by_viewport(
            warped_source, target_state.framing.reference_viewport)
        natural = (transform_bbox_by_viewport(projected_box, meta.rendered_viewport)
                   if reference_subject is not None else None)
        meta = meta.model_copy(update={
            "viewpoint_warp": warp_meta,
            "subject_mode": target_state.subject.mode,
            "source_subject_bbox": reference_subject.bbox if reference_subject is not None else None,
            "natural_subject_bbox": natural,
        })
        if target_state.subject.mode == "follow_reference":
            return self._save_follow(canvas, meta, natural, output_path)
        if reference_subject is None:
            raise ValueError("reposition requires reference_subject detection")
        expected = expected_subject_bbox(reference_subject.bbox, source.size,
                                          target_state.subject.bbox, self.size)
        if not viewpoint.active and subject_is_noop(natural, expected, self.noop_position, self.noop_scale):
            return self._save_follow(canvas, meta, natural, output_path)
        mask, bounds = load_subject_mask(source, reference_subject)
        layer = extract_subject(source, mask, bounds)
        background = build_background(source, mask)
        if viewpoint.active:
            background, _ = rotate(background)
        canvas, _ = self.viewport_renderer.transform_background_by_viewport(
            background, target_state.framing.reference_viewport)
        placed, actual = transform_subject_by_bbox(layer, target_state.subject.bbox, self.size)
        result = composite(canvas, placed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, quality=95)
        if self.debug:
            background.save(output_path.parent / "background_without_subject.jpg", quality=95)
            canvas.save(output_path.parent / "background_canvas.jpg", quality=95)
            layer.save(output_path.parent / "subject_layer.png")
            placed.save(output_path.parent / "placed_subject.png")
        meta = meta.model_copy(update={"rendered_subject_bbox": actual,
                                       "subject_transform_applied": True})
        return str(output_path.resolve()), meta

    def _render_depth(self,source,subject,observation,target,output_path):
        if observation is None:
            raise ValueError('Depth viewpoint requires reference_depth from depth estimation node')
        depth = load_depth(observation,source.size)
        scene=float(np.median(depth[depth>0]))
        background = source
        layer = None
        mask = None
        if target.subject.mode == 'reposition':
            if subject is None:
                raise ValueError('reposition requires reference_subject detection')
            mask,bounds = load_subject_mask(source,subject)
            layer = extract_subject(source,mask,bounds)
            background = build_background(source,mask)
            depth = repair_background_depth(depth,mask)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        warped,warp_meta,valid = self._apply_depth_viewpoint(background,depth,observation,target.viewpoint,
            scene,mask,subject.bbox if subject is not None and layer is None else None,output_path.parent)
        warp_meta['background_depth_repaired'] = layer is not None
        canvas,meta = self.viewport_renderer.transform_background_by_viewport(warped,target.framing.reference_viewport)
        projected = warp_meta['projected_subject_bbox']
        natural = transform_bbox_by_viewport(projected,meta.rendered_viewport) if projected else None
        actual = clip_bbox_to_canvas(natural)
        if layer is not None:
            placed,actual = transform_subject_by_bbox(layer,target.subject.bbox,self.size)
            result = composite(canvas,placed)
        else:
            result = canvas
        result.save(output_path,quality=95)
        if self.debug:
            background.save(output_path.parent/'background_without_subject.jpg',quality=95)
            warped.save(output_path.parent/'viewpoint_warped_background.jpg',quality=95)
            Image.fromarray(valid.astype(np.uint8)*255).save(output_path.parent/'viewpoint_valid_mask.png')
            np.save(output_path.parent/'background_depth.npy',depth,allow_pickle=False)
            canvas.save(output_path.parent/'background_canvas.jpg',quality=95)
            if layer is not None:
                layer.save(output_path.parent/'subject_layer.png')
                placed.save(output_path.parent/'placed_subject.png')
        meta = meta.model_copy(update={'viewpoint_warp':warp_meta,'subject_mode':target.subject.mode,
            'source_subject_bbox':subject.bbox if subject is not None else None,
            'natural_subject_bbox':natural,'rendered_subject_bbox':actual,'subject_transform_applied':layer is not None})
        return str(output_path.resolve()),meta

    def _apply_depth_viewpoint(self,image,depth,observation,viewpoint,scene,mask,bbox,output_dir):
        if viewpoint.mode=='depth_mesh':
            return self.mesh_warper.warp(image,depth,viewpoint,observation,
                fill_color=self.viewport_renderer.fill_color,subject_mask=mask,
                debug_dir=output_dir if self.debug else None,scene_reference_depth=scene)
        k,intrinsics_source=camera_intrinsics(image.size,viewpoint,observation)
        # Metric storage stays intact. Convert only this baseline's temporary render input.
        point_depth=depth/scene if observation.metric else depth
        image,meta,valid=self.point_warper.warp(image,point_depth,viewpoint,
            fill_color=self.viewport_renderer.fill_color,subject_bbox=bbox,
            focal_length_px=float(k[0,0]) if intrinsics_source=='depth_pro' else None)
        meta.update(metric_depth=observation.metric,intrinsics_source=intrinsics_source,
            scene_reference_depth=scene,translation_metric_equivalent=[viewpoint.translation_x*scene,
                viewpoint.translation_y*scene,viewpoint.translation_z*scene] if observation.metric else None)
        return image,meta,valid

    def _save_follow(self, canvas, meta, natural, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        # No mask loading, extraction, inpaint, or subject compositing on this path.
        meta = meta.model_copy(update={"rendered_subject_bbox": clip_bbox_to_canvas(natural)})
        return str(output_path.resolve()), meta
