import numpy as np
from PIL import Image
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.tools.geometry import transform_bbox_by_viewport, expected_subject_bbox, subject_is_noop
from photography_viewpoint_agent.tools.subject_utils import (
    load_subject_mask, extract_subject, build_background, transform_subject_by_bbox, composite,
)
from .images import bbox_mask


class PreparedRenderer:
    """Cache constant layers; final export always uses the real TargetSketchRenderer."""
    def __init__(self,source,observation,size,reference_ignore=None):
        self.source,self.observation,self.size = source,observation,size
        self.reference_ignore = reference_ignore
        self.viewport = ViewportRenderer(*size)
        self.background,self.layer = None,None
        if observation is not None:
            mask,bounds = load_subject_mask(source,observation)
            self.layer = extract_subject(source,mask,bounds)
            self.background = build_background(source,mask)

    def render(self,target):
        viewport = target.framing.reference_viewport
        # Decode already matches aspect ratio, except preview dimensions can round by one pixel.
        canvas,meta = self.viewport.transform_background_by_viewport(self.source,viewport)
        exclusion = self.transformed_ignore(target,self.size)
        actual = None
        if self.observation is not None:
            natural = transform_bbox_by_viewport(self.observation.bbox,meta.rendered_viewport)
            exclusion |= bbox_mask(natural,self.size)  # Also exclude the original inpaint hole.
            expected = expected_subject_bbox(self.observation.bbox,self.source.size,target.subject.bbox,self.size)
            if subject_is_noop(natural,expected,0.001,0.001):
                actual = natural
            else:
                canvas,_ = self.viewport.transform_background_by_viewport(self.background,viewport)
                placed,actual = transform_subject_by_bbox(self.layer,target.subject.bbox,self.size)
                canvas = composite(canvas,placed)
            exclusion |= bbox_mask(actual,self.size)
        return canvas,actual,exclusion

    def transformed_ignore(self,target,size):
        if self.reference_ignore is None:
            return np.zeros((size[1],size[0]),dtype=np.uint8)
        transformed,_ = ViewportRenderer(*size,fill_color=(255,255,255)).transform_background_by_viewport(
            self.reference_ignore.convert('RGB'),target.framing.reference_viewport)
        return (np.asarray(transformed).max(axis=2)>0).astype(np.uint8)*255
