from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.tools.subject_utils import (
    load_subject_mask, extract_subject, build_background, transform_subject_by_bbox, composite,
)
from photography_viewpoint_agent.tools.geometry import (
    transform_bbox_by_viewport, clip_bbox_to_canvas, expected_subject_bbox, subject_is_noop,
)


class TargetSketchRenderer:
    def __init__(self, width=1280, height=720, fill_color=(128, 128, 128), debug=False,
                 subject_noop_position_threshold=0.001, subject_noop_scale_threshold=0.001):
        self.viewport_renderer = ViewportRenderer(width, height, fill_color)
        self.size, self.debug = (width, height), debug
        self.noop_position = subject_noop_position_threshold
        self.noop_scale = subject_noop_scale_threshold

    def render(self, reference_frame: FrameInfo, reference_subject: SubjectObservation | None,
               target_state: TargetState, output_path: Path):
        with Image.open(reference_frame.path) as image:
            source = image.convert("RGB")
        canvas, meta = self.viewport_renderer.transform_background_by_viewport(
            source, target_state.framing.reference_viewport)
        natural = (transform_bbox_by_viewport(reference_subject.bbox, meta.rendered_viewport)
                   if reference_subject is not None else None)
        meta = meta.model_copy(update={
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
        if subject_is_noop(natural, expected, self.noop_position, self.noop_scale):
            return self._save_follow(canvas, meta, natural, output_path)
        mask, bounds = load_subject_mask(source, reference_subject)
        layer = extract_subject(source, mask, bounds)
        background = build_background(source, mask)
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

    def _save_follow(self, canvas, meta, natural, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        # No mask loading, extraction, inpaint, or subject compositing on this path.
        meta = meta.model_copy(update={"rendered_subject_bbox": clip_bbox_to_canvas(natural)})
        return str(output_path.resolve()), meta
