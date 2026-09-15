from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.tools.subject_utils import (
    load_subject_mask, extract_subject, build_background, transform_subject_by_bbox, composite,
)


class TargetSketchRenderer:
    def __init__(self, width=1280, height=720, fill_color=(128, 128, 128), debug=False):
        self.viewport_renderer = ViewportRenderer(width, height, fill_color)
        self.size, self.debug = (width, height), debug

    def render(self, reference_frame: FrameInfo, reference_subject: SubjectObservation,
               target_state: TargetState, output_path: Path):
        with Image.open(reference_frame.path) as image:
            source = image.convert("RGB")
        mask, bounds = load_subject_mask(source, reference_subject)
        layer = extract_subject(source, mask, bounds)
        background = build_background(source, mask)
        canvas, meta = self.viewport_renderer.transform_background_by_viewport(
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
                                       "source_subject_bbox": reference_subject.bbox})
        return str(output_path.resolve()), meta
