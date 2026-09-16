import math
from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.tools.geometry import fit_viewport


class ViewportRenderer:
    """Background-only geometry tool; the workflow uses TargetSketchRenderer."""

    def __init__(self, width: int, height: int, fill_color=(128, 128, 128)):
        self.size, self.fill_color = (width, height), fill_color

    def transform_background_by_viewport(self, source: Image.Image, viewport):
        fitted, scale = fit_viewport(viewport, source.size, self.size)
        width, height = source.size
        left, top = fitted[0]*width, fitted[1]*height
        # ONE scale for both axes, including subpixel boundaries.
        canvas = source.convert("RGB").transform(self.size, Image.Transform.AFFINE,
            (1/scale, 0, left, 0, 1/scale, top), Image.Resampling.BICUBIC,
            fillcolor=self.fill_color)
        cw, ch = self.size
        def edge(value, limit):
            return max(0, min(limit, math.ceil(value - 0.5)))
        x0, y0 = edge(-left*scale, cw), edge(-top*scale, ch)
        x1, y1 = edge((width-left)*scale, cw), edge((height-top)*scale, ch)
        meta = RenderMeta(requested_viewport=viewport, rendered_viewport=fitted, reference_size=source.size,
                          padding=(x0, y0, cw-x1, ch-y1))
        return canvas, meta

    def render(self, reference_frame: FrameInfo, target_state: TargetState,
               output_path: Path) -> tuple[str, RenderMeta]:
        with Image.open(reference_frame.path) as source:
            canvas, meta = self.transform_background_by_viewport(source, target_state.framing.reference_viewport)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        return str(output_path.resolve()), meta
