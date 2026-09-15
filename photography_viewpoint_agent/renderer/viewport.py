import math
from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta


class ViewportRenderer:
    def __init__(self, width: int, height: int, fill_color=(128, 128, 128)):
        self.size = (width, height)
        self.fill_color = fill_color

    def render(self, reference_frame: FrameInfo, target_state: TargetState,
               output_path: Path) -> tuple[str, RenderMeta]:
        viewport = target_state.framing.reference_viewport
        with Image.open(reference_frame.path) as source:
            width, height = source.size
            # Outward rounding preserves even subpixel-sized valid viewports.
            bounds = (math.floor(viewport[0] * width), math.floor(viewport[1] * height),
                      math.ceil(viewport[2] * width), math.ceil(viewport[3] * height))
            left, top, right, bottom = bounds
            span_x, span_y = right - left, bottom - top
            canvas = Image.new("RGB", self.size, self.fill_color)
            target_width, target_height = self.size

            # One mapping for every viewport: target_x = (source_x - left) / span_x.
            # Clip in source space, so no oversized expanded intermediate is allocated.
            visible = (max(0, min(width, left)), max(0, min(height, top)),
                       max(0, min(width, right)), max(0, min(height, bottom)))

            def target_x(x):
                return max(0, min(target_width, round((x - left) / span_x * target_width)))

            def target_y(y):
                return max(0, min(target_height, round((y - top) / span_y * target_height)))

            dest = (target_x(visible[0]), target_y(visible[1]),
                    target_x(visible[2]), target_y(visible[3]))
            paste_width, paste_height = dest[2] - dest[0], dest[3] - dest[1]
            if visible[2] > visible[0] and visible[3] > visible[1] and paste_width > 0 and paste_height > 0:
                patch = source.convert("RGB").crop(visible).resize(
                    (paste_width, paste_height), Image.Resampling.LANCZOS)
                canvas.paste(patch, dest[:2])
            padding = (dest[0], dest[1], target_width - dest[2], target_height - dest[3])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        actual = (bounds[0] / width, bounds[1] / height,
                  bounds[2] / width, bounds[3] / height)
        return str(output_path.resolve()), RenderMeta(rendered_viewport=actual, padding=padding)
