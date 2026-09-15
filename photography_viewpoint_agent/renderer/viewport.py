import math
from pathlib import Path
from PIL import Image
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta


class ViewportRenderer:
    def __init__(self, width: int, height: int):
        self.size = (width, height)

    def render(self, reference_frame: FrameInfo, target_state: TargetState,
               output_path: Path) -> tuple[str, RenderMeta]:
        viewport = target_state.framing.reference_viewport
        with Image.open(reference_frame.path) as source:
            width, height = source.size
            # Outward rounding preserves even subpixel-sized valid viewports.
            bounds = (math.floor(viewport[0] * width), math.floor(viewport[1] * height),
                      math.ceil(viewport[2] * width), math.ceil(viewport[3] * height))
            canvas = source.convert("RGB").crop(bounds).resize(self.size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        actual = (bounds[0] / width, bounds[1] / height,
                  bounds[2] / width, bounds[3] / height)
        return str(output_path.resolve()), RenderMeta(rendered_viewport=actual)
