from pathlib import Path
from typing import Protocol
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta


class SketchRenderer(Protocol):
    def render(self, reference_frame: FrameInfo, target_state: TargetState,
               output_path: Path) -> tuple[str, RenderMeta]: ...
