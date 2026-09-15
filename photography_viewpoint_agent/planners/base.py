from typing import Protocol
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState


class CompositionPlanner(Protocol):
    def plan(self, current_frame: FrameInfo, reference_frame: FrameInfo,
             manual_target_state: TargetState | None = None) -> TargetState: ...
