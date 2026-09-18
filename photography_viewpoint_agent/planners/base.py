from typing import Protocol
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState


class CompositionPlanner(Protocol):
    """Produce editing intent, not renderer or physical translation parameters.

    Whole-view shift/crop/zoom/expansion -> framing.reference_viewport.
    Person movement/scale relative to background -> subject.bbox (final canvas).
    Explicit turn/tilt/roll of the camera -> viewpoint yaw/pitch/roll only.
    Do not infer camera displacement from an image-space framing request.
    Execution order is orientation -> framing -> final subject layout.
    """
    def plan(self, current_frame: FrameInfo, reference_frame: FrameInfo,
             manual_target_state: TargetState | None = None) -> TargetState: ...
