from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState


class ManualCompositionPlanner:
    def plan(self, current_frame: FrameInfo, reference_frame: FrameInfo,
             manual_target_state: TargetState | None = None) -> TargetState:
        if manual_target_state is None:
            raise ValueError("ManualCompositionPlanner requires manual_target_state")
        return manual_target_state
