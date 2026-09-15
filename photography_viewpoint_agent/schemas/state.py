from typing import Annotated, TypedDict
from .target import TargetState
from .video import VideoMeta, FrameInfo
from .rendering import RenderMeta
from .validation import SketchValidation


def merge_errors(left: str | None, right: str | None) -> str | None:
    """Preserve errors from both parallel branches, without shared-key conflicts."""
    return "; ".join(sorted(set(filter(None, (left, right))))) or None


class AgentState(TypedDict, total=False):
    video_path: str
    manual_target_state: TargetState
    video_meta: VideoMeta
    frames: list[FrameInfo]
    current_frame: FrameInfo
    reference_frame: FrameInfo
    target_state: TargetState
    target_sketch_path: str
    render_meta: RenderMeta
    validation: SketchValidation
    error: Annotated[str | None, merge_errors]
