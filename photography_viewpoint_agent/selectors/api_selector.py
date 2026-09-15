from typing import Callable, Sequence
from photography_viewpoint_agent.schemas.video import FrameInfo


class APIReferenceSelector:
    """Inject an API adapter returning a candidate frame_id; transport stays outside graph."""

    def __init__(self, select_frame_id: Callable[[Sequence[FrameInfo]], str]):
        self.select_frame_id = select_frame_id

    def select(self, frames: Sequence[FrameInfo]) -> FrameInfo:
        frame_id = self.select_frame_id(frames)
        for frame in frames:
            if frame.frame_id == frame_id:
                return frame
        raise ValueError(f"Selector API returned unknown frame_id: {frame_id}")
