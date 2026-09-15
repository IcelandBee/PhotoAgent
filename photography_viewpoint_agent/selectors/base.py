from typing import Protocol, Sequence
from photography_viewpoint_agent.schemas.video import FrameInfo


class ReferenceSelector(Protocol):
    def select(self, frames: Sequence[FrameInfo]) -> FrameInfo: ...
