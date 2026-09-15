from pathlib import Path
from typing import Protocol
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.subject import SubjectObservation


class SubjectDetector(Protocol):
    def detect(self, frame: FrameInfo, mask_path: Path) -> SubjectObservation: ...
