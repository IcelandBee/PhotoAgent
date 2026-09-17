from pathlib import Path
from typing import Protocol
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.depth import DepthObservation


class DepthEstimator(Protocol):
    def estimate(self, frame: FrameInfo, output_path: Path) -> DepthObservation: ...
