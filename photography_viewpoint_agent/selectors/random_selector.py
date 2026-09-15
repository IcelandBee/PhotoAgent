import random
from typing import Sequence
from photography_viewpoint_agent.schemas.video import FrameInfo


class RandomReferenceSelector:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, frames: Sequence[FrameInfo]) -> FrameInfo:
        if not frames:
            raise ValueError("No reference candidates")
        return random.Random(self.seed).choice(list(frames))
