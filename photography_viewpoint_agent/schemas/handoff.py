"""Stable target-generator output, independent of either graph's state."""
from .base import Schema
from .video import FrameInfo
from .target import TargetState
from .rendering import RenderMeta


class TargetPackage(Schema):
    video_path: str
    reference_frame: FrameInfo
    current_frame: FrameInfo
    target_state: TargetState
    target_sketch_path: str
    render_meta: RenderMeta
