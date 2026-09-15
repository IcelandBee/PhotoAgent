from pydantic import Field
from .base import Schema


class VideoMeta(Schema):
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FrameInfo(Schema):
    frame_id: str
    path: str
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
