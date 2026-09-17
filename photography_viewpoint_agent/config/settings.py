from typing import Annotated
from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema

ColorChannel = Annotated[int, Field(ge=0, le=255, strict=True)]


class AgentConfig(Schema):
    frame_sample_interval: int = Field(default=30, gt=0, strict=True)
    random_seed: int = 42
    target_width: int = Field(default=1280, gt=0, strict=True)
    target_height: int = Field(default=720, gt=0, strict=True)
    fill_color: tuple[ColorChannel, ColorChannel, ColorChannel] = (128, 128, 128)
    yolo_model: str = "yolo11n-seg.pt"
    person_confidence: float = Field(default=0.4, gt=0, le=1)
    device: str = "cpu"
    depth_model: str = 'depth-anything/Depth-Anything-V2-Small-hf'
    depth_path: str | None = None
    depth_device: str = 'cpu'
    depth_debug: bool = False
    splat_radius: int = Field(default=1, ge=1, le=3, strict=True)
    debug: bool = False
    subject_noop_position_threshold: float = Field(default=0.001, ge=0)
    subject_noop_scale_threshold: float = Field(default=0.001, ge=0)
    subject_position_threshold: float = Field(default=0.05, ge=0)
    subject_scale_threshold: float = Field(default=0.05, ge=0)
    framing_threshold: float = Field(default=0.03, ge=0)
    work_dir: str = "./workdir"
