from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema


class AgentConfig(Schema):
    frame_sample_interval: int = Field(default=30, gt=0, strict=True)
    random_seed: int = 42
    target_width: int = Field(default=1280, gt=0, strict=True)
    target_height: int = Field(default=720, gt=0, strict=True)
    subject_position_threshold: float = Field(default=0.05, ge=0)
    subject_scale_threshold: float = Field(default=0.05, ge=0)
    framing_threshold: float = Field(default=0.03, ge=0)
    work_dir: str = "./workdir"
