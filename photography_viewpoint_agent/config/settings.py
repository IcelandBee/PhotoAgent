from typing import Annotated, Literal
from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema

ColorChannel = Annotated[int, Field(ge=0, le=255, strict=True)]


class AgentConfig(Schema):
    frame_sample_interval: int = Field(default=30, gt=0, strict=True)
    random_seed: int = 42
    target_width: int = Field(default=1280, gt=0, strict=True)
    target_height: int = Field(default=720, gt=0, strict=True)
    fill_color: tuple[ColorChannel, ColorChannel, ColorChannel] = (128, 128, 128)
    viewpoint_backend: Literal['depth_3d'] = 'depth_3d'
    viewpoint_horizontal_fov_deg: float = Field(default=60, ge=10, le=150)
    viewpoint_border_mode: Literal['constant'] = 'constant'
    depth_model: str | None = None
    depth_backend: Literal['auto', 'depth_anything', 'depth_pro', 'precomputed'] = 'auto'
    depth_path: str | None = None
    depth_metadata_path: str | None = None
    depth_device: str = 'cpu'
    depth_debug: bool = False
    splat_radius: int = Field(default=1, ge=0, le=3, strict=True)
    debug: bool = False
    framing_threshold: float = Field(default=0.03, ge=0)
    work_dir: str = './workdir'

    def needs_reference_depth(self, viewpoint=None) -> bool:
        return True
