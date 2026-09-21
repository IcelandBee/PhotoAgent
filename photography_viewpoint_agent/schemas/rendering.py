from typing import Annotated, Literal
from pydantic import Field
from .base import ViewportBox, Schema


class RenderMeta(Schema):
    viewpoint_applied: bool = False
    viewpoint_backend: Literal['depth_3d'] = 'depth_3d'
    viewpoint_rotation: dict[str, float] = Field(default_factory=lambda: {
        'yaw_deg': 0, 'pitch_deg': 0, 'roll_deg': 0})
    camera_translation: tuple[float, float, float] = (0, 0, 0)
    viewpoint_warp: dict | None = None
    rendered_viewport: ViewportBox
    requested_viewport: ViewportBox | None = None
    focal_scale: float = 1
    subject_mode: Literal['follow_reference'] = 'follow_reference'
    subject_transform_applied: Literal[False] = False
    reference_size: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] | None = None
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    valid_pixel_ratio: float = Field(default=0, ge=0, le=1)
    hole_pixel_ratio: float = Field(default=1, ge=0, le=1)
    valid_mask_path: str | None = None
