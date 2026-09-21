"""Low-level point projection pose; backend selection is no longer supported."""
from typing import Literal
from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema


class CameraWarpParameters(Schema):
    mode: Literal['depth_3d'] = 'depth_3d'
    yaw_deg: float = Field(default=0, ge=-15, le=15)
    pitch_deg: float = Field(default=0, ge=-15, le=15)
    roll_deg: float = Field(default=0, ge=-15, le=15)
    horizontal_fov_deg: float = Field(default=60, ge=10, le=150)
    border_mode: Literal['constant'] = 'constant'
    translation_x: float = Field(default=0, ge=-0.2, le=0.2)
    translation_y: float = Field(default=0, ge=-0.2, le=0.2)
    translation_z: float = Field(default=0, ge=-0.2, le=0.2)

    @property
    def active(self):
        return any((self.yaw_deg, self.pitch_deg, self.roll_deg,
                    self.translation_x, self.translation_y, self.translation_z))
