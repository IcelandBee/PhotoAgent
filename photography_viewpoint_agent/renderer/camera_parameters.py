"""Low-level rendering parameters, NOT a composition/Planner contract.

Translation remains available for direct research calls to the point/mesh warpers.
The production adapter below always constructs a zero-translation rotation.
"""
from typing import Literal
from pydantic import Field, model_validator
from photography_viewpoint_agent.schemas.base import Schema
from photography_viewpoint_agent.schemas.target import ViewpointTarget

class CameraWarpParameters(Schema):
    mode: Literal['none', 'rotation', 'depth_3d', 'depth_mesh'] = 'none'
    yaw_deg: float = Field(default=0, ge=-15, le=15)
    pitch_deg: float = Field(default=0, ge=-15, le=15)
    roll_deg: float = Field(default=0, ge=-15, le=15)
    horizontal_fov_deg: float = Field(default=60, ge=10, le=150)
    border_mode: Literal['constant', 'replicate'] = 'constant'
    translation_x: float = Field(default=0, ge=-0.2, le=0.2)
    translation_y: float = Field(default=0, ge=-0.2, le=0.2)
    translation_z: float = Field(default=0, ge=-0.2, le=0.2)

    @model_validator(mode='after')
    def no_ignored_angles(self):
        if self.mode not in ('depth_3d','depth_mesh') and any((self.translation_x,self.translation_y,self.translation_z)):
            raise ValueError('Translation requires a depth viewpoint mode')
        if self.mode == 'depth_mesh' and self.border_mode != 'constant':
            raise ValueError('depth_mesh requires constant fill; unknown geometry must remain invalid')
        if self.mode == 'none' and any((self.yaw_deg, self.pitch_deg, self.roll_deg)):
            raise ValueError('Nonzero angles require viewpoint.mode=rotation')
        return self

    @property
    def active(self):
        return self.mode != 'none' and any((self.yaw_deg, self.pitch_deg, self.roll_deg,
                                           self.translation_x,self.translation_y,self.translation_z))


def rotation_parameters(viewpoint: ViewpointTarget, backend="homography", *,
                        horizontal_fov_deg=60.0, border_mode="constant") -> CameraWarpParameters:
    """Bridge orientation intent to a selected renderer; no translation input."""
    intent = ViewpointTarget.model_validate(viewpoint.model_dump())
    if backend not in ("homography", "depth_3d", "depth_mesh"):
        raise ValueError("Unknown viewpoint backend")
    return CameraWarpParameters(
        mode="rotation" if backend == "homography" else backend,
        yaw_deg=intent.yaw_deg, pitch_deg=intent.pitch_deg, roll_deg=intent.roll_deg,
        horizontal_fov_deg=horizontal_fov_deg, border_mode=border_mode,
        translation_x=0, translation_y=0, translation_z=0,
    )
