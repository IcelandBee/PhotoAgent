from typing import Literal
from pydantic import Field, model_validator
from .base import NormalizedBox, ViewportBox, Schema

SubjectMode = Literal["follow_reference", "reposition"]


class SubjectTarget(Schema):
    """Explicit edit intent; reposition bbox is a bottom-centered contain envelope."""
    mode: SubjectMode
    bbox: NormalizedBox | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == "follow_reference" and self.bbox is not None:
            raise ValueError("follow_reference requires bbox=None")
        if self.mode == "reposition" and self.bbox is None:
            raise ValueError("reposition requires a target bbox")
        return self


class FramingTarget(Schema):
    reference_viewport: ViewportBox


class ViewpointTarget(Schema):
    mode: Literal['none', 'rotation', 'depth_3d'] = 'none'
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
        if self.mode != 'depth_3d' and any((self.translation_x,self.translation_y,self.translation_z)):
            raise ValueError('Translation requires viewpoint.mode=depth_3d')
        if self.mode == 'none' and any((self.yaw_deg, self.pitch_deg, self.roll_deg)):
            raise ValueError('Nonzero angles require viewpoint.mode=rotation')
        return self

    @property
    def active(self):
        return self.mode != 'none' and any((self.yaw_deg, self.pitch_deg, self.roll_deg,
                                           self.translation_x,self.translation_y,self.translation_z))


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
    viewpoint: ViewpointTarget = Field(default_factory=ViewpointTarget)
