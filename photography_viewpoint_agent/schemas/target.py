"""Scene-fixed camera intent shared by planning, rendering and Guidance."""
from typing import Literal
from pydantic import Field, model_validator
from .base import ViewportBox, Schema

SubjectMode = Literal['follow_reference']


class SubjectTarget(Schema):
    mode: SubjectMode = 'follow_reference'
    bbox: None = None


class FramingTarget(Schema):
    """Viewport and focal scale become target intrinsics before point projection."""
    reference_viewport: ViewportBox = (0, 0, 1, 1)
    focal_scale: float = Field(default=1, ge=0.5, le=2)


class ViewpointTarget(Schema):
    """Camera displacement: +x right, +y up, +z forward in source axes.
    Units are fractions of scene median depth, not meters.
    Positive yaw turns right, pitch looks up, roll turns clockwise.
    Legacy none/rotation names keep their original motion constraints.
    """
    mode: Literal['none', 'rotation', 'camera'] = 'none'
    yaw_deg: float = Field(default=0, ge=-15, le=15)
    pitch_deg: float = Field(default=0, ge=-15, le=15)
    roll_deg: float = Field(default=0, ge=-15, le=15)
    translation_x: float = Field(default=0, ge=-0.2, le=0.2)
    translation_y: float = Field(default=0, ge=-0.2, le=0.2)
    translation_z: float = Field(default=0, ge=-0.2, le=0.2)

    @model_validator(mode='after')
    def validate_motion(self):
        if self.mode != 'camera' and any(self.translation):
            raise ValueError('Camera translation requires viewpoint.mode=camera')
        if self.mode == 'none' and any(self.rotation.values()):
            raise ValueError('Nonzero angles require viewpoint.mode=rotation or camera')
        return self

    @property
    def translation(self):
        return self.translation_x, self.translation_y, self.translation_z

    @property
    def rotation(self):
        return {name: getattr(self, name) for name in ('yaw_deg', 'pitch_deg', 'roll_deg')}

    @property
    def active(self):
        return any((*self.translation, *self.rotation.values()))


class TargetState(Schema):
    subject: SubjectTarget = Field(default_factory=SubjectTarget)
    framing: FramingTarget = Field(default_factory=FramingTarget)
    viewpoint: ViewpointTarget = Field(default_factory=ViewpointTarget)
