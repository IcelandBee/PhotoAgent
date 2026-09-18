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
    """Whole-view 2D framing on the same-size, optionally rotated reference canvas."""
    reference_viewport: ViewportBox


class ViewpointTarget(Schema):
    """Camera orientation only; optical center stays fixed.

    Whole-view 2D shift/zoom belongs to framing.reference_viewport.
    Independent person position/scale belongs to subject.bbox.
    Renderer choice, intrinsics and border handling belong to AgentConfig.
    """
    mode: Literal['none', 'rotation'] = 'none'
    yaw_deg: float = Field(default=0, ge=-15, le=15)
    pitch_deg: float = Field(default=0, ge=-15, le=15)
    roll_deg: float = Field(default=0, ge=-15, le=15)

    @model_validator(mode='after')
    def no_ignored_angles(self):
        if self.mode == 'none' and any((self.yaw_deg, self.pitch_deg, self.roll_deg)):
            raise ValueError('Nonzero angles require viewpoint.mode=rotation')
        return self

    @property
    def active(self):
        return any((self.yaw_deg, self.pitch_deg, self.roll_deg))


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
    viewpoint: ViewpointTarget = Field(default_factory=ViewpointTarget)
