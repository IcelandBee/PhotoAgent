from typing import Literal
from pydantic import model_validator
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


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
