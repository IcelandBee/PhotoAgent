from pydantic import Field
from .base import Schema


class SketchValidation(Schema):
    passed: bool
    subject_position_error: float | None = Field(default=None, ge=0)
    subject_scale_error: float | None = Field(default=None, ge=0)
    framing_error: float | None = Field(default=None, ge=0)
    messages: list[str] = Field(default_factory=list)
