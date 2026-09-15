from .base import NormalizedBox, ViewportBox, Schema


class SubjectTarget(Schema):
    bbox: NormalizedBox


class FramingTarget(Schema):
    reference_viewport: ViewportBox


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
