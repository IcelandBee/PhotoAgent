from .base import NormalizedBox, Schema


class SubjectTarget(Schema):
    bbox: NormalizedBox


class FramingTarget(Schema):
    reference_viewport: NormalizedBox


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
