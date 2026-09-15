from .base import NormalizedBox, ViewportBox, Schema


class SubjectTarget(Schema):
    """Canvas envelope: uniformly contain the subject and align bottom-center."""
    bbox: NormalizedBox


class FramingTarget(Schema):
    reference_viewport: ViewportBox


class TargetState(Schema):
    subject: SubjectTarget
    framing: FramingTarget
