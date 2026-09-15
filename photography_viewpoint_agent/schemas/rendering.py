from .base import NormalizedBox, Schema


class RenderMeta(Schema):
    rendered_viewport: NormalizedBox
    rendered_subject_bbox: NormalizedBox | None = None
