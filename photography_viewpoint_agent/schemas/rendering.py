from typing import Annotated
from pydantic import Field
from .base import NormalizedBox, ViewportBox, Schema
from .target import SubjectMode

PixelCount = Annotated[int, Field(ge=0, strict=True)]


class RenderMeta(Schema):
    rendered_viewport: ViewportBox
    requested_viewport: ViewportBox | None = None
    subject_mode: SubjectMode = "follow_reference"
    subject_transform_applied: bool = False
    # Unclipped projection; may be partly or entirely outside the canvas.
    natural_subject_bbox: ViewportBox | None = None
    rendered_subject_bbox: NormalizedBox | None = None
    source_subject_bbox: NormalizedBox | None = None
    reference_size: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] | None = None
    # Pixel margins (left, top, right, bottom), before JPEG compression.
    padding: tuple[PixelCount, PixelCount, PixelCount, PixelCount] = (0, 0, 0, 0)
