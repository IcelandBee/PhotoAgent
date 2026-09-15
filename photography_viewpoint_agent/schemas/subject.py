from pydantic import Field
from .base import NormalizedBox, Schema


class SubjectObservation(Schema):
    """Main person's tight mask bounds in reference coordinates; mask stays on disk."""
    bbox: NormalizedBox
    mask_path: str
    confidence: float = Field(ge=0, le=1)
