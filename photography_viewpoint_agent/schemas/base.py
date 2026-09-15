import math
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict


def check_box(box: tuple[float, float, float, float]):
    if not all(0 <= v <= 1 for v in box):
        raise ValueError("Coordinates must be finite and within [0, 1]")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Box must satisfy xmin < xmax and ymin < ymax")
    return box


NormalizedBox = Annotated[tuple[float, float, float, float], AfterValidator(check_box)]


def check_viewport(box: tuple[float, float, float, float]):
    if not all(math.isfinite(v) for v in box):
        raise ValueError("Viewport coordinates must be finite")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Viewport must satisfy xmin < xmax and ymin < ymax")
    if not math.isfinite(box[2] - box[0]) or not math.isfinite(box[3] - box[1]):
        raise ValueError("Viewport dimensions must be finite")
    return box


ViewportBox = Annotated[tuple[float, float, float, float], AfterValidator(check_viewport)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
