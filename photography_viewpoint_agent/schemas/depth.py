from typing import Literal
from pydantic import Field
from .base import Schema


class DepthObservation(Schema):
    depth_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    representation: Literal['z_depth'] = 'z_depth'
    normalization: Literal['median_one'] = 'median_one'
    metadata: dict = Field(default_factory=dict)
