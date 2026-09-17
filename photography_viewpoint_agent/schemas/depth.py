from typing import Literal
from pydantic import Field, model_validator
from .base import Schema


class DepthObservation(Schema):
    depth_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    representation: Literal['z_depth'] = 'z_depth'
    normalization: Literal['median_one','none'] = 'median_one'
    metric: bool = False
    unit: Literal['relative','meter'] = 'relative'
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def consistent_scale(self):
        if self.metric != (self.normalization == 'none' and self.unit == 'meter'):
            raise ValueError('Metric depth requires normalization=none and unit=meter')
        if not self.metric and (self.unit != 'relative' or self.normalization != 'median_one'):
            raise ValueError('Relative depth requires median_one normalization')
        return self
