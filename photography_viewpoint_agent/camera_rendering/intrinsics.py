"""Explicit pinhole intrinsics shared by production and batch rendering."""
import math
import numpy as np
from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema


class CameraIntrinsics(Schema):
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)

    @property
    def size(self):
        return self.image_width, self.image_height

    def matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    @classmethod
    def from_horizontal_fov(cls, width, height, fov_deg=60.0):
        if not math.isfinite(fov_deg) or not 0 < fov_deg < 180:
            raise ValueError('Horizontal FoV must be finite and in (0, 180) degrees')
        focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
        return cls(fx=focal, fy=focal, cx=width / 2, cy=height / 2,
                   image_width=width, image_height=height)

    @property
    def centered_square_pixels(self):
        return (self.fx == self.fy and self.cx == self.image_width / 2
                and self.cy == self.image_height / 2)
