"""Explicit camera state used by production and batch rendering.

Source camera is the reference pose. Translation is CAMERA displacement in
source axes (+x right, +y up, +z forward), in scene-median-depth units.
It is NOT the world-to-camera extrinsic t. Camera coordinates have y down:
X_target = R @ (X_source - C), C = (x, -y, z), t = -R @ C.
"""
import numpy as np
from pydantic import Field
from photography_viewpoint_agent.schemas.base import Schema
from photography_viewpoint_agent.renderer.camera_parameters import CameraWarpParameters
from .geometry import rotation_matrix
from .intrinsics import CameraIntrinsics


class CameraRotation(Schema):
    yaw_deg: float = Field(default=0, ge=-15, le=15)
    pitch_deg: float = Field(default=0, ge=-15, le=15)
    roll_deg: float = Field(default=0, ge=-15, le=15)


class CameraTranslation(Schema):
    x: float = Field(default=0, ge=-0.2, le=0.2)
    y: float = Field(default=0, ge=-0.2, le=0.2)
    z: float = Field(default=0, ge=-0.2, le=0.2)

    @property
    def norm(self):
        return float(np.linalg.norm([self.x, self.y, self.z]))


class CameraState(Schema):
    source_intrinsics: CameraIntrinsics
    target_intrinsics: CameraIntrinsics
    rotation: CameraRotation = Field(default_factory=CameraRotation)
    translation: CameraTranslation = Field(default_factory=CameraTranslation)

    def warp_parameters(self):
        # Projection always uses the explicit source and target K.
        return CameraWarpParameters(mode='depth_3d', **self.rotation.model_dump(),
                                    translation_x=self.translation.x,
                                    translation_y=self.translation.y,
                                    translation_z=self.translation.z, border_mode='constant')

    def rotation_matrix(self):
        return rotation_matrix(self.warp_parameters())

    def camera_center(self):
        return np.array([self.translation.x, -self.translation.y, self.translation.z])
