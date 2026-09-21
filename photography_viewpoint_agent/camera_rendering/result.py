"""NewCameraView is a camera projection, not a final Target Sketch."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from PIL import Image
from .camera_state import CameraState


@dataclass
class CameraRenderResult:
    image: Image.Image
    valid_mask: np.ndarray
    renderer_type: str
    camera_state: CameraState
    metadata: dict[str, Any]
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def hole_mask(self):
        return ~self.valid_mask

    def record(self):
        """JSON-ready description; arrays in debug_info are saved separately."""
        return {'result_type': 'NewCameraView', 'renderer_type': self.renderer_type,
                'camera_state': self.camera_state.model_dump(mode='json'), 'metadata': self.metadata}


NewCameraView = CameraRenderResult
