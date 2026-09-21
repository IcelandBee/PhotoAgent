"""Convert target intent to one camera projection, with no post-render resize."""
from .camera_state import CameraState
from .intrinsics import CameraIntrinsics
from photography_viewpoint_agent.tools.geometry import fit_viewport


def camera_from_target(target, source_size, target_size, horizontal_fov_deg=60):
    width, height = source_size
    tw, th = target_size
    source = CameraIntrinsics.from_horizontal_fov(width, height, horizontal_fov_deg)
    viewport, scale = fit_viewport(target.framing.reference_viewport, source_size, target_size)
    left, top, _, _ = viewport
    focal = target.framing.focal_scale
    # Viewport maps source pixel coordinates onto target pixels. Additional focal
    # scaling zooms about the target canvas center, entirely inside K_target.
    destination = CameraIntrinsics(fx=source.fx * scale * focal, fy=source.fy * scale * focal,
        cx=tw / 2 + ((source.cx - left * width) * scale - tw / 2) * focal,
        cy=th / 2 + ((source.cy - top * height) * scale - th / 2) * focal,
        image_width=tw, image_height=th)
    view = target.viewpoint
    return CameraState(source_intrinsics=source, target_intrinsics=destination,
        rotation=view.rotation, translation=dict(zip(('x', 'y', 'z'), view.translation))), viewport
