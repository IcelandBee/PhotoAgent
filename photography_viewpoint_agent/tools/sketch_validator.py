"""Validate camera intent and artifact integrity, not novel-view visual accuracy."""
import numpy as np
from PIL import Image
from photography_viewpoint_agent.camera_rendering.intent import camera_from_target
from photography_viewpoint_agent.schemas.validation import SketchValidation


def validate_sketch(target, meta, path, config):
    messages, passed = [], True
    def require(condition, message):
        nonlocal passed
        if not condition:
            passed = False
            messages.append(message)
    framing = None
    if meta.reference_size:
        camera, viewport = camera_from_target(target, meta.reference_size,
            (config.target_width, config.target_height), config.viewpoint_horizontal_fov_deg)
        framing = sum(abs(a - b) for a, b in zip(viewport, meta.rendered_viewport)) / 4
        require(framing <= config.framing_threshold, 'Target intrinsics viewport differs from intent.')
        warp = meta.viewpoint_warp or {}
        require(warp.get('camera_state') == camera.model_dump(mode='json'), 'Camera pose/intrinsics differ from intent.')
        require(warp.get('parameters') == camera.warp_parameters().model_dump(), 'Camera parameters differ from intent.')
        require(warp.get('renderer') == 'point_cloud' and warp.get('depth_used') is True
                and warp.get('identity_copy_fast_path') is False, 'Point-cloud execution metadata missing.')
    else:
        require(False, 'Reference dimensions missing.')
    require(meta.viewpoint_backend == 'depth_3d', 'Unexpected renderer backend.')
    require(meta.camera_translation == target.viewpoint.translation, 'Camera translation differs from intent.')
    require(meta.viewpoint_rotation == target.viewpoint.rotation, 'Camera rotation differs from intent.')
    require(meta.viewpoint_applied == bool(target.viewpoint.active), 'Camera motion flag differs from intent.')
    require(meta.requested_viewport == target.framing.reference_viewport and
            meta.focal_scale == target.framing.focal_scale, 'Framing request differs from intent.')
    require(not meta.subject_transform_applied and meta.subject_mode == 'follow_reference',
            'Independent subject edit is forbidden.')
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
            require(image.size == (config.target_width, config.target_height), 'Target dimensions mismatch.')
        with Image.open(meta.valid_mask_path) as mask_image:
            mask = np.asarray(mask_image.convert('L'))
        require(mask.shape == pixels.shape[:2], 'Valid mask dimensions mismatch.')
        require(bool(np.isin(mask, (0, 255)).all()), 'Valid mask must be binary.')
        coverage = float((mask > 0).mean())
        require(abs(coverage - meta.valid_pixel_ratio) < 1e-8 and
                abs(1 - coverage - meta.hole_pixel_ratio) < 1e-8, 'Coverage metadata mismatch.')
        require(coverage > 0, 'Camera view contains no observed geometry.')
        if mask.shape == pixels.shape[:2]:
            require(bool((pixels[mask == 0] == config.fill_color).all()), 'Unknown pixels must use constant fill.')
    except (OSError, ValueError, TypeError) as exc:
        require(False, f'Cannot read render artifacts: {exc}')
    messages.append('Coverage validates projection plumbing, not photorealism; disocclusions remain blank.')
    return SketchValidation(passed=passed, framing_error=framing, messages=messages)
