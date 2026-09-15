import math
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.schemas.validation import SketchValidation
from photography_viewpoint_agent.tools.geometry import fit_viewport, contain_subject


def validate_sketch(target: TargetState, meta: RenderMeta, path: str,
                    config: AgentConfig) -> SketchValidation:
    messages = []
    expected_viewport = target.framing.reference_viewport
    if meta.reference_size is not None:
        expected_viewport, _ = fit_viewport(expected_viewport, meta.reference_size,
                                             (config.target_width, config.target_height))
        if any(abs(a-b) > 1e-8 for a,b in zip(expected_viewport, target.framing.reference_viewport)):
            messages.append("Viewport was aspect-fitted around its center; framing error uses the fitted viewport.")
    framing = sum(abs(a - b) for a, b in zip(expected_viewport, meta.rendered_viewport)) / 4
    passed = framing <= config.framing_threshold
    if not passed:
        messages.append("Framing error exceeds threshold.")
    position = scale = None
    if meta.rendered_subject_bbox is None:
        passed = False
        messages.append("Subject rendering is missing.")
    else:
        wanted, actual = target.subject.bbox, meta.rendered_subject_bbox
        position = math.hypot((wanted[0] + wanted[2] - actual[0] - actual[2]) / 2,
                              wanted[3] - actual[3])
        expected_height = wanted[3] - wanted[1]
        if meta.source_subject_bbox is not None and meta.reference_size is not None:
            source = meta.source_subject_bbox
            rw, rh = meta.reference_size
            sw, sh = (source[2]-source[0])*rw, (source[3]-source[1])*rh
            factor, _, _ = contain_subject((sw, sh), wanted, (config.target_width, config.target_height))
            expected_height = sh*factor/config.target_height
            if expected_height < wanted[3]-wanted[1]-1e-8:
                messages.append("Subject height is limited by contain scaling within the target envelope.")
            # Compare width in pixels with ratio-preserving width; permit raster rounding.
            ratio_error_px = abs((actual[2]-actual[0])*config.target_width -
                                 (actual[3]-actual[1])*config.target_height*sw/sh)
            if ratio_error_px > 2 + 2*sw/sh:
                passed = False
                messages.append("Subject aspect ratio is distorted.")
        else:
            passed = False
            messages.append("Source subject geometry is missing; contain scale cannot be verified.")
        scale = abs(expected_height - (actual[3] - actual[1]))
        tolerance_x, tolerance_y = 1/config.target_width, 1/config.target_height
        if (actual[0] < wanted[0]-tolerance_x or actual[2] > wanted[2]+tolerance_x or
                actual[1] < wanted[1]-tolerance_y or actual[3] > wanted[3]+tolerance_y):
            passed = False
            messages.append("Rendered subject exceeds target envelope.")
        if position > config.subject_position_threshold:
            passed = False
            messages.append("Subject position error exceeds threshold.")
        if scale > config.subject_scale_threshold:
            passed = False
            messages.append("Subject scale error exceeds threshold.")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (config.target_width, config.target_height):
                passed = False
                messages.append("Target sketch dimensions do not match configuration.")
    except (OSError, ValueError) as exc:
        passed = False
        messages.append(f"Target sketch cannot be read: {exc}")
    return SketchValidation(passed=passed, subject_position_error=position,
                            subject_scale_error=scale, framing_error=framing, messages=messages)
