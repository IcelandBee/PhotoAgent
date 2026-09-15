import math
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.schemas.validation import SketchValidation


def validate_sketch(target: TargetState, meta: RenderMeta, path: str,
                    config: AgentConfig) -> SketchValidation:
    messages = []
    framing = sum(abs(a - b) for a, b in zip(
        target.framing.reference_viewport, meta.rendered_viewport)) / 4
    passed = framing <= config.framing_threshold
    if not passed:
        messages.append("Framing error exceeds threshold.")
    position = scale = None
    if meta.rendered_subject_bbox is None:
        messages.append("Subject rendering is not enabled in V0.1.")
    else:
        wanted, actual = target.subject.bbox, meta.rendered_subject_bbox
        position = math.hypot((wanted[0] + wanted[2] - actual[0] - actual[2]) / 2,
                              (wanted[1] + wanted[3] - actual[1] - actual[3]) / 2)
        scale = abs((wanted[3] - wanted[1]) - (actual[3] - actual[1]))
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
