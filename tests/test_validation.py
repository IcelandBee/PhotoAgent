import math
import pytest
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch


def test_metrics_and_failures(frame, target):
    config = AgentConfig(target_width=100, target_height=100)
    meta = RenderMeta(rendered_viewport=target.framing.reference_viewport,
                      rendered_subject_bbox=(0.38, 0.25, 0.62, 0.90))
    result = validate_sketch(target, meta, frame.path, config)
    assert result.subject_position_error == pytest.approx(0.1)
    assert result.subject_scale_error == pytest.approx(0.1)
    assert result.framing_error == 0
    assert not result.passed


def test_missing_subject_and_unreadable_image(frame, target):
    meta = RenderMeta(rendered_viewport=target.framing.reference_viewport)
    config = AgentConfig(target_width=100, target_height=100)
    result = validate_sketch(target, meta, frame.path, config)
    assert not result.passed and result.subject_position_error is None and result.messages
    assert not validate_sketch(target, meta, "missing.jpg", config).passed
    assert not validate_sketch(target, meta, frame.path, AgentConfig()).passed


def test_framing_failure(frame, target):
    meta = RenderMeta(rendered_viewport=(0.2, 0.2, 0.8, 0.8))
    result = validate_sketch(target, meta, frame.path, AgentConfig(target_width=100, target_height=100))
    assert result.framing_error == pytest.approx(0.165)
    assert not result.passed


def test_expanded_viewport_validation(frame, target):
    data = target.model_dump()
    data["framing"]["reference_viewport"] = [-0.2, 0, 1.2, 1]
    target = type(target).model_validate(data)
    result = validate_sketch(target, RenderMeta(rendered_viewport=(-0.2, 0, 1.2, 1)),
                             frame.path, AgentConfig(target_width=100, target_height=100))
    assert result.framing_error == 0  # Negative coordinates alone do not fail framing.
    assert not result.passed  # Subject is now required.
