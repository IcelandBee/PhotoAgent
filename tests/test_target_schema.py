import pytest
from pydantic import ValidationError
from photography_viewpoint_agent.tools.plan_validator import validate_plan
from photography_viewpoint_agent.config.settings import AgentConfig


@pytest.mark.parametrize("field", ["subject", "framing"])
@pytest.mark.parametrize("box", [[0.5, 0, 0.5, 1],
    [0, 1, 1, 0], [0, 0, float("nan"), 1], [0, 0, float("inf"), 1], [0, 1]])
def test_invalid_boxes(target, field, box):
    data = target.model_dump()
    data[field]["bbox" if field == "subject" else "reference_viewport"] = box
    with pytest.raises(ValidationError):
        validate_plan(data)


def test_missing_and_extra_fields(target):
    with pytest.raises(ValidationError):
        validate_plan({"subject": {"bbox": [0, 0, 1, 1]}})
    data = target.model_dump()
    data["recommended_action"] = "move"
    with pytest.raises(ValidationError):
        validate_plan(data)
    assert validate_plan(target) == target


@pytest.mark.parametrize("config", [{"frame_sample_interval": 0}, {"target_width": -1},
    {"framing_threshold": float("nan")}, {"subject_scale_threshold": -0.1}])
def test_invalid_config(config):
    with pytest.raises(ValidationError):
        AgentConfig(**config)


@pytest.mark.parametrize("box", [[-0.1, 0, 1, 1], [0, 0, 1.1, 1],
    [0, -0.2, 1, 1], [-0.2, 0.1, 0.9, 0.9], [-0.2, -0.2, 1.2, 1.2]])
def test_viewport_expansion_does_not_relax_subject(target, box):
    data = target.model_dump()
    data["framing"]["reference_viewport"] = box
    assert validate_plan(data).framing.reference_viewport == tuple(box)
    data["subject"]["bbox"] = box
    with pytest.raises(ValidationError):
        validate_plan(data)


@pytest.mark.parametrize("color", [[-1, 128, 128], [0, 0, 256], [0, 0], [1.5, 0, 0]])
def test_invalid_fill_color(color):
    with pytest.raises(ValidationError):
        AgentConfig(fill_color=color)


def test_large_finite_viewport_and_overflow(target):
    data = target.model_dump()
    data["framing"]["reference_viewport"] = [-3, 0, 1, 1]
    assert validate_plan(data).framing.reference_viewport == (-3, 0, 1, 1)
    data["framing"]["reference_viewport"] = [-1e308, 0, 1e308, 1]
    with pytest.raises(ValidationError):
        validate_plan(data)
