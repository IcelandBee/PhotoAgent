import pytest
from PIL import Image
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.video import FrameInfo


@pytest.fixture
def target():
    return TargetState.model_validate({"subject": {"bbox": [0.28, 0.15, 0.52, 0.9]},
        "framing": {"reference_viewport": [0.05, 0.02, 0.95, 0.98]}})


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (100, 100), "navy").save(path)
    return FrameInfo(frame_id="frame_000000", path=str(path), frame_index=0,
                     timestamp=0, width=100, height=100)
