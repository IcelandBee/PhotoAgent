import random
import pytest
from photography_viewpoint_agent.selectors.random_selector import RandomReferenceSelector
from photography_viewpoint_agent.selectors.api_selector import APIReferenceSelector


def test_repeatable_and_no_global_rng_changes(frame):
    frames = [frame.model_copy(update={"frame_id": str(i), "frame_index": i}) for i in range(20)]
    selector = RandomReferenceSelector()
    before = random.getstate()
    assert selector.select(frames) == selector.select(frames) == RandomReferenceSelector().select(frames)
    assert random.getstate() == before
    with pytest.raises(ValueError):
        selector.select([])


def test_api_adapter(frame):
    assert APIReferenceSelector(lambda frames: frames[0].frame_id).select([frame]) == frame
    with pytest.raises(ValueError):
        APIReferenceSelector(lambda frames: "unknown").select([frame])
