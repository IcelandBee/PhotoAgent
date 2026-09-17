from pathlib import Path
import pytest
from PIL import Image
from video_guide.core.models import GuideResult, GuidanceAction

TARGET = {"subject": {"mode": "follow_reference"}, "framing": {"reference_viewport": [0, 0, 1, 1]}, "viewpoint": {"mode": "none"}}
META = {"rendered_viewport": [0, 0, 1, 1], "padding": [0, 0, 0, 0]}

class MockBackend:
    def __init__(self, phase="composition", actor="lens", action="zoom_in", fail_once=False):
        self.phase, self.actor, self.action = phase, actor, action
        self.fail_once, self.calls, self.received = fail_once, 0, None
    def analyze(self, current, reference, sketch, video, target_state, render_meta, video_url=None):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise TimeoutError("temporary")
        self.received = (target_state, render_meta)
        return GuideResult(self.phase, self.phase in ("composition", "reached"), self.phase == "reached",
                           [] if self.phase in ("reached", "uncertain") else [GuidanceAction(self.actor, self.action, "small")],
                           "调整拍摄位置", .8)

@pytest.fixture
def payload(tmp_path):
    video = tmp_path / "video.avi"
    video.write_bytes(b"offline backend video fixture")
    for name in ("current", "reference", "sketch"):
        Image.new("RGB", (100, 80), "green").save(tmp_path / (name + ".png"))
    import copy
    return {"video_path": str(video), "current_frame": str(tmp_path / "current.png"),
            "reference_frame": str(tmp_path / "reference.png"), "target_sketch": str(tmp_path / "sketch.png"),
            "target_state": copy.deepcopy(TARGET), "render_meta": copy.deepcopy(META),
            "session_directory": str(tmp_path / "session"), "session_id": "offline", "step_id": 1}
