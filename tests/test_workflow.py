import json
import cv2
import numpy as np
import pytest
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.tools.video_utils import load_video_meta, extract_video_frames
from photography_viewpoint_agent.app import main


def video(path, count):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(count):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize("count,expected", [(1, [0]), (6, [0, 3, 5]), (7, [0, 3, 6])])
def test_last_frame(tmp_path, count, expected):
    path = tmp_path / "input.avi"
    video(path, count)
    meta = load_video_meta(str(path))
    frames = extract_video_frames(str(path), meta, 3, tmp_path / "frames")
    assert [f.frame_index for f in frames] == expected
    assert frames[-1].timestamp == (count - 1) / 10
    assert abs(float(cv2.imread(frames[-1].path).mean()) - (count - 1) * 10) < 3


def test_graph_join_and_cli(tmp_path, target):
    path = tmp_path / "input.avi"
    video(path, 8)
    class SpyPlanner:
        calls = 0
        def plan(self, current_frame, reference_frame, manual_target_state=None):
            self.calls += 1
            assert current_frame.frame_index == 7
            assert reference_frame.frame_index in [0, 3, 6, 7]
            return target
    planner = SpyPlanner()
    graph = build_workflow(AgentConfig(work_dir=str(tmp_path / "graph"), frame_sample_interval=3), planner=planner)
    result = graph.invoke({"video_path": str(path)})
    assert result.get("error") is None
    assert result["validation"].passed and planner.calls == 1
    assert "validate_plan" not in graph.get_graph().nodes
    manual = tmp_path / "target.json"
    manual.write_text(target.model_dump_json(), encoding="utf-8")
    args = ["--video", str(path), "--target-state", str(manual), "--work-dir", str(tmp_path / "cli")]
    assert main(args) == 0
    data = json.loads((tmp_path / "cli/result.json").read_text(encoding="utf-8"))
    assert data["current_frame"]["frame_index"] == 7
    assert data["validation"]["passed"]
    assert main(args) == 1  # Existing artifacts must remain intact.


def test_error_state(tmp_path):
    result = build_workflow(AgentConfig(work_dir=str(tmp_path))).invoke({"video_path": "missing.mp4"})
    assert "load_video" in result["error"]
    assert "target_sketch_path" not in result


def test_both_parallel_errors(tmp_path, target, monkeypatch):
    import shutil
    path = tmp_path / "input.avi"
    video(path, 2)
    def fail(*args):
        raise OSError("copy failure")
    monkeypatch.setattr(shutil, "copyfile", fail)
    result = build_workflow(AgentConfig(work_dir=str(tmp_path / "run"))).invoke(
        {"video_path": str(path), "manual_target_state": target})
    assert "select_reference_frame" in result["error"]
    assert "get_current_frame" in result["error"]
    assert "target_state" not in result


def test_expansion_cli(tmp_path, target):
    path = tmp_path / "input.avi"
    video(path, 2)
    data = target.model_dump()
    data["framing"]["reference_viewport"] = [-0.25, -0.25, 1.25, 1.25]
    manual = tmp_path / "target.json"
    manual.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "expanded"
    assert main(["--video", str(path), "--target-state", str(manual), "--work-dir", str(output),
                 "--target-width", "120", "--target-height", "120", "--fill-color", "50", "50", "50"]) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["render_meta"]["padding"] == [20, 20, 20, 20]
    assert result["validation"]["passed"]
    assert np.all(cv2.imread(str(output / "target_sketch.jpg"))[5, 5] == 50)
