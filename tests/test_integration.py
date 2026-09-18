import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.integration.guidance_adapter import build_target_package, to_guide_input
from photography_viewpoint_agent.schemas.handoff import TargetPackage
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.schemas.target import TargetState


@pytest.fixture
def package_state(frame, tmp_path):
    video = tmp_path / "video.avi"
    video.write_bytes(b"mock video")
    return {"video_path": str(video), "current_frame": frame, "reference_frame": frame,
            "target_state": TargetState.model_validate({"subject": {"mode": "follow_reference"},
                "framing": {"reference_viewport": [0, 0, 1, 1]}}),
            "target_sketch_path": frame.path,
            "render_meta": RenderMeta(rendered_viewport=(0, 0, 1, 1))}


def test_handoff_roundtrip_and_new_current(package_state, tmp_path):
    package = build_target_package({**package_state, "frames": [], "private_internal_field": "not exported"})
    assert set(package.model_dump()) == set(TargetPackage.model_fields)
    assert TargetPackage.model_validate_json(package.model_dump_json()) == package
    inputs = to_guide_input(package, tmp_path / "session", current_frame=tmp_path / "new.png", session_id="test", step_id=2)
    assert inputs["current_frame"].endswith("new.png")
    assert inputs["reference_frame"] == package.reference_frame.path
    assert inputs["target_sketch"] == package.target_sketch_path
    assert inputs["target_state"] == package.target_state.model_dump(mode="json")
    assert inputs["render_meta"] == package.render_meta.model_dump(mode="json")
    assert package.current_frame.path == package_state["current_frame"].path


@pytest.mark.parametrize("change", [{"error": "renderer failed"}, {"validation": {"passed": False, "messages": ["bad render"]}}, {"render_meta": None}])
def test_handoff_rejects_failed_generation(package_state, change):
    with pytest.raises(ValueError):
        build_target_package({**package_state, **change})


@pytest.mark.parametrize("mode", ["none", "homography", "depth_3d", "depth_mesh"])
def test_full_viewpoint_metadata_survives(package_state, tmp_path, mode):
    target = package_state["target_state"].model_dump(mode="json")
    target["subject"] = {"mode": "reposition", "bbox": [.1, .2, .5, .9]}
    target["framing"]["reference_viewport"] = [-.2, -.1, 1.2, 1.1]
    target["viewpoint"]["mode"] = "none" if mode == "none" else "rotation"
    target["viewpoint"]["yaw_deg"] = 0 if mode == "none" else 5
    package_state["target_state"] = TargetState.model_validate(target)
    package_state["render_meta"] = RenderMeta(rendered_viewport=(-.2, -.1, 1.2, 1.1),
        padding=(10, 10, 10, 10), subject_mode="reposition", viewpoint_backend=mode, viewpoint_warp={"mode": mode, "custom": [1, 2, 3]})
    payload = to_guide_input(build_target_package(package_state), tmp_path / "session")
    assert payload["target_state"] == target
    assert payload["render_meta"]["viewpoint_warp"]["custom"] == [1, 2, 3]


def test_real_generation_to_guidance_and_locked_second_step(tmp_path):
    from video_guide.core import build_guide_graph
    from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow

    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        y, x = np.indices((48, 64))
        pixels = np.stack((x * 3, y * 4, x + y), axis=-1).astype(np.uint8)
        for _ in range(5):
            writer.write(pixels)
    finally:
        writer.release()
    session = tmp_path / "session"
    result = build_integrated_workflow(AgentConfig(target_width=64, target_height=48)).invoke({
        "video_path": str(video), "session_directory": str(session),
        "manual_target_state": {"subject": {"mode": "follow_reference"}, "framing": {"reference_viewport": [0, 0, 1, 1]}},
        "session_id": "e2e", "step_id": 1,
    })
    json.dumps(result)
    assert result["guidance"]["phase"] == "reached"
    assert Path(result["target_sketch"]).is_file()
    package = TargetPackage.model_validate(result["target_package"])
    target_files = {p.name: p.stat().st_mtime_ns for p in (session / "target").iterdir()}
    new_current = tmp_path / "new.png"
    Image.new("RGB", (64, 48), "white").save(new_current)
    second = build_guide_graph().invoke(to_guide_input(package, session,
        current_frame=new_current, session_id="e2e", step_id=2))
    assert second["result"]["phase"] == "uncertain"
    assert target_files == {p.name: p.stat().st_mtime_ns for p in (session / "target").iterdir()}
    assert len(list((session / "guidance").glob("step_*"))) == 2


def test_parent_failure_does_not_call_guide(package_state, tmp_path):
    from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow
    class Failed:
        def invoke(self, state):
            return {"error": "intentional failure"}
    class Never:
        def invoke(self, *args):
            pytest.fail("Guidance must not run on failed generation")
    with pytest.raises(ValueError, match="intentional failure"):
        build_integrated_workflow(target_graph=Failed(), guide_graph=Never()).invoke({
            "video_path": package_state["video_path"], "manual_target_state": package_state["target_state"],
            "session_directory": str(tmp_path / "session")})


def test_parent_checkpoint_resumes_guidance_without_regenerating(package_state, tmp_path, monkeypatch):
    import video_guide.core.graph as guide_module
    from langgraph.checkpoint.memory import InMemorySaver
    from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow
    from video_guide.core.models import GuideResult

    class Generator:
        calls = 0
        def invoke(self, state):
            self.calls += 1
            return package_state
    class Backend:
        calls = 0
        def analyze(self, *args):
            self.calls += 1
            return GuideResult("reached", True, True, [], "可以拍摄", .9)
    generator, backend = Generator(), Backend()
    graph = build_integrated_workflow(target_graph=generator, backend=backend, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "integrated-recovery"}}
    original = guide_module.write_json
    def fail_result(path, value):
        if path.name == "result.json":
            raise OSError("temporary disk failure")
        return original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(guide_module, "write_json", fail_result)
        with pytest.raises(OSError):
            graph.invoke({"video_path": package_state["video_path"],
                          "manual_target_state": package_state["target_state"],
                          "session_directory": str(tmp_path / "session")}, config)
    assert generator.calls == backend.calls == 1
    result = graph.invoke(None, config)
    assert result["guidance"]["phase"] == "reached"
    assert generator.calls == backend.calls == 1


@pytest.mark.parametrize('field', ['translation_x', 'translation_y', 'translation_z'])
def test_handoff_rejects_legacy_translation(package_state, tmp_path, field):
    package = build_target_package(package_state)
    data = package_state['target_state'].model_dump()
    data['viewpoint'][field] = 0
    with pytest.raises(ValueError, match=field):
        build_target_package({**package_state, 'target_state': data})
    invalid = package.model_copy(update={'target_state': data})
    with pytest.raises(ValueError, match=field):
        to_guide_input(invalid, tmp_path / 'session')
