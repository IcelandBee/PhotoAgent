import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import pytest
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from video_guide.core import build_guide_graph, GuideGraphInput, GuideGraphOutput
from video_guide.core.graph import transient_model_error
from .conftest import MockBackend

@pytest.mark.parametrize("phase,actor,action", [("navigation", "camera", "move_left"), ("composition", "lens", "zoom_out"), ("reached", "camera", "pan_left"), ("uncertain", "camera", "pan_left")])
def test_phases(payload, phase, actor, action):
    engine = MockBackend(phase, actor, action)
    updates = list(build_guide_graph(engine).stream(payload, stream_mode="updates"))
    json.dumps(updates)
    assert [k for u in updates for k in u] == ["validate_input", "prepare_inputs", "analyze_alignment", "persist_result"]
    folder = Path(updates[-1]["persist_result"]["output_directory"])
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert result["phase"] == phase
    assert result["reference_reached"] == (phase in ("composition", "reached"))
    assert result["target_reached"] == (phase == "reached")
    assert not (folder / "crop.png").exists()
    assert engine.calls == 1

@pytest.mark.parametrize("case", ["framing", "reposition", "expansion", "depth_3d", "depth_mesh"])
def test_target_intents_preserved(payload, case):
    actor, action = "lens", "zoom_out"
    if case == "reposition":
        payload["target_state"]["subject"] = {"mode": "reposition", "bbox": [.1, .2, .4, .9]}
        actor, action = "subject", "subject_left"
    if case == "expansion":
        payload["target_state"]["framing"]["reference_viewport"] = [-.2, -.1, 1.2, 1.1]
        payload["render_meta"].update(padding=[10, 10, 10, 10], rendered_viewport=[-.2, -.1, 1.2, 1.1])
        Image.new("RGB", (150, 100), "navy").save(payload["target_sketch"])
    if case.startswith("depth"):
        payload["target_state"]["viewpoint"] = {"mode": case, "translation_x": .1}
        payload["render_meta"]["viewpoint_warp"] = {"mode": case, "metadata": {"holes": .12}}
        actor, action = "camera", "move_right"
    engine = MockBackend(actor=actor, action=action)
    result = build_guide_graph(engine).invoke(payload)["result"]
    assert result["actions"][0]["actor"] == actor
    assert engine.received == (payload["target_state"], payload["render_meta"])
    assert "crop_box" not in result


def test_session_reuses_target_and_keeps_steps(payload):
    graph = build_guide_graph(MockBackend())
    first = graph.invoke(payload)
    target = Path(payload["session_directory"]) / "target"
    original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}
    Image.new("RGB", (100, 80), "red").save(payload["current_frame"])
    second = graph.invoke({**payload, "step_id": 2})
    assert first["output_directory"] != second["output_directory"]
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()} == original
    for output in (first, second):
        folder = Path(output["output_directory"])
        assert not list(folder.glob("*.avi"))
        assert not list(folder.glob("reference*"))
        assert not list(folder.glob("target*"))
    with pytest.raises(ValueError, match="already exists"):
        graph.invoke(payload)
    payload["target_state"]["framing"]["reference_viewport"] = [-.1, 0, 1, 1]
    with pytest.raises(ValueError, match="locked"):
        graph.invoke({**payload, "step_id": 3})


def test_subgraph_and_checkpoint_recovery(payload):
    engine = MockBackend()
    graph = build_guide_graph(engine, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "recover"}}
    import video_guide.core.graph as module
    original = module.write_json
    def fail_result(path, value):
        if path.name == "result.json":
            raise OSError("disk temporarily unavailable")
        original(path, value)
    with patch.object(module, "write_json", side_effect=fail_result), pytest.raises(OSError):
        graph.invoke(payload, config)
    assert graph.get_state(config).next == ("persist_result",)
    assert engine.calls == 1
    result = graph.invoke(None, config)
    assert engine.calls == 1
    graph.update_state(config, {}, as_node="analyze_alignment")
    assert graph.invoke(None, config)["output_directory"] == result["output_directory"]
    class Parent(GuideGraphInput, GuideGraphOutput):
        pass
    parent = StateGraph(Parent)
    parent.add_node("guide", build_guide_graph(MockBackend()))
    parent.add_edge(START, "guide")
    parent.add_edge("guide", END)
    assert parent.compile(checkpointer=InMemorySaver()).invoke({**payload, "step_id": 2}, {"configurable": {"thread_id": "parent"}})["result"]["phase"] == "composition"


def test_retry_and_failure_guards(payload):
    engine = MockBackend(fail_once=True)
    build_guide_graph(engine).invoke(payload)
    assert engine.calls == 2
    for code in (400, 401, 403, 404):
        assert not transient_model_error(HTTPError("https://example.org", code, "error", {}, None))
    assert transient_model_error(HTTPError("https://example.org", 429, "error", {}, None))
    assert not transient_model_error(ValueError("bad JSON"))
    class Broken:
        calls = 0
        def analyze(self, *args):
            self.calls += 1
            raise ValueError("bad schema")
    broken = Broken()
    with pytest.raises(ValueError):
        build_guide_graph(broken).invoke({**payload, "step_id": 2})
    assert broken.calls == 1
    assert not (Path(payload["session_directory"]) / "guidance/step_000002").exists()


def test_auto_step_and_target_tampering(payload):
    graph = build_guide_graph(MockBackend())
    payload.pop("step_id")
    assert graph.invoke(payload)["output_directory"].endswith("step_000001")
    assert graph.invoke(payload)["output_directory"].endswith("step_000002")
    target = Path(payload["session_directory"]) / "target/target_sketch.png"
    Image.new("RGB", (100, 80), "red").save(target)
    with pytest.raises(ValueError, match="modified"):
        graph.invoke(payload)


def test_concurrent_steps_reserve_unique_slots(payload):
    from concurrent.futures import ThreadPoolExecutor
    payload.pop("step_id")
    # Same target, independent graph calls may publish the target concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: build_guide_graph(MockBackend()).invoke(payload), range(2)))
    assert len({r["output_directory"] for r in results}) == 2
    assert {Path(r["output_directory"]).name for r in results} == {"step_000001", "step_000002"}
    root = Path(payload["session_directory"])
    assert not list(root.glob(".target-*"))


def test_invalid_backend_result_not_published(payload):
    class Invalid(MockBackend):
        def analyze(self, *args):
            result = super().analyze(*args)
            result.target_reached = True
            return result
    with pytest.raises(ValueError, match="disagree"):
        build_guide_graph(Invalid()).invoke(payload)
    assert not (Path(payload["session_directory"]) / "guidance/step_000001").exists()
