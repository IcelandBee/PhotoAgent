"""One target-generation invocation followed by one Guidance invocation."""
from pathlib import Path
from typing import TypedDict, NotRequired
from langgraph.graph import START, END, StateGraph
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.integration.guidance_adapter import build_target_package, to_guide_input
from photography_viewpoint_agent.schemas.handoff import TargetPackage
from photography_viewpoint_agent.schemas.target import TargetState


class IntegratedInput(TypedDict):
    video_path: str
    manual_target_state: TargetState | dict
    session_directory: NotRequired[str]
    session_id: NotRequired[str | None]
    step_id: NotRequired[int | None]
    video_url: NotRequired[str | None]


class IntegratedOutput(TypedDict):
    target_package: dict
    reference_frame: dict
    current_frame: dict
    target_state: dict
    target_sketch: str
    render_meta: dict
    guidance: dict
    output_directory: str


class IntegratedState(IntegratedInput, IntegratedOutput, total=False):
    guide_input: dict


def build_integrated_workflow(config: AgentConfig | None = None, *, backend=None,
                              checkpointer=None, target_graph=None, guide_graph=None,
                              **target_dependencies):
    # Optional dependency: importing/running the standalone generator stays independent.
    from video_guide.core import build_guide_graph

    config = config or AgentConfig()
    guide = guide_graph if guide_graph is not None else build_guide_graph(backend)

    def generate_target(state):
        root = Path(state.get("session_directory", config.work_dir)).resolve()
        generation = root / "generation"
        generation.mkdir(parents=True, exist_ok=False)
        generator = target_graph if target_graph is not None else build_workflow(
            config.model_copy(update={"work_dir": str(generation)}), **target_dependencies)
        generated = generator.invoke({
            "video_path": str(Path(state["video_path"]).resolve()),
            "manual_target_state": TargetState.model_validate(state["manual_target_state"]),
            "error": None,
        })
        package = build_target_package(generated)
        return {"target_package": package.model_dump(mode="json"), "session_directory": str(root)}

    def handoff(state):
        package = TargetPackage.model_validate(state["target_package"])
        return {"guide_input": to_guide_input(
            package, state["session_directory"], session_id=state.get("session_id"),
            step_id=state.get("step_id"), video_url=state.get("video_url"))}

    def guide_once(state, config):
        output = guide.invoke(state["guide_input"], config)
        package = state["target_package"]
        return {"guidance": output["result"], "output_directory": output["output_directory"],
                "reference_frame": package["reference_frame"], "current_frame": package["current_frame"],
                "target_state": package["target_state"], "target_sketch": package["target_sketch_path"],
                "render_meta": package["render_meta"]}

    builder = StateGraph(IntegratedState, input_schema=IntegratedInput, output_schema=IntegratedOutput)
    builder.add_node("generate_target", generate_target)
    builder.add_node("build_handoff", handoff)
    builder.add_node("guide_once", guide_once)
    builder.add_edge(START, "generate_target")
    builder.add_edge("generate_target", "build_handoff")
    builder.add_edge("build_handoff", "guide_once")
    builder.add_edge("guide_once", END)
    return builder.compile(checkpointer=checkpointer)
