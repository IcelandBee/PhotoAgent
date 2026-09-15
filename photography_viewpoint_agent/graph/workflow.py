from functools import partial, wraps
import logging
from langgraph.graph import END, START, StateGraph
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.state import AgentState
from photography_viewpoint_agent.selectors.base import ReferenceSelector
from photography_viewpoint_agent.selectors.random_selector import RandomReferenceSelector
from photography_viewpoint_agent.planners.base import CompositionPlanner
from photography_viewpoint_agent.planners.manual import ManualCompositionPlanner
from photography_viewpoint_agent.renderer.base import SketchRenderer
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.nodes.load_video import load_video
from photography_viewpoint_agent.nodes.extract_frames import extract_frames
from photography_viewpoint_agent.nodes.select_reference import select_reference_frame
from photography_viewpoint_agent.nodes.get_current_frame import get_current_frame
from photography_viewpoint_agent.nodes.composition_planner import composition_planner
from photography_viewpoint_agent.nodes.render_sketch import render_target_sketch
from photography_viewpoint_agent.nodes.validate_sketch import validate_sketch

logger = logging.getLogger(__name__)


def guarded(name, function):
    def run(state: AgentState):
        if state.get("error"):
            return {}
        try:
            return function(state)
        except Exception as exc:
            logger.exception("Node %s failed", name)
            return {"error": f"{name}: {type(exc).__name__}: {exc}"}
    return run


def build_workflow(config: AgentConfig | None = None, *,
                   selector: ReferenceSelector | None = None,
                   planner: CompositionPlanner | None = None,
                   renderer: SketchRenderer | None = None):
    config = config or AgentConfig()
    selector = selector if selector is not None else RandomReferenceSelector(config.random_seed)
    planner = planner if planner is not None else ManualCompositionPlanner()
    renderer = renderer if renderer is not None else ViewportRenderer(config.target_width, config.target_height)
    nodes = {
        "load_video": load_video,
        "extract_frames": partial(extract_frames, config=config),
        "select_reference_frame": partial(select_reference_frame, selector=selector, config=config),
        "get_current_frame": partial(get_current_frame, config=config),
        "composition_planner": partial(composition_planner, planner=planner),
        "render_target_sketch": partial(render_target_sketch, renderer=renderer, config=config),
        "validate_sketch": partial(validate_sketch, config=config),
    }
    graph = StateGraph(AgentState)
    for name, function in nodes.items():
        graph.add_node(name, guarded(name, function))
    graph.add_edge(START, "load_video")
    graph.add_edge("load_video", "extract_frames")
    graph.add_edge("extract_frames", "select_reference_frame")
    graph.add_edge("extract_frames", "get_current_frame")
    graph.add_edge(["select_reference_frame", "get_current_frame"], "composition_planner")
    graph.add_edge("composition_planner", "render_target_sketch")
    graph.add_edge("render_target_sketch", "validate_sketch")
    graph.add_edge("validate_sketch", END)
    return graph.compile()
