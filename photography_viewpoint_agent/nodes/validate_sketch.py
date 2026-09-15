from photography_viewpoint_agent.tools.sketch_validator import validate_sketch as validate


def validate_sketch(state, config):
    return {"validation": validate(state["target_state"], state["render_meta"],
                                   state["target_sketch_path"], config)}
