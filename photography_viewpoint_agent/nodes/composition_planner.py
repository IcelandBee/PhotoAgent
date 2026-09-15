from photography_viewpoint_agent.tools.plan_validator import validate_plan


def composition_planner(state, planner):
    target = planner.plan(state["current_frame"], state["reference_frame"],
                          state.get("manual_target_state"))
    return {"target_state": validate_plan(target)}
