from photography_viewpoint_agent.schemas.target import TargetState


def validate_plan(plan: TargetState | dict) -> TargetState:
    data = plan.model_dump() if isinstance(plan, TargetState) else plan
    return TargetState.model_validate(data)
