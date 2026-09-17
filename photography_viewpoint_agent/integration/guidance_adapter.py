from pathlib import Path
from photography_viewpoint_agent.schemas.handoff import TargetPackage
from photography_viewpoint_agent.schemas.validation import SketchValidation


def build_target_package(state) -> TargetPackage:
    if state.get("error"):
        raise ValueError(f"Target generation failed: {state['error']}")
    if state.get("validation") is not None:
        validation = SketchValidation.model_validate(state["validation"])
        if not validation.passed:
            raise ValueError(f"Target sketch validation failed: {validation.messages}")
    package = TargetPackage.model_validate({
        key: state[key] for key in TargetPackage.model_fields if key in state
    })
    return package.model_copy(update={
        "video_path": str(Path(package.video_path).resolve()),
        "target_sketch_path": str(Path(package.target_sketch_path).resolve()),
        "reference_frame": package.reference_frame.model_copy(update={
            "path": str(Path(package.reference_frame.path).resolve())}),
        "current_frame": package.current_frame.model_copy(update={
            "path": str(Path(package.current_frame.path).resolve())}),
    })


def to_guide_input(package: TargetPackage, session_directory, *, current_frame=None,
                   session_id=None, step_id=None, video_url=None) -> dict:
    """Reuse a locked package with a new current frame; never regenerate a target."""
    return {
        "video_path": package.video_path,
        "current_frame": str(Path(current_frame or package.current_frame.path).resolve()),
        "reference_frame": package.reference_frame.path,
        "target_sketch": package.target_sketch_path,
        "target_state": package.target_state.model_dump(mode="json"),
        "render_meta": package.render_meta.model_dump(mode="json"),
        "session_directory": str(Path(session_directory).resolve()),
        "session_id": session_id,
        "step_id": step_id,
        "video_url": video_url,
    }
