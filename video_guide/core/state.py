"""Only JSON data and file references cross node boundaries."""
from typing import TypedDict, NotRequired

class GuideGraphInput(TypedDict):
    video_path: str
    current_frame: str
    reference_frame: str
    target_sketch: str
    target_state: dict
    render_meta: dict
    session_directory: str
    session_id: NotRequired[str | None]
    step_id: NotRequired[int | None]
    video_url: NotRequired[str | None]

class GuideGraphOutput(TypedDict):
    result: dict
    output_directory: str

class GuideState(GuideGraphInput, GuideGraphOutput, total=False):
    execution_id: str
    staging_directory: str
    saved_inputs: dict[str, str]
