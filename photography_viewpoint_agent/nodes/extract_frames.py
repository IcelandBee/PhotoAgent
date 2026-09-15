from pathlib import Path
from photography_viewpoint_agent.tools.video_utils import extract_video_frames


def extract_frames(state, config):
    return {"frames": extract_video_frames(state["video_path"], state["video_meta"],
            config.frame_sample_interval, Path(config.work_dir) / "frames")}
