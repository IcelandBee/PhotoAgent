from photography_viewpoint_agent.tools.video_utils import load_video_meta


def load_video(state):
    return {"video_meta": load_video_meta(state["video_path"])}
