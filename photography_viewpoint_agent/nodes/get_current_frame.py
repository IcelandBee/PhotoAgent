from pathlib import Path
import shutil


def get_current_frame(state, config):
    frame = max(state["frames"], key=lambda frame: frame.frame_index)
    shutil.copyfile(frame.path, Path(config.work_dir) / "current_frame.jpg")
    return {"current_frame": frame}
