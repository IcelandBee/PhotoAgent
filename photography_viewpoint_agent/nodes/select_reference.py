from pathlib import Path
import shutil


def select_reference_frame(state, selector, config):
    frame = selector.select(state["frames"])
    if frame not in state["frames"]:
        raise ValueError("Reference selector must return a candidate frame")
    shutil.copyfile(frame.path, Path(config.work_dir) / "reference_frame.jpg")
    return {"reference_frame": frame}
