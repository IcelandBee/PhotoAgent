from pathlib import Path


def render_target_sketch(state, renderer, config):
    path, meta = renderer.render(state["reference_frame"], state["target_state"],
                                 Path(config.work_dir) / "target_sketch.jpg")
    return {"target_sketch_path": path, "render_meta": meta}
